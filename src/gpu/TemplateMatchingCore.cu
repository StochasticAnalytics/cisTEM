#include "gpu_core_headers.h"
#include "TemplateMatchingCore.h"
#include "../ext/FastFFT/include/FastFFT.cuh"

#define CHECK_FOR_BAD_FLOATS

#define DO_HISTOGRAM true

__global__ void MipPixelWiseKernel(__half* correlation_output, __half2* my_peaks, const int numel,
                                   __half psi, __half theta, __half phi, __half2* my_stats, __half2* my_new_peaks, const bool use_fast_fft);

TemplateMatchingCore::TemplateMatchingCore( ){

};

TemplateMatchingCore::TemplateMatchingCore(int number_of_jobs) {

    Init(number_of_jobs);
};

TemplateMatchingCore::~TemplateMatchingCore( ){

        // FIXME
        //	if (is_allocated_cummulative_histogram)
        //	{
        //		cudaErr(cudaFree((void *)cummulative_histogram));
        //		cudaErr(cudaFreeHost((void *)h_cummulative_histogram));
        //	}

};

void TemplateMatchingCore::Init(int number_of_jobs) {
    this->nThreads                        = 1;
    this->number_of_jobs_per_image_in_gui = 1;
    this->nGPUs                           = 1;
};

void TemplateMatchingCore::Init(MyApp*           parent_pointer,
                                Image&           template_reconstruction,
                                Image&           input_image,
                                Image&           current_projection,
                                float            pixel_size_search_range,
                                float            pixel_size_step,
                                float            pixel_size,
                                float            defocus_search_range,
                                float            defocus_step,
                                float            defocus1,
                                float            defocus2,
                                float            psi_max,
                                float            psi_start,
                                float            psi_step,
                                AnglesAndShifts& angles,
                                EulerSearch&     global_euler_search,
                                float            histogram_min_scaled,
                                float            histogram_step_scaled,
                                int              histogram_number_of_bins,
                                int              max_padding,
                                int              first_search_position,
                                int              last_search_position,
                                ProgressBar*     my_progress,
                                long             total_correlation_positions,
                                bool             is_running_locally,
                                bool             use_fast_fft,
                                int              number_of_global_search_images_to_save)

{

    this->first_search_position          = first_search_position;
    this->last_search_position           = last_search_position;
    this->angles                         = angles;
    this->global_euler_search            = global_euler_search;
    this->use_fast_fft                   = use_fast_fft;
    this->n_global_search_images_to_save = number_of_global_search_images_to_save;

    this->psi_start = psi_start;
    this->psi_step  = psi_step;
    this->psi_max   = psi_max;

    // It seems that I need a copy for these - 1) confirm, 2) if already copying, maybe put straight into pinned mem with cudaHostMalloc
    this->template_reconstruction.CopyFrom(&template_reconstruction);
    this->input_image.CopyFrom(&input_image);
    this->current_projection.CopyFrom(&current_projection);

    d_input_image.Init(this->input_image);
    d_input_image.CopyHostToDevice(input_image);

    d_current_projection.Init(this->current_projection);
    d_padded_reference.Allocate(d_input_image.dims.x, d_input_image.dims.y, 1, true);
    d_max_intensity_projection.Allocate(d_input_image.dims.x, d_input_image.dims.y, number_of_global_search_images_to_save, true);
    d_best_psi.Allocate(d_input_image.dims.x, d_input_image.dims.y, number_of_global_search_images_to_save, true);
    d_best_theta.Allocate(d_input_image.dims.x, d_input_image.dims.y, number_of_global_search_images_to_save, true);
    d_best_phi.Allocate(d_input_image.dims.x, d_input_image.dims.y, number_of_global_search_images_to_save, true);

    d_sum1.Allocate(d_input_image.dims.x, d_input_image.dims.y, 1, true);
    d_sumSq1.Allocate(d_input_image.dims.x, d_input_image.dims.y, 1, true);
    d_sum2.Allocate(d_input_image.dims.x, d_input_image.dims.y, 1, true);
    d_sumSq2.Allocate(d_input_image.dims.x, d_input_image.dims.y, 1, true);
    d_sum3.Allocate(d_input_image.dims.x, d_input_image.dims.y, 1, true);
    d_sumSq3.Allocate(d_input_image.dims.x, d_input_image.dims.y, 1, true);

    wxPrintf("Setting up the histogram\n\n");
    histogram.Init(histogram_number_of_bins, histogram_min_scaled, histogram_step_scaled);
    if ( max_padding > 2 ) {
        histogram.max_padding = max_padding;
    }

    this->my_progress                 = my_progress;
    this->total_correlation_positions = total_correlation_positions;
    this->is_running_locally          = is_running_locally;

    this->parent_pointer = parent_pointer;

    // For now we are only working on the inner loop, so no need to track best_defocus and best_pixel_size

    // At the outset these are all empty cpu images, so don't xfer, just allocate on gpuDev

    // Transfer the input image_memory_should_not_be_deallocated

    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));
};

void TemplateMatchingCore::RunInnerLoop(Image& projection_filter, float c_pixel, float c_defocus, int threadIDX, long& current_correlation_position) {

    // Make sure we are starting with zeros
    d_max_intensity_projection.Zeros( );
    d_best_psi.Zeros( );
    d_best_phi.Zeros( );
    d_best_theta.Zeros( );
    d_padded_reference.Zeros( );

    d_sum1.Zeros( );
    d_sumSq1.Zeros( );
    d_sum2.Zeros( );
    d_sumSq2.Zeros( );
    d_sum3.Zeros( );
    d_sumSq3.Zeros( );

    this->c_defocus                 = c_defocus;
    this->c_pixel                   = c_pixel;
    total_number_of_cccs_calculated = 0;

    // FIXME: FastFFT works on transposed 2D xforms so for testing. Really we should just do the original transform in match template with this.
    // TODO: need a scale and a scale and shift functor
    if ( use_fast_fft ) {
        std::cerr << "Using FastFFT\n";
        // FastFFT pads from the upper left corner, so we need to shift the image so the origins coinicide
        d_input_image.PhaseShift(-(d_input_image.physical_address_of_box_center.x - d_current_projection.physical_address_of_box_center.x),
                                 -(d_input_image.physical_address_of_box_center.y - d_current_projection.physical_address_of_box_center.y),
                                 0);

        d_input_image.BackwardFFT( );

        FastFFT::FourierTransformer<float, float, float2, 2> FT;

        // TODO: overload that takes and short4's int4's instead of the individual values
        FT.SetForwardFFTPlan(input_image.logical_x_dimension, input_image.logical_y_dimension, d_input_image.logical_z_dimension, d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);
        FT.SetInverseFFTPlan(d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, d_input_image.dims.x, d_input_image.dims.y, d_input_image.dims.z, true);

        FT.FwdFFT(d_input_image.real_values);

        // We've done a round trip iFFT/FFT since the input image was normalized to STD 1.0, so re-normalize by 1/n
        d_input_image.is_in_real_space = false;
        // d_input_image.MultiplyByConstant(sqrtf(1.f / d_input_image.number_of_real_space_pixels));
        d_input_image.MultiplyByConstant((1.f / d_input_image.number_of_real_space_pixels));
    }

    d_input_image.CopyFP32toFP16buffer(false);
    d_padded_reference.CopyFP32toFP16buffer(false);

    cudaErr(cudaMalloc((void**)&my_peaks, sizeof(__half2) * d_input_image.real_memory_allocated));
    cudaErr(cudaMalloc((void**)&my_new_peaks, sizeof(__half2) * d_input_image.real_memory_allocated));
    cudaErr(cudaMalloc((void**)&my_stats, sizeof(__half2) * d_input_image.real_memory_allocated));
    cudaErr(cudaMemset(my_peaks, 0, sizeof(__half2) * d_input_image.real_memory_allocated));
    cudaErr(cudaMemset(my_new_peaks, 0, sizeof(__half2) * d_input_image.real_memory_allocated));
    if ( n_global_search_images_to_save > 1 ) {
        cudaErr(cudaMalloc((void**)&secondary_peaks, sizeof(__half) * d_input_image.real_memory_allocated * n_global_search_images_to_save * 4));
        cudaErr(cudaMemset(secondary_peaks, 0, sizeof(__half) * d_input_image.real_memory_allocated * n_global_search_images_to_save * 4));
    }
    //	cudaErr(cudaMemset(my_stats,0,sizeof(Peaks)*d_input_image.real_memory_allocated));

    cudaEvent_t projection_is_free_Event, gpu_work_is_done_Event;
    cudaErr(cudaEventCreateWithFlags(&projection_is_free_Event, cudaEventDisableTiming));
    cudaErr(cudaEventCreateWithFlags(&gpu_work_is_done_Event, cudaEventDisableTiming));

    FastFFT::FourierTransformer<float, __half, __half2, 2>                           FT;
    FastFFT::KernelFunction::my_functor<float, 0, FastFFT::KernelFunction::NOOP>     noop;
    FastFFT::KernelFunction::my_functor<float, 4, FastFFT::KernelFunction::CONJ_MUL> conj_mul;

    if ( use_fast_fft ) {
        // TODO: overload that takes and short4's int4's instead of the individual values
        FT.SetForwardFFTPlan(current_projection.logical_x_dimension, current_projection.logical_y_dimension, current_projection.logical_z_dimension, d_padded_reference.dims.x, d_padded_reference.dims.y, d_padded_reference.dims.z, true);
        FT.SetInverseFFTPlan(d_padded_reference.dims.x, d_padded_reference.dims.y, d_padded_reference.dims.z, d_padded_reference.dims.x, d_padded_reference.dims.y, d_padded_reference.dims.z, true);
    }

    int   ccc_counter = 0;
    int   current_search_position;
    float average_on_edge;
    float average_of_reals;
    float temp_float;
    bool  second_loop = false;
    int   thisDevice;
    cudaGetDevice(&thisDevice);
    wxPrintf("Thread %d is running on device %d\n", threadIDX, thisDevice);

    //	cudaErr(cudaFuncSetCacheConfig(SumPixelWiseKernel, cudaFuncCachePreferL1));

    //	bool make_graph = true;
    //	bool first_loop_complete = false;

    for ( current_search_position = first_search_position; current_search_position <= last_search_position; current_search_position++ ) {

        if ( current_search_position % 10 == 0 ) {
            wxPrintf("Starting position %d/ %d\n", current_search_position, last_search_position);
        }

        for ( float current_psi = psi_start; current_psi <= psi_max; current_psi += psi_step ) {

            angles.Init(global_euler_search.list_of_search_parameters[current_search_position][0], global_euler_search.list_of_search_parameters[current_search_position][1], current_psi, 0.0, 0.0);
            //			current_projection.SetToConstant(0.0f); // This also sets the FFT padding to zero
            template_reconstruction.ExtractSlice(current_projection, angles, 1.0f, false);
            current_projection.complex_values[0] = 0.0f + I * 0.0f;
            current_projection.SwapRealSpaceQuadrants( );
            current_projection.MultiplyPixelWise(projection_filter);
            current_projection.BackwardFFT( );
            average_on_edge = current_projection.ReturnAverageOfRealValuesOnEdges( );

            average_of_reals = current_projection.ReturnAverageOfRealValues( ) - average_on_edge;

            // Make sure the device has moved on to the padded projection
            cudaErr(cudaStreamWaitEvent(cudaStreamPerThread, projection_is_free_Event, cudaEventWaitDefault));

            // FIXME: For current TM test hack, need to leave these ops on the CPU until sorting out
            // association with non-owned GPU memory.
            //// TO THE GPU ////
            d_current_projection.CopyHostToDevice(current_projection);
            if ( ! use_fast_fft ) {
                d_current_projection.AddConstant(-average_on_edge);
            }
            // The average in the full padded image will be different;
            average_of_reals *= ((float)d_current_projection.number_of_real_space_pixels / (float)d_padded_reference.number_of_real_space_pixels);

            if ( use_fast_fft ) {
                // FIXME:
                float scale_factor = rsqrtf(d_current_projection.ReturnSumOfSquares( ) / (float)d_padded_reference.number_of_real_space_pixels - (average_of_reals * average_of_reals));
                scale_factor /= powf((float)d_current_projection.number_of_real_space_pixels, 1.5);
                // d_current_projection.MultiplyByConstant(scale_factor);

                d_current_projection.CopyFP32toFP16bufferAndScale(scale_factor);
                cudaErr(cudaEventRecord(projection_is_free_Event, cudaStreamPerThread));
                FT.FwdImageInvFFT(d_current_projection.real_values_fp16, (__half2*)d_input_image.complex_values_fp16, d_padded_reference.real_values_fp16, noop, conj_mul, noop);
                // d_padded_reference.CopyFP32toFP16buffer(false);
            }
            else {
                d_current_projection.MultiplyByConstant(rsqrtf(d_current_projection.ReturnSumOfSquares( ) / (float)d_padded_reference.number_of_real_space_pixels - (average_of_reals * average_of_reals)));
                d_current_projection.ClipInto(&d_padded_reference, 0, false, 0, 0, 0, 0);
                cudaErr(cudaEventRecord(projection_is_free_Event, cudaStreamPerThread));

                d_padded_reference.ForwardFFT(false);
                d_padded_reference.BackwardFFTAfterComplexConjMul(d_input_image.complex_values_fp16, true);
            }

            if ( DO_HISTOGRAM ) {
                if ( ! histogram.is_allocated_histogram ) {
                    d_padded_reference.NppInit( );
                    histogram.BufferInit(d_padded_reference.npp_ROI);
                }
                histogram.AddToHistogram(d_padded_reference);
            }

            this->MipPixelWise(__float2half_rn(current_psi), __float2half_rn(global_euler_search.list_of_search_parameters[current_search_position][1]),
                               __float2half_rn(global_euler_search.list_of_search_parameters[current_search_position][0]));

            ccc_counter++;
            total_number_of_cccs_calculated++;

            if ( ccc_counter % 10 == 0 ) {
                this->AccumulateSums(my_stats, d_sum1, d_sumSq1);
            }

            if ( ccc_counter % 100 == 0 ) {

                d_sum2.AddImage(d_sum1);
                d_sum1.Zeros( );

                d_sumSq2.AddImage(d_sumSq1);
                d_sumSq1.Zeros( );
            }

            if ( ccc_counter % 10000 == 0 ) {

                d_sum3.AddImage(d_sum2);
                d_sum2.Zeros( );

                d_sumSq3.AddImage(d_sumSq2);
                d_sumSq2.Zeros( );
            }

            current_projection.is_in_real_space = false;
            d_padded_reference.is_in_real_space = true;
            cudaEventRecord(gpu_work_is_done_Event, cudaStreamPerThread);

            //			first_loop_complete = true;

            if ( is_running_locally ) {
                if ( ReturnThreadNumberOfCurrentThread( ) == 0 ) {
                    current_correlation_position++;
                    if ( current_correlation_position > total_correlation_positions )
                        current_correlation_position = total_correlation_positions;
                    my_progress->Update(current_correlation_position);
                }
            }
            else {
                temp_float             = current_correlation_position;
                JobResult* temp_result = new JobResult;
                temp_result->SetResult(1, &temp_float);
                parent_pointer->AddJobToResultQueue(temp_result);
            }

        } // loop over psi angles

        // The current goal is to have only one peak per search position.
        if ( n_global_search_images_to_save > 1 )
            UpdateSecondaryPeaks( );

    } // end of outer loop euler sphere position

    wxPrintf("\t\t\ntotal number %d\n", ccc_counter);

    cudaStreamWaitEvent(cudaStreamPerThread, gpu_work_is_done_Event, 0);

    this->AccumulateSums(my_stats, d_sum1, d_sumSq1);

    d_sum2.AddImage(d_sum1);
    d_sumSq2.AddImage(d_sumSq1);

    d_sum3.AddImage(d_sum2);
    d_sumSq3.AddImage(d_sumSq2);

    this->MipToImage( );

    MyAssertTrue(histogram.is_allocated_histogram, "Trying to accumulate a histogram that has not been initialized!")
            histogram.Accumulate(d_padded_reference);

    cudaErr(cudaStreamSynchronize(cudaStreamPerThread));

    cudaErr(cudaFree(my_peaks));
    cudaErr(cudaFree(my_stats));
    cudaErr(cudaFree(my_new_peaks));
}

void TemplateMatchingCore::MipPixelWise(__half psi, __half theta, __half phi) {

    precheck;
    // N
    d_padded_reference.ReturnLaunchParametersLimitSMs(5.f, 1024);

    MipPixelWiseKernel<<<d_padded_reference.gridDims, d_padded_reference.threadsPerBlock, 0, cudaStreamPerThread>>>((__half*)d_padded_reference.real_values_16f, my_peaks,
                                                                                                                    (int)d_padded_reference.real_memory_allocated,
                                                                                                                    psi, theta, phi, my_stats, my_new_peaks, use_fast_fft);
    postcheck;
}

__global__ void MipPixelWiseKernel(__half* correlation_output, __half2* my_peaks, const int numel,
                                   __half psi, __half theta, __half phi, __half2* my_stats, __half2* my_new_peaks, const bool use_fast_fft) {

    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x ) {

#ifdef CHECK_FOR_BAD_FLOATS
        __half half_val = correlation_output[i];
        if ( half_val == CUDART_INF_FP16 )
            half_val = half(-225.0f);
        else if ( half_val == CUDART_NAN_FP16 )
            half_val = half(-226.0f);
#else
        const __half half_val = correlation_output[i];
#endif

        const __half2 input  = __half2half2(half_val * __half(10000.0));
        const __half2 mulVal = __halves2half2((__half)1.0, half_val);
        //    	my_stats[i].sum = __hadd(my_stats[i].sum, half_val);
        //    	my_stats[i].sq_sum = __hfma(__half(1000.)*half_val,half_val,my_stats[i].sq_sum);
        my_stats[i] = __hfma2(input, mulVal, my_stats[i]);

        // We are always reading the mip value
        // And likely often writing both that value and the updated angles.
        // If we had some minimum threshold, (5?) we could skip all these memory ops while still collecting
        // the full set of image statistics as well as histogram values.
        if ( half_val > __low2half(my_peaks[i]) ) {
            //				tmp_peak.mip = half_val;
            my_peaks[i]     = __halves2half2(half_val, psi);
            my_new_peaks[i] = __halves2half2(theta, phi);
        }
    }
    //
}

__global__ void
UpdateSecondaryPeaksKernel(__half*   secondary_peaks,
                           __half2*  my_peaks,
                           __half2*  my_new_peaks,
                           const int NY,
                           const int NX) {

    //	When returning more than one search result, the peaks are stored in a 3d array,
    // numel * n_peaks * 4 (mip, psi, theta, phi)
    int best_index = 0;
    int offset     = 0;
    for ( int img_index = blockIdx.x * blockDim.x + threadIdx.x; img_index < NX; img_index += blockDim.x * gridDim.x ) {

        best_index = NY;
        for ( int i_peak = 0; i_peak < NY; i_peak++ ) {
            // Check to see if any peak from this search position is in the top n_peaks scores
            if ( __low2half(my_peaks[img_index]) > secondary_peaks[img_index + i_peak * NX] ) {
                best_index = i_peak;
                break;
            }
        }

        // If we didn't find a better peak, this loop will not execute
        // We have a numel * n_peaks * 4 (score, psi, theta, phi) array
        for ( int worst_peak = NY - 1; worst_peak > best_index; worst_peak-- ) {
            offset = img_index + NX * worst_peak;
            // Move the worst peak down one
            secondary_peaks[offset] = secondary_peaks[offset - NX];
            // Psi
            offset += NX * NY;
            secondary_peaks[offset] = secondary_peaks[offset - NX];
            // Theta
            offset += NX * NY;
            secondary_peaks[offset] = secondary_peaks[offset - NX];
            // Phi
            offset += NX * NY;
            secondary_peaks[offset] = secondary_peaks[offset - NX];
        }
        // Now insert the new peak
        if ( best_index < NY ) {
            offset                  = img_index + best_index * NX;
            secondary_peaks[offset] = __low2half(my_peaks[img_index]);
            // Psi
            offset += NX * NY;
            secondary_peaks[offset] = __high2half(my_peaks[img_index]);
            // Theta
            offset += NX * NY;
            secondary_peaks[offset] = __low2half(my_new_peaks[img_index]);
            // Phi
            offset += NX * NY;
            secondary_peaks[offset] = __high2half(my_new_peaks[img_index]);
        }
    }
}

void TemplateMatchingCore::UpdateSecondaryPeaks( ) {

    precheck;
    // N
    d_padded_reference.ReturnLaunchParametersLimitSMs(5.f, 1024);

    UpdateSecondaryPeaksKernel<<<d_padded_reference.gridDims, d_padded_reference.threadsPerBlock, 0, cudaStreamPerThread>>>((__half*)secondary_peaks,
                                                                                                                            my_peaks,
                                                                                                                            my_new_peaks,
                                                                                                                            n_global_search_images_to_save,
                                                                                                                            (int)d_padded_reference.real_memory_allocated);
    postcheck;

    // We need to reset this each outer angle search or we'll never see new maximums
    cudaErr(cudaMemsetAsync(my_peaks, 0, sizeof(__half2) * d_input_image.real_memory_allocated, cudaStreamPerThread));
    cudaErr(cudaMemsetAsync(my_new_peaks, 0, sizeof(__half2) * d_input_image.real_memory_allocated, cudaStreamPerThread));
}

__global__ void MipToImageKernel(const __half2* my_peaks,
                                 const __half2* my_new_peaks,
                                 const __half*  secondary_peaks,
                                 const int      numel,
                                 cufftReal*     mip,
                                 cufftReal*     psi,
                                 cufftReal*     theta,
                                 cufftReal*     phi,
                                 const int      n_peaks) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if ( x >= numel )
        return;

    if ( n_peaks == 1 ) {

        mip[x]   = (cufftReal)__low2float(my_peaks[x]);
        psi[x]   = (cufftReal)__high2float(my_peaks[x]);
        theta[x] = (cufftReal)__low2float(my_new_peaks[x]);
        phi[x]   = (cufftReal)__high2float(my_new_peaks[x]);
    }
    else {
        int offset;
        for ( int iPeak = 0; iPeak < n_peaks; iPeak++ ) {
            offset = x + numel * iPeak; // out puts are NX * NY * NZ

            mip[offset]   = (cufftReal)secondary_peaks[offset];
            psi[offset]   = (cufftReal)secondary_peaks[offset + numel * n_peaks];
            theta[offset] = (cufftReal)secondary_peaks[offset + numel * n_peaks * 2];
            phi[offset]   = (cufftReal)secondary_peaks[offset + numel * n_peaks * 3];
        }
    }
}

void TemplateMatchingCore::MipToImage( ) {

    precheck;
    dim3 threadsPerBlock = dim3(1024, 1, 1);
    dim3 gridDims        = dim3((d_max_intensity_projection.real_memory_allocated + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);

    MipToImageKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(my_peaks,
                                                                            my_new_peaks,
                                                                            secondary_peaks,
                                                                            d_padded_reference.real_memory_allocated,
                                                                            d_max_intensity_projection.real_values,
                                                                            d_best_psi.real_values,
                                                                            d_best_theta.real_values,
                                                                            d_best_phi.real_values,
                                                                            n_global_search_images_to_save);
    postcheck;
}

__global__ void AccumulateSumsKernel(__half2* my_stats, const int numel, cufftReal* sum, cufftReal* sq_sum, const bool use_fast_fft);

void TemplateMatchingCore::AccumulateSums(__half2* my_stats, GpuImage& sum, GpuImage& sq_sum) {

    precheck;
    dim3 threadsPerBlock = dim3(1024, 1, 1);
    dim3 gridDims        = dim3((sum.real_memory_allocated + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);

    AccumulateSumsKernel<<<gridDims, threadsPerBlock, 0, cudaStreamPerThread>>>(my_stats, sum.real_memory_allocated, sum.real_values, sq_sum.real_values, use_fast_fft);
    postcheck;
}

__global__ void AccumulateSumsKernel(__half2* my_stats, const int numel, cufftReal* sum, cufftReal* sq_sum, const bool use_fast_fft) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if ( x < numel ) {

        sum[x]    = __fmaf_rn(0.0001f, __low2float(my_stats[x]), sum[x]);
        sq_sum[x] = __fmaf_rn(0.0001f, __high2float(my_stats[x]), sq_sum[x]);

        my_stats[x] = __halves2half2((__half)0., (__half)0.);
    }
}
