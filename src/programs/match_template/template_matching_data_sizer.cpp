#include <cistem_config.h>

#ifdef ENABLEGPU
#include "../../gpu/gpu_core_headers.h"
#include "../../gpu/DeviceManager.h"
#include "../../gpu/TemplateMatchingCore.h"
#else
#include "../../core/core_headers.h"
#endif

#include "../../constants/constants.h"

#if defined(ENABLE_FastFFT) && defined(ENABLEGPU)
#include "../../ext/FastFFT/include/FastFFT.h"
#endif

#include "template_matching_data_sizer.h"

TemplateMatchingDataSizer::TemplateMatchingDataSizer(Image& input_image, Image& template, const float pixel_size, const float wanted_template_padding) : pixel_size(pixel_size),
                                                                                                                                                         template_padding(wanted_template_padding) {

    MyDebugAssertTrue(pixel_size > 0.0f, "Pixel size must be greater than zero");
    // TODO: remove this constraint
    MyAssertTrue(template_padding == 1.0f, "Padding must be greater equal to 1.0");
    image_size.x = input_image.logical_x_dimension;
    image_size.y = input_image.logical_y_dimension;
    image_size.z = input_image.logical_z_dimension;
    image_size.w = (input_image.logical_x_dimension + input_image.padding_jump_value) / 2;

    template_size.x = template.logical_x_dimension;
    template_size.y = template.logical_y_dimension;
    template_size.z = template.logical_z_dimension;
    template_size.w = (template.logical_x_dimension + template.padding_jump_value) / 2;
};

void TemplateMatchingDataSizer::SetImageAndTemplateSizing(const float wanted_high_resolution_limit, const bool use_fast_fft) {

    // Make sure we aren't trying to limit beyond Nyquist, and if < Nyquist set resampling needed to true.
    SetHighResolutionLimit(wanted_high_resolution_limit);

    // Setup some limits. These could probably just go directly into their specific methods in this class
    if ( use_fast_fft ) {
        primes.assign({2});
        max_increase_by_fraction_of_image = 2.f;
    }
    else {
        primes.assign({2, 3, 5, 7, 9, 13});
        max_increase_by_fraction_of_image = 0.1f;
    }

    if ( resampling_is_needed ) {
        if ( use_fast_fft ) {
            GetResampledFFTSize( );
        }
        else {
            MyAssertFalse(true, "This branch is not yet implemented.");
        }
    }
    else {
        if ( use_fast_fft ) {
            GetResampledFFTSize( );
        }
        else {
            GetGenericFFTSize( );

            // If we get to this block our only constraint is to make the input image a nice size for general FFTs
            // and possible to rotate by 90 to make the template dimension better for fastFFT>
        }
    }
};

/**
 * @brief Always remove outliers, center and whiten prior to any transormations, resampling or chunking of the input image.
 * 
 * We ALWAYS want the starting image statistics to be the same, regardless of the final size.
 * 
 * @param input_image 
 */
void TemplateMatchingDataSizer::PreProcessInputImage(Image& input_image) {
    Curve whitening_filter;
    Curve number_of_terms;
    whitening_filter.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_image.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
    number_of_terms.SetupXAxis(0.0, 0.5 * sqrtf(2.0), int((input_image.logical_x_dimension / 2.0 + 1.0) * sqrtf(2.0) + 1.0));
    input_image.ReplaceOutliersWithMean(5.0f);
    input_image.ForwardFFT( );

    input_image.ZeroCentralPixel( );
    input_image.Compute1DPowerSpectrumCurve(&whitening_filter, &number_of_terms);
    whitening_filter.SquareRoot( );
    whitening_filter.Reciprocal( );
    whitening_filter.MultiplyByConstant(1.0f / whitening_filter.ReturnMaximumValue( ));

    input_image.ApplyCurveFilter(&whitening_filter);
    input_image.ZeroCentralPixel( );
    input_image.DivideByConstant(sqrtf(input_image.ReturnSumOfSquares( )));
    input_image.BackwardFFT( );
};

void TemplateMatchingDataSizer::CheckSizing( ) {
    use_fast_fft = false;
    if ( use_fast_fft ) {
        // We currently can only use FastFFT for power of 2 size images and template
        // TODO: the limit on templates should be trivial to remove since they are padded.
        // TOOD: Should probably consider this in the code above working out next good size, rather than only allowing power of 2,
        //       which will take a k3 to 8k x 8k it would be better to pad either
        //       a) crop to 4k x 4k (lossy but immediately supported)
        //       b) split into two images and pad each to either 4k x 4k (possibly a bit slower and not yet supported)
        // FIXME:
        if ( image_search_size.x != image_search_size.y ) {
            SendInfo("FastFFT currently only supports square images, padding smaller up\n");
        }
        if ( image_search_size.x > image_search_size.y ) {
            image_search_size.y = image_search_size.x;
        }
        else {
            image_search_size.x = image_search_size.y;
        }
        if ( image_search_size.x > 4096 * 2 || image_search_size.y > 4096 * 2 ) {
            SendError("FastFFT only supports images up to 8192x8192\n");
        }
        else if ( image_search_size.x > 4096 || image_search_size.y > 4096 ) {
            if ( use_fast_fft_and_crop ) {
                SendInfo("Warning, cropping image to max 4096x4096\n");
                image_search_size.x = std::min(image_search_size.x, 4096);
                image_search_size.y = std::min(image_search_size.y, 4096);
            }
            else {
                SendInfo("Use of FastFFT for images larger than 4k x 4k is likely a pessimation\n");
            }
        }
    }
};

void int TemplateMatchingDataSizer::GetGenericFFTSize( ) {
    // for 5760 this will return
    // 5832 2     2     2     3     3     3     3     3     3 - this is ~ 10% faster than the previous solution BUT
    int factor_result_pos{ };
    int factor_result_neg{ };

    for ( auto& prime_value : primes ) {
        factor_result_neg = ReturnClosestFactorizedLower(image_size.x, prime_value, true, MUST_BE_FACTOR_OF_FOUR);
        factor_result_pos = ReturnClosestFactorizedUpper(image_size.x, prime_value, true, MUST_BE_FACTOR_OF_FOUR);

        if ( (float)(image_size.x - factor_result_neg) < (float)input_reconstruction.logical_x_dimension * max_reduction_by_fraction_of_reference ) {
            image_search_size.x = factor_result_neg;
            break;
        }
        if ( (float)(-image_size.x + factor_result_pos) < (float)input_image.logical_x_dimension * max_increase_by_fraction_of_image ) {
            image_search_size.x = factor_result_pos;
            break;
        }
    }

    for ( auto& prime_value : primes ) {
        factor_result_neg = ReturnClosestFactorizedLower(image_size.y, prime_value, true, MUST_BE_FACTOR_OF_FOUR);
        factor_result_pos = ReturnClosestFactorizedUpper(image_size.y, prime_value, true, MUST_BE_FACTOR_OF_FOUR);

        if ( (float)(image_size.y - factor_result_neg) < (float)input_reconstruction_file.ReturnYSize( ) * max_reduction_by_fraction_of_reference ) {
            image_search_size.y = factor_result_neg;
            break;
        }
        if ( (float)(-image_size.y + factor_result_pos) < (float)input_image.logical_y_dimension * max_increase_by_fraction_of_image ) {
            image_search_size.y = factor_result_pos;
            break;
        }
    }

    //  TODO: this is currently used to restrict the region that is valid for the histogram, however, we will probably need a better descriptor
    // when we get to chunking an image.

    int max_padding{ };
    if ( image_search_size.x - image_size.x > max_padding )
        max_padding = image_search_size.x - image_size.x;
    if ( image_search_size.y - image_size.y > max_padding )
        max_padding = image_search_size.y - image_size.y;

    // There are no restrictions on the template being  a power of two, but we should want a decent size
    template_search_size.x = ReturnClosestFactorizedUpper(template_size.x, 5, true, MUST_BE_POWER_OF_TWO);
    template_search_size.y = template_search_size.x;
    template_search_size.z = template_search_size.x;
    // We know this is an even dimension so adding 2
    template_search_size.w = (template_search_size.x + 2) / 2;

    // Make sure these are set even if we don't plan to use them righ tnow.
    template_pre_scaling_size = template_size;
    template_cropped_size     = template_size;

    search_pixel_size = pixel_size;

    return max_padding;
};

void TemplateMatchingDataSizer::GetResampledFFTSize( ) {

    // We want the binning to be isotropic, and the easiest way to ensure that is to first pad any non-square input_image to a square size in realspace.
    // Presumably we'll be using a power of 2 square size anyway for FastFFT (though rectangular images should be supported at some point.)
    // The other requirement is to ensure the resulting pixel size is the same for the reference and the search images.
    // Ideally, we would just calculate a scattering potential at the correct size. (unless the user has a map they wan tt o use)
    // In that case, we want to first calculate our wanted size in the image, then determine how much wiggle room we have until the next power of 2,
    // then determine the best matching binning considering the input 3d
    int   max_square_size       = std::max(image_size.x, image_size.y);
    float wanted_binning_factor = high_resolution_limit / pixel_size / 2.0f;
    int   wanted_binned_size    = int(float(max_square_size) / wanted_binning_factor + 0.5f);
    if ( IsOdd(wanted_binned_size) )
        wanted_binned_size++;

    float actual_image_binning = float(image_size.x) / float(wanted_binned_size);

    // Get the closest we can with this size
    int closest_3d_binned_size = int(template_size.x / actual_image_binning + 0.5f);
    if ( IsOdd(closest_3d_binned_size) )
        closest_3d_binned_size++;
    float closest_3d_binning = float(template_size.x) / float(closest_3d_binned_size);

    wxPrintf("input sizes are %i %i\n", image_size.x, image.size.y);
    wxPrintf("input 3d sizes are %i %i\n", template_size.x, template_size.y);
    // Print out some values for testing
    wxPrintf("wanted image bin factor and new pixel size = %f %f\n", actual_image_binning, pixel_size * actual_image_binning);
    wxPrintf("closest 3d bin factor and new pixel size = %f %f\n", closest_3d_binning, closest_3d_binning * pixel_size);

    // TODO: this should consider how close we are to the next power of two, which for the time being,
    // we are explicitly padding to.
    int padding_3d = 0;

    // FIXME: The threshold here should be in constants and determined empirically.
    constexpr float pixel_threshold = 0.0005f;
    bool            match_found     = false;
    if ( fabsf(closest_3d_binning * pixel_size - pixel_size * actual_image_binning) > pixel_threshold ) {
        wxPrintf("Warning, the pixel size of the input 3d and the input images are not the same\n");

        for ( padding_3d = 1; padding_3d < 100; padding_3d++ ) {
            closest_3d_binned_size = int((template_size + padding_3d) / actual_image_binning + 0.5f);
            if ( IsOdd(closest_3d_binned_size) )
                closest_3d_binned_size++;
            closest_3d_binning = float(template_size + padding_3d) / float(closest_3d_binned_size);

            wxPrintf("after padding by %d closest 3d bin factor and new pixel size = %f %f\n", padding_3d, closest_3d_binning, closest_3d_binning * pixel_size);

            float pix_diff = closest_3d_binning * pixel_size - pixel_size * actual_image_binning;
            if ( fabsf(pix_diff) > 0.0001f )
                wxPrintf("Warning, the pixel size of the input 3d and the input images are not the same, difference is %3.6f\n", pix_diff);
            else {
                wxPrintf("Success!, with padding %d the pixel size of the input 3d and the input images are not the same, difference is %3.6f\n", padding_3d, pix_diff);
                match_found = true;
                break;
            }
        }
    }
    else
        match_found = true;

    MyAssertTrue(match_found, "Could not find a match between the input 3d and the input images");

    // FIXME: this should eventulally not be required by FastFFT for template_size < image_size
    int power_of_two_size_3d = get_next_power_of_two(closest_3d_binned_size);
    int power_of_two_size_2d = get_next_power_of_two(wanted_binned_size);

    template_pre_scaling_size.x = padding_3d + template_size.x;
    template_pre_scaling_size.y = padding_3d + template_size.y;
    template_pre_scaling_size.z = padding_3d + template_size.z;

    template_cropped_size.x = closest_3d_binned_size;
    template_cropped_size.y = closest_3d_binned_size;
    template_cropped_size.z = closest_3d_binned_size;

    template_search_size.x = power_of_two_size_3d;
    template_search_size.y = power_of_two_size_3d;
    template_search_size.z = power_of_two_size_3d;

    image_pre_scaling_size.x = max_square_size;
    image_pre_scaling_size.y = max_square_size;
    image_pre_scaling_size.z = 1; // FIXME: once we add chunking ...

    image_cropped_size.x = wanted_binned_size;
    image_cropped_size.y = wanted_binned_size;
    image_cropped_size.z = 1; // FIXME: once we add chunking ...

    image_search_size.x = power_of_two_size_2d;
    image_search_size.y = power_of_two_size_2d;
    image_search_size.z = 1; // FIXME: once we add chunking ...

    wxPrintf("The reference will be padded by %d, cropped to %d, and then padded again to %d\n", padding_3d, closest_3d_binned_size, power_of_two_size_3d);
    wxPrintf("The input image will be padded by %d,%d, cropped to %d, and then padded again to %d\n", max_square_size - image_size.x, max_square_size - image_size.y, wanted_binned_size, power_of_two_size_2d);
    wxPrintf("template_size = %i\n", template_size.x);
    wxPrintf("closest_3d_binned_size = %i\n", closest_3d_binned_size);
    wxPrintf("closest_3d_binning = %f\n", closest_3d_binning);
    wxPrintf("closest_3d_binning * pixel_size = %f\n", closest_3d_binning * pixel_size);
    wxPrintf("original image size = %i\n", int(image_size.x));
    wxPrintf("wanted_binned_size = %i\n", wanted_binned_size);
    wxPrintf("actual_image_binning = %f\n", actual_image_binning);
    wxPrintf("new pixel size = actual_image_binning * pixel_size = %f\n", actual_image_binning * pixel_size);
    search_pixel_size = pixel_size * actual_image_binning;
    // Now try to increase the padding of the input image to match the 3d
};

void TemplateMatchingDataSizer::SetHighResolutionLimit(const float wanted_high_resolution_limit) {
    if ( wanted_high_resolution_limit < 2.0f * pixel_size )
        high_resolution_limit = 2.0f * pixel_size;
    else
        high_resolution_limit = wanted_high_resolution_limit;

    if ( FloatsAreAlmostTheSame(high_resolution_limit, 2.0f * pixel_size) )
        resampling_is_needed = false;
    else
        resampling_is_needed = true;
};

void TemplateMatchingDataSizer::ResizeTemplate_preSearch(Image& template_image) {
    template_image.Resize(template_pre_scaling_size.x, template_pre_scaling_size.y, template_pre_scaling_size.z, template_image.ReturnAverageOfRealValuesOnEdges( ));
    template_image.ForwardFFT( );
    template_image.Resize(template_cropped_size.x, template_cropped_size.y, template_cropped_size.z);
    template_image.BackwardFFT( );
    template_image.Resize(template_search_size.x, template_search_size.y, template_search_size.z, template_image.ReturnAverageOfRealValuesOnEdges( ));
};

void TemplateMatchingDataSizer::ResizeTemplate_postSearch(Image& template_image) {
    if ( template_cropped_size.x != template_size.x ) {
        template_image.Resize(template_cropped_size.x, template_cropped_size.y, template_cropped_size.z);
    }
};

void TemplateMatchingDataSizer::ResizeImage_preSearch(Image& input_image) {

    Image tmp_sq;

    tmp_sq.Allocate(image_pre_scaling_size.x, image_pre_scaling_size.y, image_pre_scaling_size.z, true);
    tmp_sq.AddGaussianNoise(1.0f);

    input_image.ClipInto(&tmp_sq, 0.0f, false, 1.0f, 0, 0, 0, true);

    tmp_sq.ForwardFFT( );
    tmp_sq.Resize(image_cropped_size.x, image_cropped_size.y, image_cropped_size.z);
    tmp_sq.ZeroCentralPixel( );
    tmp_sq.DivideByConstant(sqrtf(tmp_sq.ReturnSumOfSquares( )));
    tmp_sq.BackwardFFT( );

    input_image.Allocate(image_search_size.x, image_search_size.y, image_search_size.z, true);
    input_image.AddGaussianNoise(1.0f);
    tmp_sq.ClipInto(&input_image, 0.0f, false, 1.0f, 0, 0, 0, true);

#ifdef ROTATEFORSPEED
    if ( ! is_power_of_two(factorizable_x) && is_power_of_two(factorizable_y) ) {
        // The speedup in the FFT for better factorization is also dependent on the dimension. The full transform (in cufft anyway) is faster if the best dimension is on X.
        // TODO figure out how to check the case where there is no factor of two, but one dimension is still faster. Probably getting around to writing an explicit planning tool would be useful.
        if ( ReturnThreadNumberOfCurrentThread( ) == 0 ) {
            wxPrintf("Rotating the search image for speed\n");
        }
        input_image.RotateInPlaceAboutZBy90Degrees(true);
        // bool preserve_origin = true;
        // input_reconstruction.RotateInPlaceAboutZBy90Degrees(true, preserve_origin);
        // The amplitude spectrum is also rotated
        is_rotated_by_90 = true;
    }
    else {
        if ( ReturnThreadNumberOfCurrentThread( ) == 0 ) {
            wxPrintf("Not rotating the search image for speed even though it is enabled\n");
        }
        is_rotated_by_90 = false;
    }
#endif
};

void TemplateMatchingDataSizer::ResizeImage_postSearch(Image& input_image,
                                                       Image& max_intensity_projection,
                                                       Image& best_psi,
                                                       Image& best_phi,
                                                       Image& best_theta,
                                                       Image& best_defocus,
                                                       Image& best_pixel_size,
                                                       Image& correlation_pixel_sum_image,
                                                       Image& correlation_pixel_sum_of_squares_image) {

    // Work through the transformations backward to get to the original image size
    if ( is_rotated_by_90 ) {
        // swap back all the images prior to re-sizing
        input_image.BackwardFFT( );
        input_image.RotateInPlaceAboutZBy90Degrees(false);
        max_intensity_projection.RotateInPlaceAboutZBy90Degrees(false);

        best_psi.RotateInPlaceAboutZBy90Degrees(false);
        // If the template is also rotated, then this additional accounting is not needed.
        // To account for the pre-rotation, psi needs to have 90 added to it.
        best_psi.AddConstant(90.0f);
        // We also want the angles to remain in (0,360] so loop over and clamp
        for ( int idx = 0; idx < best_psi.real_memory_allocated; idx++ ) {
            best_psi.real_values[idx] = clamp_angular_range_0_to_2pi(best_psi.real_values[idx], true);
        }
        best_theta.RotateInPlaceAboutZBy90Degrees(false);
        best_phi.RotateInPlaceAboutZBy90Degrees(false);
        best_defocus.RotateInPlaceAboutZBy90Degrees(false);
        best_pixel_size.RotateInPlaceAboutZBy90Degrees(false);

        correlation_pixel_sum_image.RotateInPlaceAboutZBy90Degrees(false);
        correlation_pixel_sum_of_squares_image.RotateInPlaceAboutZBy90Degrees(false);
    }

    // We need to use nearest neighbor interpolation to cast all existing values back to the original size.
    Image tmp_mip, tmp_psi, tmp_phi, tmp_theta, tmp_defocus, tmp_pixel_size, tmp_sum, tmp_sum_sq;

    // original size -> pad to square -> crop to binned -> pad to fourier
    // The new images at the square binned size (remove the padding to power of two)
    tmp_mip.Allocate(wanted_binned_size, wanted_binned_size, 1);
    tmp_psi.Allocate(wanted_binned_size, wanted_binned_size, 1);
    tmp_phi.Allocate(wanted_binned_size, wanted_binned_size, 1);
    tmp_theta.Allocate(wanted_binned_size, wanted_binned_size, 1);
    tmp_defocus.Allocate(wanted_binned_size, wanted_binned_size, 1); // NOTE: we could avoid allocating this if we know it is not needed
    tmp_pixel_size.Allocate(wanted_binned_size, wanted_binned_size, 1); // NOTE: we could avoid allocating this if we know it is not needed
    tmp_sum.Allocate(wanted_binned_size, wanted_binned_size, 1);
    tmp_sum_sq.Allocate(wanted_binned_size, wanted_binned_size, 1);

    max_intensity_projection.ClipInto(&tmp_mip);
    best_psi.ClipInto(&tmp_psi);
    best_phi.ClipInto(&tmp_phi);
    best_theta.ClipInto(&tmp_theta);
    best_defocus.ClipInto(&tmp_defocus);
    best_pixel_size.ClipInto(&tmp_pixel_size);
    correlation_pixel_sum_image.ClipInto(&tmp_sum);
    correlation_pixel_sum_of_squares_image.ClipInto(&tmp_sum_sq);

    // // Now reset the images to the original size
    // max_intensity_projection.Allocate(original_input_image_x, original_input_image_y, 1);
    // best_psi.Allocate(original_input_image_x, original_input_image_y, 1);
    // best_phi.Allocate(original_input_image_x, original_input_image_y, 1);
    // best_theta.Allocate(original_input_image_x, original_input_image_y, 1);
    // best_defocus.Allocate(original_input_image_x, original_input_image_y, 1);
    // best_pixel_size.Allocate(original_input_image_x, original_input_image_y, 1);
    // Now reset the images to the original size

    // Setting to the max_sq_size to map back the resampled values
    max_intensity_projection.Allocate(max_square_size, max_square_size, 1);
    best_psi.Allocate(max_square_size, max_square_size, 1);
    best_phi.Allocate(max_square_size, max_square_size, 1);
    best_theta.Allocate(max_square_size, max_square_size, 1);
    best_defocus.Allocate(max_square_size, max_square_size, 1);
    best_pixel_size.Allocate(max_square_size, max_square_size, 1);
    correlation_pixel_sum_image.Allocate(max_square_size, max_square_size, 1);
    correlation_pixel_sum_of_squares_image.Allocate(max_square_size, max_square_size, 1);

    // Make sure they are all zero
    constexpr float no_value = -std::numeric_limits<float>::max( );
    max_intensity_projection.SetToConstant(no_value);
    best_psi.SetToConstant(no_value);
    best_phi.SetToConstant(no_value);
    best_theta.SetToConstant(no_value);
    best_defocus.SetToConstant(no_value);
    best_pixel_size.SetToConstant(no_value);
    correlation_pixel_sum_image.SetToConstant(no_value);
    correlation_pixel_sum_of_squares_image.SetToConstant(no_value);

    long nn_counter          = 0;
    long out_of_bounds_value = 0;
    long address;
    for ( int j = 0; j < tmp_mip.logical_y_dimension; j++ ) {
        int y_offset_from_origin = j - tmp_mip.physical_address_of_box_center_y;
        for ( int i = 0; i < tmp_mip.logical_x_dimension; i++ ) {
            // Get this pixels offset from the center of the box
            int x_offset_from_origin = i - tmp_mip.physical_address_of_box_center_x;

            // Scale by the binning
            // TODO: not sure if round or truncation (floor) makes more sense here
            int x_non_binned = max_intensity_projection.physical_address_of_box_center_x + myroundint(float(x_offset_from_origin) * actual_image_binning);
            int y_non_binned = max_intensity_projection.physical_address_of_box_center_y + myroundint(float(y_offset_from_origin) * actual_image_binning);

            if ( x_non_binned >= 0 && x_non_binned < max_intensity_projection.logical_x_dimension && y_non_binned >= 0 && y_non_binned < max_intensity_projection.logical_y_dimension )
                address = max_intensity_projection.ReturnReal1DAddressFromPhysicalCoord(x_non_binned, y_non_binned, 0);
            else
                address = -1;

            // There really shouldn't be any peaks out of bounds
            // I think we should only every update an address once, so let's check it here for now.
            if ( address < 0 || address > max_intensity_projection.real_memory_allocated ) {
                out_of_bounds_value++;
            }
            else {
                MyDebugAssertTrue(max_intensity_projection.real_values[address] == no_value, "Address already updated");
                max_intensity_projection.real_values[address]               = tmp_mip.real_values[nn_counter];
                best_psi.real_values[address]                               = tmp_psi.real_values[nn_counter];
                best_phi.real_values[address]                               = tmp_phi.real_values[nn_counter];
                best_theta.real_values[address]                             = tmp_theta.real_values[nn_counter];
                best_defocus.real_values[address]                           = tmp_defocus.real_values[nn_counter];
                best_pixel_size.real_values[address]                        = tmp_pixel_size.real_values[nn_counter];
                correlation_pixel_sum_image.real_values[address]            = tmp_sum.real_values[nn_counter];
                correlation_pixel_sum_of_squares_image.real_values[address] = tmp_sum_sq.real_values[nn_counter];
            }
            nn_counter++;
        }
        nn_counter += tmp_mip.padding_jump_value;
    }

    // Get rid of the -FLT_MAX values
    for ( int i = 0; i < max_intensity_projection.real_memory_allocated; i++ ) {
        if ( max_intensity_projection.real_values[i] == no_value )
            max_intensity_projection.real_values[i] = 0.0f;
    }
    std::cerr << "\nThere were " << out_of_bounds_value << " out of bounds values\n";

    int final_resize_x = original_input_image_x;
    int final_resize_y = original_input_image_y;
};