#include <cistem_config.h>

#ifdef ENABLEGPU
#include "../../../gpu/gpu_core_headers.h"
#else
#error "GPU is not enabled"
#include "../../../core/core_headers.h"
#endif

#include "../../../gpu/GpuImage.h"

#include "../common/common.h"
#include "resample.h"

void ResampleRunner(const wxString& temp_directory) {

    SamplesPrintTestStartMessage("Starting downsampling tests:", false);

    wxString cistem_ref_dir = CheckForReferenceImages( );

    constexpr bool test_is_to_be_run = true;
    if constexpr ( test_is_to_be_run ) {
        // If we are in the dev container the CISTEM_REF_IMAGES variable should be defined, pointing to images we need.
        TEST(DoCTFImageVsTexture(cistem_ref_dir, temp_directory));
        TEST(DoFourierCropVsLerpResize(cistem_ref_dir, temp_directory));
    }
    else
        SamplesTestResultCanFail(false);

    SamplesPrintEndMessage( );

    return;
}

bool DoCTFImageVsTexture(const wxString& cistem_ref_dir, const wxString& temp_directory) {
    MyAssertFalse(cistem_ref_dir == temp_directory, "The temp directory should not be the same as the CISTEM_REF_IMAGES directory.");

    bool          passed             = true;
    bool          all_passed         = true;
    constexpr int logical_input_size = 384;

    SamplesBeginTest("CTF image vs texture", passed);

    std::string volume_filename = cistem_ref_dir.ToStdString( ) + "/ribo_ref.mrc";
    // Read in and normalize the 3d to use for projection
    Image      cpu_volume;
    ImageFile  cpu_volume_file;
    const bool over_write_input = false;
    cpu_volume_file.OpenFile(volume_filename, over_write_input);
    cpu_volume.ReadSlices(&cpu_volume_file, 1, cpu_volume_file.ReturnNumberOfSlices( ));
    cpu_volume.ZeroFloatAndNormalize( );

    // Make sure the volume has the expected size
    MyAssertTrue(cpu_volume.logical_x_dimension == logical_input_size && cpu_volume.IsCubic( ), "The volume should be 384x384x384");

    // Prepare for GPU projection
    GpuImage       gpu_volume;
    constexpr bool also_swap_real_space_quadrants = true;
    cpu_volume.SwapFourierSpaceQuadrants(also_swap_real_space_quadrants);
    // Associate the gpu volume with the cpu volume, getting meta data and pinning the host pointer.
    gpu_volume.Init(cpu_volume, false, true);
    gpu_volume.CopyHostToDeviceTextureComplex<3>(cpu_volume);

    const float pixel_size                = 1.0f;
    const float resolution_limit          = 1.0f;
    float       real_space_binning_factor = 1.0f;
    const bool  apply_resolution_limit    = false;
    const bool  apply_shifts              = false;
    const bool  swap_real_space_quadrants = true;
    const bool  apply_ctf                 = true;
    const bool  absolute_ctf              = false;
    const bool  zero_central_pixel        = true;

    // Generate a CTF image
    CTF   ctf(300.f, 2.7f, 0.07f, 12000.f, 12000.f, 40.f, pixel_size, 0.f);
    Image ctf_image(logical_input_size, logical_input_size, 1, false);
    ctf_image.CalculateCTFImage(ctf);

    Image swapped_ctf_image(ctf_image);

    swapped_ctf_image.SwapFourierSpaceQuadrants(false, true);

    // Now we'll grab a projection and apply the CTF to it in the same kernel.
    AnglesAndShifts prj_angles(10.f, -20.f, 130.f, 0.f, 0.f);
    GpuImage        gpu_prj(ctf_image);
    GpuImage        d_projection_filter(ctf_image);
    d_projection_filter.CopyHostToDevice(ctf_image);
    d_projection_filter.CopyFP32toFP16buffer(false);

    gpu_prj.ExtractSliceShiftAndCtf(&gpu_volume,
                                    &d_projection_filter,
                                    prj_angles,
                                    pixel_size,
                                    real_space_binning_factor,
                                    resolution_limit,
                                    apply_resolution_limit,
                                    swap_real_space_quadrants,
                                    apply_shifts,
                                    apply_ctf,
                                    absolute_ctf,
                                    zero_central_pixel);

    gpu_prj.QuickAndDirtyWriteSlice("ctf_image.mrc", 1);

    // ctf_image.is_in_real_space = true;

    // ctf_image.QuickAndDirtyWriteSlice("ctf_image_ref.mrc", 1);
    // ctf_image.ForwardFFT( );
    // ctf_image.PhaseShift(0.f, float(ctf_image.logical_y_dimension / 2), 0.f);
    // ctf_image.BackwardFFT( );
    // ctf_image.is_in_real_space = true;
    // ctf_image.QuickAndDirtyWriteSlice("ctf_image_ref2.mrc", 1);
    // Now, copy the projection filter into the texture cache so that we can read from that
    d_projection_filter.CopyHostToDeviceTextureRealValued<2>(swapped_ctf_image);

    constexpr bool use_ctf_texture = true;
    gpu_prj.ExtractSliceShiftAndCtf<use_ctf_texture>(&gpu_volume,
                                                     &d_projection_filter,
                                                     prj_angles,
                                                     pixel_size,
                                                     real_space_binning_factor,
                                                     resolution_limit,
                                                     apply_resolution_limit,
                                                     swap_real_space_quadrants,
                                                     apply_shifts,
                                                     apply_ctf,
                                                     absolute_ctf,
                                                     zero_central_pixel);

    gpu_prj.QuickAndDirtyWriteSlice("ctf_texture.mrc", 1);
    exit(0);

    SamplesTestResult(passed);

    return all_passed;
}

bool DoFourierCropVsLerpResize(const wxString& cistem_ref_dir, const wxString& temp_directory) {
    MyAssertFalse(cistem_ref_dir == temp_directory, "The temp directory should not be the same as the CISTEM_REF_IMAGES directory.");

    bool passed     = true;
    bool all_passed = true;

    constexpr int logical_input_size = 384;

    AnglesAndShifts prj_angles(10.f, -20.f, 130.f, 0.f, 0.f);

    SamplesBeginTest("Extract slice and downsample", passed);

    std::string volume_filename          = cistem_ref_dir.ToStdString( ) + "/ribo_ref.mrc";
    std::string prj_input_filename_base  = cistem_ref_dir.ToStdString( ) + "/ribo_ref_prj_";
    std::string prj_output_filename_base = temp_directory.ToStdString( ) + "/ribo_ref_prj_";

    bool      over_write_input = false;
    Image     cpu_volume;
    ImageFile cpu_volume_file;

    GpuImage gpu_volume;
    GpuImage gpu_prj_full; // project 384 then crop to 192 (downsample by 2)
    GpuImage gpu_prj_cropped;
    GpuImage gpu_prj_lerp; // project and resample in the same step
    GpuImage gpu_prj_lerp_non_binned_size;

    // Read in and normalize the 3d to use for projection
    cpu_volume_file.OpenFile(volume_filename, over_write_input);
    cpu_volume.ReadSlices(&cpu_volume_file, 1, cpu_volume_file.ReturnNumberOfSlices( ));
    cpu_volume.ZeroFloatAndNormalize( );

    // Make sure the volume has the expected size
    MyAssertTrue(cpu_volume.logical_x_dimension == logical_input_size && cpu_volume.IsCubic( ), "The volume should be 384x384x384");

    // Prepare for GPU projection
    constexpr bool also_swap_real_space_quadrants = true;
    cpu_volume.SwapFourierSpaceQuadrants(also_swap_real_space_quadrants);
    // Associate the gpu volume with the cpu volume, getting meta data and pinning the host pointer.
    gpu_volume.Init(cpu_volume, false, true);
    gpu_volume.CopyHostToDeviceTextureComplex<3>(cpu_volume);

    // For the positive control, project at the full size, and fourier crop to the binned size
    gpu_prj_full.Allocate(logical_input_size, logical_input_size, 1, false, false);
    gpu_prj_cropped.Allocate(logical_input_size / 2, logical_input_size / 2, 1, false, false);

    // For direct comparison to gpu_prj_cropped, incorporate the lerp into the projection obviating the need for a separate crop.
    gpu_prj_lerp.Allocate(logical_input_size / 2, logical_input_size / 2, 1, false, false);
    // For the case where we would first bin but then zero-pad to some other larger size, for example, to have a nice power of 2 image.
    gpu_prj_lerp_non_binned_size.Allocate(logical_input_size + 128, logical_input_size + 128, 1, false, false);
    // Make sure there are no non-zero vals
    gpu_prj_full.SetToConstant(0.0f);
    gpu_prj_cropped.SetToConstant(0.0f);
    gpu_prj_lerp.SetToConstant(0.0f);
    gpu_prj_lerp_non_binned_size.SetToConstant(0.0f);

    // The gpu projection method expects quadrants to be swapped.
    gpu_prj_full.object_is_centred_in_box                 = false;
    gpu_prj_cropped.object_is_centred_in_box              = false;
    gpu_prj_lerp.object_is_centred_in_box                 = false;
    gpu_prj_lerp_non_binned_size.object_is_centred_in_box = false;

    // Dummy ctf image
    GpuImage dummy_ctf_image;

    constexpr float resolution_limit          = 1.0f;
    constexpr bool  apply_resolution_limit    = false;
    constexpr bool  apply_shifts              = false;
    constexpr bool  swap_real_space_quadrants = true;
    constexpr bool  apply_ctf                 = false;
    constexpr bool  absolute_ctf              = false;
    constexpr bool  zero_central_pixel        = true;

    float real_space_binning_factor = 1.0f;
    // Project the full size image
    gpu_prj_full.ExtractSliceShiftAndCtf(&gpu_volume, &dummy_ctf_image, prj_angles, 1.0f, real_space_binning_factor, resolution_limit, apply_resolution_limit, swap_real_space_quadrants, apply_ctf, absolute_ctf, zero_central_pixel);

    // Crop the full size image
    gpu_prj_full.ClipIntoFourierSpace(&gpu_prj_cropped, 0.f);

    std::array<int, 5> cropped_sizes{382, 192, 96, 48, 24};
    for ( auto& cropped_size : cropped_sizes ) {
        GpuImage binned_img, cropped_img;
        binned_img.Allocate(cropped_size, cropped_size, 1, false, false);
        cropped_img.Allocate(cropped_size, cropped_size, 1, false, false);
        real_space_binning_factor = float(logical_input_size) / float(cropped_size);
        binned_img.ExtractSliceShiftAndCtf(&gpu_volume, &dummy_ctf_image, prj_angles, 1.0f, real_space_binning_factor, resolution_limit, apply_resolution_limit, swap_real_space_quadrants, apply_shifts, apply_ctf, absolute_ctf, zero_central_pixel);
        gpu_prj_full.ClipIntoFourierSpace(&cropped_img, 0.f);

        binned_img.BackwardFFT( );
        cropped_img.BackwardFFT( );
        // Calculate the mean square error between the two images
        cropped_img.SubtractImage(binned_img);
        float SS = cropped_img.ReturnSumOfSquares( );
        passed   = passed && (FloatsAreAlmostTheSame(SS, 0.0f));
    }

    all_passed = passed ? all_passed : false;
    SamplesTestResult(passed);

    return true;
}