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

// #define DEBUG_IMG_OUTPUT "/tmp"

TemplateMatchingDataSizer::TemplateMatchingDataSizer(MyApp* wanted_parent_ptr,
                                                     Image& input_image,
                                                     Image& wanted_template,
                                                     float  wanted_pixel_size,
                                                     float  wanted_template_padding)
    : parent_match_template_app_ptr{wanted_parent_ptr},
      pixel_size{wanted_pixel_size},
      template_padding{wanted_template_padding} {

    MyDebugAssertTrue(pixel_size > 0.0f, "Pixel size must be greater than zero");
    // TODO: remove this constraint
    MyAssertTrue(template_padding == 1.0f, "Padding must be greater equal to 1.0");
    image_size.x = input_image.logical_x_dimension;
    image_size.y = input_image.logical_y_dimension;
    image_size.z = input_image.logical_z_dimension;
    image_size.w = (input_image.logical_x_dimension + input_image.padding_jump_value) / 2;

    template_size.x = wanted_template.logical_x_dimension;
    template_size.y = wanted_template.logical_y_dimension;
    template_size.z = wanted_template.logical_z_dimension;
    template_size.w = (wanted_template.logical_x_dimension + wanted_template.padding_jump_value) / 2;
};

TemplateMatchingDataSizer::~TemplateMatchingDataSizer( ){
        // Nothing to do here
};

void TemplateMatchingDataSizer::SetImageAndTemplateSizing(const float wanted_high_resolution_limit, const bool use_fast_fft) {
    MyDebugAssertFalse(sizing_is_set, "Sizing has already been set");
    // Make sure we aren't trying to limit beyond Nyquist, and if < Nyquist set resampling needed to true.
    SetHighResolutionLimit(wanted_high_resolution_limit);
    this->use_fast_fft = use_fast_fft;

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
#ifdef DEBUG_IMG_OUTPUT
    if ( ReturnThreadNumberOfCurrentThread( ) == 0 )
        input_image.QuickAndDirtyWriteSlice(DEBUG_IMG_OUTPUT "/input_image.mrc", 1);
#endif
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

#ifdef DEBUG_IMG_OUTPUT
    if ( ReturnThreadNumberOfCurrentThread( ) == 0 )
        input_image.QuickAndDirtyWriteSlice(DEBUG_IMG_OUTPUT "/input_image_whitened.mrc", 1);
#endif
    // Saving a copy of the pre-processed image for later use to determine peak heights not damped by resampling.
    // TODO: we can also use this (with z > )
    pre_processed_image.at(0) = input_image;
};

void TemplateMatchingDataSizer::CheckSizing( ) {
    // TODO: remove thiss
    if ( use_fast_fft ) {
        // We currently can only use FastFFT for power of 2 size images and template
        // TODO: the limit on templates should be trivial to remove since they are padded.
        // TOOD: Should probably consider this in the code above working out next good size, rather than only allowing power of 2,
        //       which will take a k3 to 8k x 8k it would be better to pad either
        //       a) crop to 4k x 4k (lossy but immediately supported)
        //       b) split into two images and pad each to either 4k x 4k (possibly a bit slower and not yet supported)
        // FIXME:
        if ( image_search_size.x != image_search_size.y ) {
            parent_match_template_app_ptr->SendInfo("FastFFT currently only supports square images, padding smaller up\n");
        }
        if ( image_search_size.x > image_search_size.y ) {
            image_search_size.y = image_search_size.x;
        }
        else {
            image_search_size.x = image_search_size.y;
        }
        if ( image_search_size.x > 4096 || image_search_size.y > 4096 ) {
            parent_match_template_app_ptr->SendError("FastFFT only supports images up to 4096x4096\n");
        }
    }
    else {
        // TODO: what else should be be verifying - rotation if warranted and power of two / nice siziing?
    }
};

void TemplateMatchingDataSizer::GetGenericFFTSize( ) {
    // for 5760 this will return
    // 5832 2     2     2     3     3     3     3     3     3 - this is ~ 10% faster than the previous solution BUT
    int factor_result_pos{ };

    for ( auto& prime_value : primes ) {
        factor_result_pos = ReturnClosestFactorizedUpper(image_size.x - 1, prime_value, true, MUST_BE_FACTOR_OF);
        if ( (float)(-image_size.x + factor_result_pos) < float(image_size.x) * max_increase_by_fraction_of_image ) {
            image_search_size.x = factor_result_pos;
            break;
        }
    }

    for ( auto& prime_value : primes ) {
        factor_result_pos = ReturnClosestFactorizedUpper(image_size.y - 1, prime_value, true, MUST_BE_FACTOR_OF);
        if ( (float)(-image_size.y + factor_result_pos) < float(image_size.y) * max_increase_by_fraction_of_image ) {
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

    CheckSizing( );
    sizing_is_set = true;

    int pre_binning_padding_x;
    int post_binning_padding_x;
    int pre_binning_padding_y;
    int post_binning_padding_y;

    // NOTE: there are two seperate functions because more the more complicated resizing in GetResampledFFTSize requires handling the binning factor
    // in-between these two calls.
    GetInputImageToEvenAndSquareOrPrimeFactoredSizePadding(pre_binning_padding_x, pre_binning_padding_y, post_binning_padding_x, post_binning_padding_y);
    SetValidSearchImageIndiciesFromPadding(pre_binning_padding_x, pre_binning_padding_y, post_binning_padding_x, post_binning_padding_y);
    return;
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

    wxPrintf("input sizes are %i %i\n", image_size.x, image_size.y);
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
            // NOTE: this line assumes a cubic volume
            closest_3d_binned_size = int((template_size.x + padding_3d) / actual_image_binning + 0.5f);
            if ( IsOdd(closest_3d_binned_size) )
                closest_3d_binned_size++;
            // NOTE: this line assumes a cubic volume
            closest_3d_binning = float(template_size.x + padding_3d) / float(closest_3d_binned_size);

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

    CheckSizing( );
    sizing_is_set = true;

    int pre_binning_padding_x;
    int post_binning_padding_x;
    int pre_binning_padding_y;
    int post_binning_padding_y;

    // Things are simplified because the padding is always resulting in an even dimensions
    // NOTE: assuming integer division.
    GetInputImageToEvenAndSquareOrPrimeFactoredSizePadding(pre_binning_padding_x, pre_binning_padding_y, post_binning_padding_x, post_binning_padding_y);

    // Here I think the easiest way to handle fractional reduction, which could result in an odd number of invalid rows/columns is to round up
    float binning_factor   = search_pixel_size / pixel_size;
    pre_binning_padding_x  = myroundint(ceilf(float(pre_binning_padding_x) / binning_factor));
    pre_binning_padding_y  = myroundint(ceilf(float(pre_binning_padding_y) / binning_factor));
    post_binning_padding_x = myroundint(ceilf(float(post_binning_padding_x) / binning_factor));
    post_binning_padding_y = myroundint(ceilf(float(post_binning_padding_y) / binning_factor));

    // Now add on any padding needed to make the image a power of two
    // These are both even dimensions, so we can just use the symmetric padding.
    pre_binning_padding_x += (image_search_size.x - image_cropped_size.x) / 2;
    pre_binning_padding_y += (image_search_size.y - image_cropped_size.y) / 2;
    post_binning_padding_x += (image_search_size.x - image_cropped_size.x) / 2;
    post_binning_padding_y += (image_search_size.y - image_cropped_size.y) / 2;

    SetValidSearchImageIndiciesFromPadding(pre_binning_padding_x, pre_binning_padding_y, post_binning_padding_x, post_binning_padding_y);
};

void TemplateMatchingDataSizer::SetValidSearchImageIndiciesFromPadding(const int pre_padding_x, const int pre_padding_y, const int post_padding_x, const int post_padding_y) {
    MyDebugAssertTrue(sizing_is_set, "Sizing has not been set");
    MyDebugAssertFalse(valid_bounds_are_set, "Valid bounds have already been set");

    // This could be too large if the user created a reference with a very large box size
    // This could be too small if the search is far from focus and has a tightly cropped box size
    // TODO: Consider these cases, but for now we are only considering what is valid in the sense of the image processing.
    // NOTE: On whether or not to remove these results:
    //       While it is true that a peak halfway into the template width from the edge may still be strong enough to be detectible,
    //       and thereby we create "false negatives" by enforcing this exclusion border the more important issue is to return only peak heights that are not due to the location
    //       in the image. We cannot determine how the downstream processing may compare peak to peak within an image, and so we must be conservative,
    //       in the goal of not leading these downstream analysis to errant conclusions.
    int template_padding = template_cropped_size.x / 2;

    search_image_valid_area_lower_bound_x = pre_padding_x + template_padding;
    search_image_valid_area_lower_bound_y = pre_padding_y + template_padding;
    search_image_valid_area_upper_bound_x = image_search_size.x - 1 - post_padding_x - template_padding;
    search_image_valid_area_upper_bound_y = image_search_size.y - 1 - post_padding_y - template_padding;

    number_of_valid_search_pixels = (search_image_valid_area_upper_bound_x - search_image_valid_area_lower_bound_x + 1) * (search_image_valid_area_upper_bound_y - search_image_valid_area_lower_bound_y + 1);
    MyDebugAssertTrue(number_of_valid_search_pixels > 0, "The number of valid search pixels is less than 1");
    valid_bounds_are_set = true;
};

void TemplateMatchingDataSizer::GetInputImageToEvenAndSquareOrPrimeFactoredSizePadding(int& pre_padding_x, int& pre_padding_y, int& post_padding_x, int& post_padding_y) {
    MyDebugAssertTrue(sizing_is_set, "Sizing has not been set");
    MyDebugAssertFalse(padding_is_set, "Padding has already been set");
    // There are no restrictions on the input image for this function, it may be sq or rect, even or odd,
    // but presumably there is only one layer of padding and it is >= 0
    // TODO: we need to consider the template
    int padding_x_TOTAL;
    int padding_y_TOTAL;

    if ( use_fast_fft ) {
        padding_x_TOTAL = image_pre_scaling_size.x - image_size.x;
        padding_y_TOTAL = image_pre_scaling_size.y - image_size.y;
    }
    else {
        // When not useing fast FFT there is at most one padding step from input size to a nice fourier size.
        padding_x_TOTAL = image_search_size.x - image_size.x;
        padding_y_TOTAL = image_search_size.y - image_size.y;
    }

    if ( IsEven(image_size.x) ) {
        post_padding_x = padding_x_TOTAL / 2;
        pre_padding_x  = padding_x_TOTAL / 2;
    }
    else {
        post_padding_x = padding_x_TOTAL / 2;
        pre_padding_x  = padding_x_TOTAL / 2 + 1;
    }
    if ( IsEven(image_size.y) ) {
        post_padding_y = padding_y_TOTAL / 2;
        pre_padding_y  = padding_y_TOTAL / 2;
    }
    else {
        post_padding_y = padding_y_TOTAL / 2;
        pre_padding_y  = padding_y_TOTAL / 2 + 1;
    }

    padding_is_set = true;
    return;
}

// There are no restrictions on the input image for this function, it may be sq or rect, even or odd,
// but presumably there is only one layer of padding and it is >= 0

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

#ifdef DEBUG_IMG_OUTPUT
    if ( ReturnThreadNumberOfCurrentThread( ) == 0 ) {
        // Print out the size of each step
        wxPrintf("template_size = %i %i %i\n", template_size.x, template_size.y, template_size.z);
        wxPrintf("template_pre_scaling_size = %i %i %i\n", template_pre_scaling_size.x, template_pre_scaling_size.y, template_pre_scaling_size.z);
        wxPrintf("template_cropped_size = %i %i %i\n", template_cropped_size.x, template_cropped_size.y, template_cropped_size.z);
        wxPrintf("template_search_size = %i %i %i\n", template_search_size.x, template_search_size.y, template_search_size.z);
        template_image.QuickAndDirtyWriteSlices(DEBUG_IMG_OUTPUT "/template_image.mrc", 1, template_size.z / 2);
    }
#endif
    template_image.Resize(template_pre_scaling_size.x, template_pre_scaling_size.y, template_pre_scaling_size.z, template_image.ReturnAverageOfRealValuesOnEdges( ));
#ifdef DEBUG_IMG_OUTPUT
    template_image.QuickAndDirtyWriteSlices(DEBUG_IMG_OUTPUT "/template_image_resized_pre_scale.mrc", 1, template_pre_scaling_size.z / 2);
#endif
    template_image.ForwardFFT( );
    template_image.Resize(template_cropped_size.x, template_cropped_size.y, template_cropped_size.z);
    template_image.BackwardFFT( );
#ifdef DEBUG_IMG_OUTPUT
    template_image.QuickAndDirtyWriteSlices(DEBUG_IMG_OUTPUT "/template_image_resized_cropped.mrc", 1, template_cropped_size.z / 2);
#endif
    template_image.Resize(template_search_size.x, template_search_size.y, template_search_size.z, template_image.ReturnAverageOfRealValuesOnEdges( ));
#ifdef DEBUG_IMG_OUTPUT
    if ( ReturnThreadNumberOfCurrentThread( ) == 0 ) {
        template_image.QuickAndDirtyWriteSlices(DEBUG_IMG_OUTPUT "/template_image_resized.mrc", 1, template_search_size.z / 2);
    }
#endif
};

void TemplateMatchingDataSizer::ResizeTemplate_postSearch(Image& template_image) {
    MyAssertTrue(false, "Not yet implemented");
};

void TemplateMatchingDataSizer::ResizeImage_preSearch(Image& input_image) {
    MyDebugAssertTrue(sizing_is_set, "Sizing has not been set");

    if ( resampling_is_needed ) {
        Image tmp_sq;

        tmp_sq.Allocate(image_pre_scaling_size.x, image_pre_scaling_size.y, image_pre_scaling_size.z, true);
        tmp_sq.FillWithNoiseFromNormalDistribution(0.f, 1.0f);

        input_image.ClipInto(&tmp_sq, 0.0f, false, 1.0f, 0, 0, 0, true);
#ifdef DEBUG_IMG_OUTPUT
        if ( ReturnThreadNumberOfCurrentThread( ) == 0 )
            tmp_sq.QuickAndDirtyWriteSlice(DEBUG_IMG_OUTPUT "/tmp_sq.mrc", 1);
#endif
        tmp_sq.ForwardFFT( );
        tmp_sq.Resize(image_cropped_size.x, image_cropped_size.y, image_cropped_size.z);
        tmp_sq.ZeroCentralPixel( );
        tmp_sq.DivideByConstant(sqrtf(tmp_sq.ReturnSumOfSquares( )));
        tmp_sq.BackwardFFT( );
#ifdef DEBUG_IMG_OUTPUT
        if ( ReturnThreadNumberOfCurrentThread( ) == 0 )
            tmp_sq.QuickAndDirtyWriteSlice(DEBUG_IMG_OUTPUT "/tmp_sq_resized.mrc", 1);
#endif

        input_image.Allocate(image_search_size.x, image_search_size.y, image_search_size.z, true);
        input_image.FillWithNoiseFromNormalDistribution(0.f, 1.0f);
        tmp_sq.ClipInto(&input_image, 0.0f, false, 1.0f, 0, 0, 0, true);

#ifdef DEBUG_IMG_OUTPUT
        if ( ReturnThreadNumberOfCurrentThread( ) == 0 )
            input_image.QuickAndDirtyWriteSlice(DEBUG_IMG_OUTPUT "/input_image_resized.mrc", 1);
        DEBUG_ABORT;
#endif
    }

// NOTE: rotation must always be the FINAL step in pre-processing / resizing and it is always the first to be inverted at the end.
#ifdef ROTATEFORSPEED
    if ( ! is_power_of_two(image_search_size.x) && is_power_of_two(image_search_size.y) ) {
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
#ifdef DEBUG_IMG_OUTPUT
        if ( ReturnThreadNumberOfCurrentThread( ) == 0 )
            input_image.QuickAndDirtyWriteSlice(DEBUG_IMG_OUTPUT "/input_image_rotated.mrc", 1);
#endif
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

    MyDebugAssertTrue(sizing_is_set, "Sizing has not been set");
    MyDebugAssertFalse(use_fast_fft ? is_rotated_by_90 : false, "Rotating the search image when using fastfft does  not make sense given the current square size restriction of FastFFT");
    MyDebugAssertTrue(max_intensity_projection.logical_x_dimension <= (is_rotated_by_90 ? image_size.y : image_size.x), "The max intensity projection is larger than the original image size");
    MyDebugAssertTrue(max_intensity_projection.logical_y_dimension <= (is_rotated_by_90 ? image_size.x : image_size.y), "The max intensity projection is larger than the original image size");
    MyDebugAssertTrue(pre_processed_image.at(0).is_in_memory, "The pre-processed image is not in memory");
    MyDebugAssertFalse(pre_processed_image.at(1).is_in_memory, "Chunking the search image is not supported, but the pre-processed image has allocated mem in the second chunk");
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
    tmp_mip.Allocate(image_size.x, image_size.y, image_size.z, true);
    tmp_phi.Allocate(image_size.x, image_size.y, image_size.z, true);
    tmp_theta.Allocate(image_size.x, image_size.y, image_size.z, true);
    tmp_psi.Allocate(image_size.x, image_size.y, image_size.z, true);
    tmp_defocus.Allocate(image_size.x, image_size.y, image_size.z, true);
    tmp_pixel_size.Allocate(image_size.x, image_size.y, image_size.z, true);
    tmp_sum.Allocate(image_size.x, image_size.y, image_size.z, true);
    tmp_sum_sq.Allocate(image_size.x, image_size.y, image_size.z, true);

    Image debug_mip;
    debug_mip.CopyFrom(&max_intensity_projection);
    debug_mip.ForwardFFT( );
    tmp_mip.SetToConstant(0.f);
    tmp_mip.ForwardFFT( );
    debug_mip.ClipInto(&tmp_mip);
    tmp_mip.BackwardFFT( );
    tmp_mip.QuickAndDirtyWriteSlice("/tmp/tmp_mip.mrc", 1);

    // We'll fill all the images with -FLT_MAX to indicate to downstream code that the values are not valid measurements from an experiment.
    constexpr float no_value = -std::numeric_limits<float>::max( );
    tmp_mip.SetToConstant(no_value);
    tmp_phi.SetToConstant(no_value);
    tmp_theta.SetToConstant(no_value);
    tmp_psi.SetToConstant(no_value);
    tmp_defocus.SetToConstant(no_value);
    tmp_pixel_size.SetToConstant(no_value);
    tmp_sum.SetToConstant(no_value);
    tmp_sum_sq.SetToConstant(no_value);

    long        searched_image_address = 0;
    long        out_of_bounds_value    = 0;
    long        address                = 0;
    const float actual_image_binning   = search_pixel_size / pixel_size;

    // Loop over the (possibly) binned image coordinates
    for ( int j = search_image_valid_area_lower_bound_y; j <= search_image_valid_area_upper_bound_y; j++ ) {
        int y_offset_from_origin = j - max_intensity_projection.physical_address_of_box_center_y;
        for ( int i = search_image_valid_area_lower_bound_x; i <= search_image_valid_area_upper_bound_x; i++ ) {
            // Get this pixels offset from the center of the box
            int x_offset_from_origin = i - max_intensity_projection.physical_address_of_box_center_x;

            // Scale by the binning
            // TODO: not sure if round or truncation (floor) makes more sense here
            int x_non_binned = tmp_mip.physical_address_of_box_center_x + myroundint(float(x_offset_from_origin) * actual_image_binning);
            int y_non_binned = tmp_mip.physical_address_of_box_center_y + myroundint(float(y_offset_from_origin) * actual_image_binning);

            if ( x_non_binned >= 0 && x_non_binned < tmp_mip.logical_x_dimension && y_non_binned >= 0 && y_non_binned < tmp_mip.logical_y_dimension ) {
                address                = tmp_mip.ReturnReal1DAddressFromPhysicalCoord(x_non_binned, y_non_binned, 0);
                searched_image_address = max_intensity_projection.ReturnReal1DAddressFromPhysicalCoord(i, j, 0);
            }

            else {
                // FIXME: This print block needs to be removed after initial debugging.
                wxPrintf("x_non_binned = %d, y_non_binned = %d\n", x_non_binned, y_non_binned);
                wxPrintf("%f actual_image_binning = %f\n", search_pixel_size, actual_image_binning);
                wxPrintf("tmp mip size = %d %d\n", tmp_mip.logical_x_dimension, tmp_mip.logical_y_dimension);
                wxPrintf("max_intensity_projection size = %d %d\n", max_intensity_projection.logical_x_dimension, max_intensity_projection.logical_y_dimension);
                address = -1;
            }

            // There really shouldn't be any peaks out of bounds
            // I think we should only every update an address once, so let's check it here for now.
            if ( address < 0 || address > tmp_mip.real_memory_allocated ) {
                out_of_bounds_value++;
            }
            else {
                // FIXME: This if block needs to be removed after initial debugging.
                if ( tmp_mip.real_values[address] != no_value ) {
                    wxPrintf("Address %ld already updated\n", address);
                    wxPrintf("Value is %f\n", tmp_mip.real_values[address]);
                }
                MyDebugAssertTrue(tmp_mip.real_values[address] == no_value, "Address already updated");
                tmp_mip.real_values[address]        = max_intensity_projection.real_values[searched_image_address];
                tmp_phi.real_values[address]        = best_phi.real_values[searched_image_address];
                tmp_theta.real_values[address]      = best_theta.real_values[searched_image_address];
                tmp_psi.real_values[address]        = best_psi.real_values[searched_image_address];
                tmp_defocus.real_values[address]    = best_defocus.real_values[searched_image_address];
                tmp_pixel_size.real_values[address] = best_pixel_size.real_values[searched_image_address];
                tmp_sum.real_values[address]        = correlation_pixel_sum_image.real_values[searched_image_address];
                tmp_sum_sq.real_values[address]     = correlation_pixel_sum_of_squares_image.real_values[searched_image_address];
            }
        }
    }

    MyDebugAssertTrue(out_of_bounds_value == 0, "There are out of bounds values in calculating the NN interpolation of the max intensity projection");

    // Now have the input images consume the resampled images, deallocating the search results and stealing the memory
    max_intensity_projection.Consume(&tmp_mip);
    best_psi.Consume(&tmp_psi);
    best_phi.Consume(&tmp_phi);
    best_theta.Consume(&tmp_theta);
    best_defocus.Consume(&tmp_defocus);
    best_pixel_size.Consume(&tmp_pixel_size);
    correlation_pixel_sum_image.Consume(&tmp_sum);
    correlation_pixel_sum_of_squares_image.Consume(&tmp_sum_sq);
};