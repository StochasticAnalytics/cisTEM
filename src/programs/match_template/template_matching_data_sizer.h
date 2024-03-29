#ifndef __SRC_PROGRAMS_MATCH_TEMPLATE_TEMPLATE_MATCHING_DATA_SIZER_H_
#define __SRC_PROGRAMS_MATCH_TEMPLATE_TEMPLATE_MATCHING_DATA_SIZER_H_

#define RESIZE_TEST

/**
 * @brief This class is used to optionally resample, pad or cut into chunks the search image.
 * 
 * It relies on knowing the template size, particularly in the case of chunking.
 * This is a draft object and will be moved to a more appropriate location. TODO
 * 
 */
class TemplateMatchingDataSizer {

    constexpr bool MUST_BE_POWER_OF_TWO   = false; // Required for half-precision xforms
    constexpr bool MUST_BE_FACTOR_OF_FOUR = true; // May be faster

    // This is a non-data owning class, but we want references to the underlying image/template data
    int4 image_size;
    int4 image_pre_scaling_size;
    int4 image_cropped_size;
    int4 image_search_size;

    int4 template_size;
    int4 template_pre_scaling_size;
    int4 template_cropped_size;
    int4 template_search_size;

    float pixel_size{ };
    float search_pixel_size{ };
    float template_padding{ };
    float high_resolution_limit{-1.f};
    bool  resampling_is_needed{false};
    bool  is_rotated_by_90{false};

    std::vector<int> primes;

    constexpr float max_reduction_by_fraction_of_reference = 0.000001f; // FIXME the cpu version is crashing when the image is reduced, but not the GPU

    float max_increase_by_fraction_of_image;

    void SetHighResolutionLimit(const float wanted_high_resolution_limit);
    int  GetGenericFFTSize( );
    void GetResampledFFTSize( );
    void CheckSizing( );

  public:
    TemplateMatchingDataSizer(Image& input_image, Image& template, const float pixel_size, const float wanted_template_padding);

    // Don't allow copy or move. FIXME: if we don't add any dynamically allocated data, we can remove this.
    TemplateMatchingDataSizer(const TemplateMatchingDataSizer&)            = delete;
    TemplateMatchingDataSizer& operator=(const TemplateMatchingDataSizer&) = delete;
    TemplateMatchingDataSizer(TemplateMatchingDataSizer&&)                 = delete;
    TemplateMatchingDataSizer& operator=(TemplateMatchingDataSizer&&)      = delete;

    void SetImageAndTemplateSizing(const float wanted_high_resolution_limit, const bool use_fast_fft);
    void PreProcessInputImage(Image& input_image);

    void ResizeTemplate_preSearch(Image& template_image);
    void ResizeTemplate_postSearch(Image& template_image);

    void ResizeImage_preSearch(Image& input_image);
    void ResizeImage_postSearch(Image& input_image);

    inline void PrintImageSizes( ) {
        if ( ReturnThreadNumberOfCurrentThread( ) == 0 ) {
            wxPrintf("old x, y = %i %i\n  new x, y = %i %i\n", image_size.x, image_size.y, image_search_size.x, image_search_size.y);
        }
    }

    inline bool IsResamplingNeeded( ) {
        return resampling_is_needed;
    }
};

#endif