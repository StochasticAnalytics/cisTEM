#include "core_headers.h"

#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
WX_DEFINE_OBJARRAY(ArrayOfTemplateMatchFoundPeakInfos);
WX_DEFINE_OBJARRAY(ArrayOfTemplateMatchJobResults);

TemplateMatchJobResults::TemplateMatchJobResults( ) {
    job_name                   = "";
    job_type                   = -1;
    input_job_id               = -1;
    job_id                     = -1;
    datetime_of_run            = 0;
    image_asset_id             = -1;
    ref_volume_asset_id        = -1;
    symmetry                   = "C1";
    pixel_size                 = 0.0f;
    voltage                    = 0.0f;
    spherical_aberration       = 0.0f;
    amplitude_contrast         = 0.0f;
    defocus1                   = 0.0f;
    defocus2                   = 0.0f;
    defocus_angle              = 0.0f;
    phase_shift                = 0.0f;
    low_res_limit              = 0.0f;
    high_res_limit             = 0.0f;
    out_of_plane_step          = 0.0f;
    in_plane_step              = 0.0f;
    defocus_search_range       = 0.0f;
    defocus_step               = 0.0f;
    pixel_size_search_range    = 0.0f;
    pixel_size_step            = 0.0f;
    mask_radius                = 0.0f;
    min_peak_radius            = 0.0f;
    xy_change_threshold        = 0.0f;
    exclude_above_xy_threshold = false;

    mip_filename               = "";
    scaled_mip_filename        = "";
    psi_filename               = "";
    theta_filename             = "";
    phi_filename               = "";
    defocus_filename           = "";
    pixel_size_filename        = "";
    histogram_filename         = "";
    projection_result_filename = "";
    avg_filename               = "";
    std_filename               = "";

    refinement_threshold            = 0.0f;
    used_threshold                  = 0.0f;
    reference_box_size_in_angstroms = 0;
}

TemplateMatchImageSizer::TemplateMatchImageSizer(const Image& input_image,
                                                 const Image& input_template,
                                                 const float  pixel_size,
                                                 const float  wanted_high_resolution_limit) : input_pixel_size(pixel_size),
                                                                                             input_image_size_x(input_image.logical_x_dimension),
                                                                                             input_image_size_y(input_image.logical_y_dimension),
                                                                                             image_pre_scaling_size_x(input_image.logical_x_dimension),
                                                                                             image_pre_scaling_size_y(input_image.logical_y_dimension),
                                                                                             image_cropped_size_x(input_image.logical_x_dimension),
                                                                                             image_cropped_size_y(input_image.logical_y_dimension),
                                                                                             image_search_size_x(input_image.logical_x_dimension),
                                                                                             image_search_size_y(input_image.logical_y_dimension),
                                                                                             template_size(input_template.logical_x_dimension),
                                                                                             template_pre_scaling_size(input_template.logical_x_dimension),
                                                                                             template_cropped_size(input_template.logical_x_dimension),
                                                                                             template_search_size(input_template.logical_x_dimension) {

    // This should not occur, and may be silently ignoring an error the user would have preferred to know about.
    // TODO: This should probably throw an error and have the user enter the correct pixel size.
    high_resolution_limit = (wanted_high_resolution_limit < 2.0f * pixel_size) ? 2.0f * pixel_size : wanted_high_resolution_limit;
}

/**
 * @brief Construct a new Template Match Image Sizer:: Template Match Image Sizer object
 *  Rather than deal with textfiles, the size values may be stored in an mrc image, where the values are in
 *  the real_values array. The order of the values is defined in the match_template::Enum.
 * @param template_image_sizer_info 
 */
TemplateMatchImageSizer::TemplateMatchImageSizer(const std::string& template_image_sizer_info) {
    ReadMetaDataFromImage(template_image_sizer_info);
}

void TemplateMatchImageSizer::ReadMetaDataFromImage(const std::string& template_image_sizer_info) {
    Image size_info;
    size_info.QuickAndDirtyReadSlice(template_image_sizer_info, 1);
    input_pixel_size          = size_info.real_values[cistem::match_template::input_pixel_size];
    search_pixel_size         = size_info.real_values[cistem::match_template::search_pixel_size];
    high_resolution_limit     = size_info.real_values[cistem::match_template::search_pixel_size] * 2.0f;
    input_image_size_x        = size_info.real_values[cistem::match_template::image_size_x];
    input_image_size_y        = size_info.real_values[cistem::match_template::image_size_y];
    image_pre_scaling_size_x  = size_info.real_values[cistem::match_template::image_pre_scaling_size_x];
    image_pre_scaling_size_y  = size_info.real_values[cistem::match_template::image_pre_scaling_size_y];
    image_cropped_size_x      = size_info.real_values[cistem::match_template::image_cropped_size_x];
    image_cropped_size_y      = size_info.real_values[cistem::match_template::image_cropped_size_y];
    image_search_size_x       = size_info.real_values[cistem::match_template::image_search_size_x];
    image_search_size_y       = size_info.real_values[cistem::match_template::image_search_size_y];
    template_size             = size_info.real_values[cistem::match_template::template_size];
    template_pre_scaling_size = size_info.real_values[cistem::match_template::template_pre_scaling_size];
    template_cropped_size     = size_info.real_values[cistem::match_template::template_cropped_size];
    template_search_size      = size_info.real_values[cistem::match_template::template_search_size];
}

void TemplateMatchImageSizer::WriteMetaDataToImage(const std::string& wanted_filename) {
    Image size_info;
    size_info.Allocate(cistem::match_template::number_of_meta_data_values, 1, 1, true);
    size_info.real_values[cistem::match_template::input_pixel_size]          = input_pixel_size;
    size_info.real_values[cistem::match_template::search_pixel_size]         = search_pixel_size;
    size_info.real_values[cistem::match_template::image_size_x]              = input_image_size_x;
    size_info.real_values[cistem::match_template::image_size_y]              = input_image_size_y;
    size_info.real_values[cistem::match_template::image_pre_scaling_size_x]  = image_pre_scaling_size_x;
    size_info.real_values[cistem::match_template::image_pre_scaling_size_y]  = image_pre_scaling_size_y;
    size_info.real_values[cistem::match_template::image_cropped_size_x]      = image_cropped_size_x;
    size_info.real_values[cistem::match_template::image_cropped_size_y]      = image_cropped_size_y;
    size_info.real_values[cistem::match_template::image_search_size_x]       = image_search_size_x;
    size_info.real_values[cistem::match_template::image_search_size_y]       = image_search_size_y;
    size_info.real_values[cistem::match_template::template_size]             = template_size;
    size_info.real_values[cistem::match_template::template_pre_scaling_size] = template_pre_scaling_size;
    size_info.real_values[cistem::match_template::template_cropped_size]     = template_cropped_size;
    size_info.real_values[cistem::match_template::template_search_size]      = template_search_size;
    size_info.QuickAndDirtyWriteSlice(wanted_filename, 1, false, search_pixel_size);
}