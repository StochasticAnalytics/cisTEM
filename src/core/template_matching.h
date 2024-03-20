class Image;
class ImageFile;

class TemplateMatchFoundPeakInfo {
  public:
    int   peak_number;
    float x_pos;
    float y_pos;
    float psi;
    float theta;
    float phi;
    float defocus;
    float pixel_size;
    float peak_height;
    int   original_peak_number;
    int   new_peak_number;
};

WX_DECLARE_OBJARRAY(TemplateMatchFoundPeakInfo, ArrayOfTemplateMatchFoundPeakInfos);

class TemplateMatchJobResults {
  public:
    TemplateMatchJobResults( );

    wxString job_name;
    int      job_type;
    long     input_job_id;
    long     job_id;
    long     datetime_of_run;
    long     image_asset_id;
    long     ref_volume_asset_id;
    wxString symmetry;
    float    pixel_size;
    float    voltage;
    float    spherical_aberration;
    float    amplitude_contrast;
    float    defocus1;
    float    defocus2;
    float    defocus_angle;
    float    phase_shift;
    float    low_res_limit;
    float    high_res_limit;
    float    out_of_plane_step;
    float    in_plane_step;
    float    defocus_search_range;
    float    defocus_step;
    float    pixel_size_search_range;
    float    pixel_size_step;
    float    mask_radius;
    float    min_peak_radius;
    float    xy_change_threshold;
    bool     exclude_above_xy_threshold;

    wxString mip_filename;
    wxString scaled_mip_filename;
    wxString psi_filename;
    wxString theta_filename;
    wxString phi_filename;
    wxString defocus_filename;
    wxString pixel_size_filename;
    wxString histogram_filename;
    wxString projection_result_filename;
    wxString avg_filename;
    wxString std_filename;

    float refinement_threshold;
    float used_threshold;
    float reference_box_size_in_angstroms;

    ArrayOfTemplateMatchFoundPeakInfos found_peaks;
    ArrayOfTemplateMatchFoundPeakInfos peak_changes;
};

WX_DECLARE_OBJARRAY(TemplateMatchJobResults, ArrayOfTemplateMatchJobResults);

/**
 * @brief Resize and or decompose an image and template and track changes as well as valid regions for convolution.
 * Methods to reconstruct the valid areas at the original image size via nearest neighbor interpolation.
 * 
 * Assumes a 2d search image, not necessarily square, and a cubic template.
 * 
 */
class TemplateMatchImageSizer {

    float input_pixel_size;
    float search_pixel_size;
    float high_resolution_limit;
    int   input_image_size_x;
    int   input_image_size_y;
    int   image_pre_scaling_size_x;
    int   image_pre_scaling_size_y;
    int   image_cropped_size_x;
    int   image_cropped_size_y;
    int   image_search_size_x;
    int   image_search_size_y;
    int   template_size;
    int   template_pre_scaling_size;
    int   template_cropped_size;
    int   template_search_size;

  public:
    TemplateMatchImageSizer(const Image& input_image,
                            const Image& input_template,
                            const float  pixel_size,
                            const float  wanted_high_resolution_limit);

    TemplateMatchImageSizer(const std::string& template_image_sizer_info);

    void ReadMetaDataFromImage(const std::string& wanted_filename);
    void WriteMetaDataToImage(const std::string& wanted_filename);
};

// /**
//  * @brief Find peaks in a 2D image.
//  *
//  */
// class TemplateMatchPeakFinder( ) {
// }