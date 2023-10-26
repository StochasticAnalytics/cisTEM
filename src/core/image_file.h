/*
 * Provide a common interface to interact with all the different
 * files we support (MRC, DM, TIF, etc.)
 */

#include "../constants/constants.h"

class ImageFile : public AbstractImageFile {

  private:
    using img_file_t = cistem::supported_image_file_types::Enum;

    // These are the actual file objects doing the work
    MRCFile  mrc_file;
    TiffFile tiff_file;
    DMFile   dm_file;
    EerFile  eer_file;

    img_file_t file_type;

    wxString file_type_string;
    void     SetFileTypeFromExtension( );

  public:
    ImageFile( );
    ImageFile(std::string wanted_filename, bool overwrite = false);
    ~ImageFile( );

    inline img_file_t ReturnFileType( ) { return file_type; };

    int   ReturnXSize( );
    int   ReturnYSize( );
    int   ReturnZSize( );
    int   ReturnNumberOfSlices( );
    float ReturnPixelSize(int dim = 0);

    float ReturnPixelSize_X( ) {
        if ( file_type == img_file_t::MRC_FILE )
            return mrc_file.ReturnPixelSize_X( );
        else
            return ReturnPixelSize( );
    }

    float ReturnPixelSize_Y( ) {
        if ( file_type == img_file_t::MRC_FILE )
            return mrc_file.ReturnPixelSize_Y( );
        else
            return ReturnPixelSize( );
    }

    float ReturnPixelSize_Z( ) {
        if ( file_type == img_file_t::MRC_FILE )
            return mrc_file.ReturnPixelSize_Z( );
        else
            return ReturnPixelSize( );
    }

    void SetPixelSize(float new_pixel_size_x, float new_pixel_size_y, float new_pixel_size_z) {
        MyDebugAssertTrue(file_type == img_file_t::MRC_FILE, "Only implemented for MRC files");
        mrc_file.SetPixelSize(new_pixel_size_x, new_pixel_size_y, new_pixel_size_z);
    }

    void SetPixelSize(float new_pixel_size) {
        SetPixelSize(new_pixel_size, new_pixel_size, new_pixel_size);
    }

    bool IsOpen( );

    bool OpenFile(std::string wanted_filename, bool overwrite, bool wait_for_file_to_exist = false, bool check_only_the_first_image = false, int eer_super_res_factor = 1, int eer_frames_per_image = 0);
    void CloseFile( );

    void ReadSliceFromDisk(int slice_number, float* output_array);
    void ReadSlicesFromDisk(int start_slice, int end_slice, float* output_array);

    void WriteSliceToDisk(int slice_number, float* input_array);
    void WriteSlicesToDisk(int start_slice, int end_slice, float* input_array);

    void PrintInfo( );
};
