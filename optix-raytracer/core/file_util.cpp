#include "file_util.h"
#include <sutil/sutil.h>

namespace oprt {

// -------------------------------------------------------------------------------s
std::filesystem::path find_datapath( const std::filesystem::path& relative_path )
{
    std::array<std::string, 2> parent_dirs = 
    {
        OPTIX_RAYTRACER_DIR,
        path_join(OPTIX_RAYTRACER_DIR, "data").string()
    };

    for (auto &parent : parent_dirs)
    {
        auto filepath = path_join(parent, relative_path);
        if ( std::filesystem::exists(filepath) )
            return filepath;
    }
    Throw("Failed to find datapath '" + relative_path.string() + "'.");
}

// -------------------------------------------------------------------------------
std::string get_extension( const std::filesystem::path& filepath )
{
    return filepath.has_extension() ? filepath.extension().string() : "";
}

// -------------------------------------------------------------------------------
void create_dir( const std::string& abs_path )
{
    // Check if the directory is existed.
    if (std::filesystem::exists(abs_path)) {
        Message("The directory '", abs_path, "' is already existed.");
        return;
    }
    // Create new directory with path specified.
    bool result = std::filesystem::create_directory(abs_path);
    Assert(result, "Failed to create directory '" + abs_path + "'.");
}

// -------------------------------------------------------------------------------
void create_dirs( const std::string& abs_path )
{
    // Check if the directory is existed.
    if (std::filesystem::exists(abs_path)) {
        Message("The directory '", abs_path, "' is already existed.");
        return;
    }
    bool result = std::filesystem::create_directories( abs_path );
    Assert(result, "Failed to create directories '" + abs_path + "'.");
}

}