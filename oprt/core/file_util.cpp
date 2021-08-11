#include "file_util.h"
#include <sutil/sutil.h>

namespace oprt {

// -------------------------------------------------------------------------------s
std::optional<std::filesystem::path> findDataPath( const std::filesystem::path& relative_path )
{
    std::array<std::string, 2> parent_dirs = 
    {
        rootDir().string(),
        pathJoin(OPRT_ROOT_DIR, "data").string()
    };

    for (auto &parent : parent_dirs)
    {
        auto filepath = pathJoin(parent, relative_path);
        if ( std::filesystem::exists(filepath) )
            return filepath;
    }
    return std::nullopt;
}

// -------------------------------------------------------------------------------
std::filesystem::path rootDir() {
    return std::filesystem::path(OPRT_ROOT_DIR);
}

// -------------------------------------------------------------------------------
std::string getExtension( const std::filesystem::path& filepath )
{
    return filepath.has_extension() ? filepath.extension().string() : "";
}

// -------------------------------------------------------------------------------
void createDir( const std::string& abs_path )
{
    // Check if the directory is existed.
    if (std::filesystem::exists(abs_path)) {
        Message(MSG_WARNING, "The directory '", abs_path, "' is already existed.");
        return;
    }
    // Create new directory with path specified.
    bool result = std::filesystem::create_directory(abs_path);
    Assert(result, "Failed to create directory '" + abs_path + "'.");
}

// -------------------------------------------------------------------------------
void createDirs( const std::string& abs_path )
{
    // Check if the directory is existed.
    if (std::filesystem::exists(abs_path)) {
        Message(MSG_WARNING, "The directory '", abs_path, "' is already existed.");
        return;
    }
    bool result = std::filesystem::create_directories( abs_path );
    Assert(result, "Failed to create directories '" + abs_path + "'.");
}

}