#include "file_util.h"
#include <sutil/sutil.h>

namespace oprt {

namespace fs = std::filesystem;

// -------------------------------------------------------------------------------s
std::optional<fs::path> findDataPath( const fs::path& relative_path )
{
#ifdef OPRT_APP_DIR
    std::array<std::string, 5> parent_dirs = 
    {
        oprtAppDir().string(), 
        pathJoin(oprtAppDir(), "data").string(), 
#else 
    std::array<std::string, 3> parent_dirs = 
    {
#endif
        "", // @note これで絶対パスにも対応できる？
        oprtRootDir().string(),
        pathJoin(OPRT_ROOT_DIR, "data").string()
    };

    for (auto &parent : parent_dirs)
    {
        auto filepath = pathJoin(parent, relative_path);
        if ( fs::exists(filepath) )
            return filepath;
    }
    return std::nullopt;
}

// -------------------------------------------------------------------------------
fs::path oprtRootDir() {
    return fs::path(OPRT_ROOT_DIR);
}

#ifdef OPRT_APP_DIR
fs::path oprtAppDir()
{
    return fs::path(OPRT_APP_DIR);
}
#endif

// -------------------------------------------------------------------------------
std::string getExtension( const fs::path& filepath )
{
    return filepath.has_extension() ? filepath.extension().string() : "";
}

// -------------------------------------------------------------------------------
void createDir( const fs::path& abs_path )
{
    // Check if the directory is existed.
    if (fs::exists(abs_path)) {
        Message(MSG_WARNING, "The directory '", abs_path, "' is already existed.");
        return;
    }
    // Create new directory with path specified.
    bool result = fs::create_directory(abs_path);
    Assert(result, "Failed to create directory '" + abs_path.string() + "'.");
}

// -------------------------------------------------------------------------------
void createDirs( const fs::path& abs_path )
{
    // Check if the directory is existed.
    if (fs::exists(abs_path)) {
        Message(MSG_WARNING, "The directory '", abs_path, "' is already existed.");
        return;
    }
    bool result = fs::create_directories( abs_path );
    Assert(result, "Failed to create directories '" + abs_path.string() + "'.");
}

// -------------------------------------------------------------------------------
std::string getTextFromFile(const fs::path& relative_path)
{
    std::optional<fs::path> filepath = findDataPath(relative_path);
    Assert(filepath, "oprt::getTextFromFile(): A text file with the path '" + relative_path.string() + "' is not found.");

    std::ifstream file_stream; 
    try
    {
        file_stream.open(filepath.value());
        std::stringstream content_stream;
        content_stream << file_stream.rdbuf();
        file_stream.close();
        return content_stream.str();
    }
    catch(const std::istream::failure& e)
    {
        Message(MSG_ERROR, "oprt::getTextureFromFile(): Failed to load text file due to '" + std::string(e.what()) + "'.");
        return "";
    }
}

}