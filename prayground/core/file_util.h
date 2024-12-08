#pragma once 

#include <filesystem>
#include <optional>

namespace prayground {

    // Join pathes recursively as like os.path.join() in python
    template <class ParentPath, class... SubPathes>
    inline std::filesystem::path pgPathJoin(
        const ParentPath& parent_path,
        const SubPathes&... sub_pathes
    )
    {
        const size_t num_sub_pathes = sizeof...(sub_pathes);
        std::filesystem::path path(parent_path);
        if constexpr ( num_sub_pathes > 0 )
        {
            path.append( pgPathJoin( sub_pathes... ).string() );
        }
        return path;
    }

    std::filesystem::path pgGetExecutableDir();

    // Check if the file specified by relative path exists. 
    // Parent directories to seek the file are pgRootDir(), pgAppDir(), pgAppDir()/data and <path/to/app.exe>
    std::optional<std::filesystem::path> pgFindDataPath( const std::filesystem::path& relative_path );

    // Return absolute path to root directory of PRayGround
    std::filesystem::path pgRootDir();

    // Set absolute path to app directory
    void pgSetAppDir(const std::filesystem::path& dir);
    // Get absolute path to the app directory
    // This return the empty path unless to set the path to app using pgSetAppDir(APP_DIR). 
    std::filesystem::path pgAppDir();

    void pgAddSearchDir(const std::filesystem::path& dir);

    /// @todo: Add 'pg' prefix to get~~() functions
    // Get the extension of file. 
    std::string pgGetExtension( const std::filesystem::path& filepath );

    std::string pgGetStem(const std::filesystem::path& filepath, bool is_dir = true);

    std::filesystem::path pgGetFilename(const std::filesystem::path& filepath);
    std::filesystem::path pgGetDir(const std::filesystem::path& filepath);

    // Create a single directory
    void pgCreateDir( const std::filesystem::path& abs_path );

    // Create directories recursively
    void pgCreateDirs( const std::filesystem::path& abs_path );

    // Extract text data from the file
    std::string pgGetTextFromFile(const std::filesystem::path& filepath);

} // namespace prayground