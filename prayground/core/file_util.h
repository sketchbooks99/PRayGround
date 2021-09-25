#pragma once 

#include <filesystem>
#include <optional>

namespace prayground {

// Join pathes recursively as like os.path.join() in python
template <class ParentPath, class... SubPathes>
inline std::filesystem::path pathJoin(
    const ParentPath& parent_path,
    const SubPathes&... sub_pathes
)
{
    const size_t num_sub_pathes = sizeof...(sub_pathes);
    std::filesystem::path path(parent_path);
    if constexpr ( num_sub_pathes > 0 )
    {
        path.append( pathJoin( sub_pathes... ).string() );
    }
    return path;
}

std::optional<std::filesystem::path> findDataPath( const std::filesystem::path& relative_path );

std::filesystem::path pgRootDir();

void pgSetAppDir(const std::filesystem::path& dir);
std::filesystem::path pgAppDir();

// Get the extension of file. 
std::string getExtension( const std::filesystem::path& filepath );

std::filesystem::path getDir(const std::filesystem::path& filepath);

// Create a single directory.
void createDir( const std::string& abs_path );

// Create directories recursively.
void createDirs( const std::string& abs_path );

std::string getTextFromFile(const std::filesystem::path& filepath);

}