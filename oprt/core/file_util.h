#include "util.h"

namespace oprt {

/**
 * @brief Join pathes recursively as like \c os.path.join() in python
 */
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

/**
 * @brief 
 * Check if the file specified by the relative path exists
 * in the data/ or the root directory of OptiX-Raytracer.
 * If the file exists, return the absolute path to the file, 
 * but if not, throw the runtime error that file is not found.
 */
std::optional<std::filesystem::path> findDatapath( const std::filesystem::path& relative_path );

std::filesystem::path rootDir();

/**
 * @brief Get the extension of file.
 */
std::string getExtension( const std::filesystem::path& filepath );

/**
 * @brief Create a single directory.
 */
void createDir( const std::string& abs_path );

/**
 * @brief Create directories recursively.
 */
void createDirs( const std::string& abs_path );

}