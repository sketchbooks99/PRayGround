#include "util.h"
#include <filesystem>

namespace oprt {

/**
 * @brief Join pathes recursively as like \c os.path.join() in python
 */
template <class ParentPath, class... SubPathes>
inline std::filesystem::path path_join(
    const ParentPath& parent_path,
    const SubPathes&... sub_pathes
)
{
    const size_t num_sub_pathes = sizeof...(sub_pathes);
    std::filesystem::path path(parent_path);
    if constexpr ( num_sub_pathes > 0 )
    {
        path.append( path_join( sub_pathes... ).string() );
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
std::filesystem::path find_datapath( const std::filesystem::path& relative_path );

/**
 * @brief Get the extension of file.
 */
std::string get_extension( const std::filesystem::path& filepath );

/**
 * @brief Create a single directory.
 */
void create_dir( const std::string& abs_path );

/**
 * @brief Create directories recursively.
 */
void create_dirs( const std::string& abs_path );

}