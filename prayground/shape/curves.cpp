#include "curves.h"
#include <prayground/core/cudabuffer.h>
#include <prayground/core/file_util.h>
#include <prayground/math/util.h>

namespace prayground {

    namespace fs = std::filesystem;

    Curves::Curves() {}

    Curves::Curves(
        const std::vector<Vec3f>& vertices, 
        const std::vector<int32_t>& indices, 
        const std::vector<Vec3f>& normals, 
        const std::vector<float>& widths, 
        const std::vector<uint32_t>& sbt_indices) 
        : m_vertices(vertices)
        , m_indices(indices)
        , m_normals(normals)
        , m_widths(widths)
        , m_sbt_indices(sbt_indices)
    {

    }

    constexpr ShapeType Curves::type() 
    {
        return ShapeType::Curves;
    }

} // namespace prayground