#include "shape.h"

namespace prayground {

void Shape::setSbtIndex(const uint32_t sbt_idx)
{
    m_sbt_index = sbt_idx;
}

uint32_t Shape::sbtIndex() const
{
    return m_sbt_index;
}

void Shape::free()
{
    freeAabbBuffer();
    cuda_free(d_data);
}

void Shape::freeAabbBuffer()
{
    if (d_aabb_buffer) cuda_free(d_aabb_buffer);
}

void* Shape::devicePtr() const
{
    return d_data;
}

} // ::prayground
