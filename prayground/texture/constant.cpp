#include "constant.h"

namespace prayground {

// ------------------------------------------------------------------
ConstantTexture::ConstantTexture(const float3& c, int prg_id)
: m_color(c), Texture(prg_id)
{

}

// ------------------------------------------------------------------
void ConstantTexture::setColor(const float3& c)
{
    m_color = c;
}

float3 ConstantTexture::color() const 
{
    return m_color;
}

// ------------------------------------------------------------------
void ConstantTexture::copyToDevice()
{
    ConstantTextureData data = { m_color };

    if (!d_data) 
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(ConstantTextureData)));
    CUDA_CHECK(cudaMemcpy(
        d_data,
        &data, sizeof(ConstantTextureData),
        cudaMemcpyHostToDevice
    ));
}

} // ::prayground