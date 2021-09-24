#include "checker.h"

namespace prayground {

// ------------------------------------------------------------------
CheckerTexture::CheckerTexture(const float3& c1, const float3& c2, float s, int prg_id)
: Texture(prg_id), m_color1(c1), m_color2(c2), m_scale(s)
{

}

// ------------------------------------------------------------------
void CheckerTexture::setColor1(const float3& c1)
{
    m_color1 = c1;
}

float3 CheckerTexture::color1() const 
{
    return m_color1;
}

// ------------------------------------------------------------------
void CheckerTexture::setColor2(const float3& c2)
{
    m_color2 = c2;
}

float3 CheckerTexture::color2() const 
{
    return m_color2;
}

// ------------------------------------------------------------------
void CheckerTexture::setScale(const float s)
{
    m_scale = s;
}

float CheckerTexture::scale() const 
{
    return m_scale;
}

// ------------------------------------------------------------------
void CheckerTexture::copyToDevice()
{
    CheckerTextureData data = {
        m_color1, 
        m_color2, 
        m_scale
    };

    if (!d_data) CUDA_CHECK(cudaMalloc(&d_data, sizeof(CheckerTextureData)));
    CUDA_CHECK(cudaMemcpy(
        d_data,
        &data, sizeof(CheckerTextureData),
        cudaMemcpyHostToDevice
    ));
}

} // ::prayground