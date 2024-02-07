#include "constant.h"

namespace prayground {

    // ------------------------------------------------------------------
    ConstantTexture::ConstantTexture(const float3& c, int prg_id)
        : Texture(prg_id), m_color(make_float4(c, 1.0f))
    {

    }


    ConstantTexture::ConstantTexture(const float4& c, int prg_id)
        : Texture(prg_id), m_color(c)
    {

    }

    // ------------------------------------------------------------------
    void ConstantTexture::setColor(const float3& c)
    {
        m_color = make_float4(c, 1.0f);
    }

    void ConstantTexture::setColor(const float4& c)
    {
        m_color = c;
    }

    float4 ConstantTexture::color() const 
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

} // namespace prayground