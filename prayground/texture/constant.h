#pragma once

#include <prayground/core/texture.h>

namespace prayground {

struct ConstantTextureData {
    float4 color;
};

#ifndef __CUDACC__
class ConstantTexture final : public Texture {
public:
    ConstantTexture(const float3& c, int prg_id);
    ConstantTexture(const float4& c, int prg_id);

    void setColor(const float3& c);
    void setColor(const float4& c);
    float4 color() const;

    void copyToDevice() override;
    
private:
    float4 m_color;
};

#endif

}