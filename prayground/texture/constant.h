#pragma once

#include <prayground/core/texture.h>

namespace prayground {

struct ConstantTextureData {
    float3 color;
};

#ifndef __CUDACC__
class ConstantTexture final : public Texture {
public:
    ConstantTexture(const float3& c);

    void setColor(const float3& c);
    float3 color() const;

    void copyToDevice() override;
    
private:
    float3 m_color;
};

#endif

}