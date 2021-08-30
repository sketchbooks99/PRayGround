#pragma once
#include <prayground/core/texture.h>

namespace prayground {

struct CheckerTextureData {
    float3 color1;
    float3 color2;
    float scale;
};

#ifndef __CUDACC__
class CheckerTexture final : public Texture {
public:
    CheckerTexture(const float3& c1, const float3& c2, float s=5);

    void setColor1(const float3& c1);
    float3 color1() const;

    void setColor2(const float3& c2);
    float3 color2() const;

    void setScale(const float s);
    float scale() const;

    void copyToDevice() override;
private:
    float3 m_color1, m_color2;
    float m_scale;
}; 

#endif // __CUDACC__

}