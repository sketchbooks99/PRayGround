#pragma once
#include <prayground/core/texture.h>

namespace prayground {

template <typename T>
struct CheckerTextureData {
    T color1;
    T color2;
    float scale;
};

template <typename T>
class CheckerTexture_ final : public Texture {
public:
    using DataType = T;
    // using Data = CheckerTextureData<T>;
    struct Data
    {
        T color1;
        T color2;
        float scale;
    };

    CheckerTexture_(const T& c1, const T& c2, float s, int prg_id)
        : Texture(prg_id), m_color1(c1), m_color2(c2), m_scale(s)
    {}

#ifndef __CUDACC__
    void setColor1(const T& c1)
    {
        m_color1 = c1;
    }
    T color1() const
    {
        return m_color1;
    }

    void setColor2(const T& c2)
    {
        m_color2 = c2;
    }
    T color2() const
    {
        return m_color2;
    }

    void setScale(const float& s)
    {
        m_scale = s;
    }
    float scale() const
    {
        return m_scale;
    }

    void copyToDevice() override
    {
        Data data =
        {
            m_color1, 
            m_color2, 
            m_scale
        };

        if (!d_data) CUDA_CHECK(cudaMalloc(&d_data, sizeof(Data)));
        CUDA_CHECK(cudaMemcpy(
            d_data,
            &data, sizeof(Data),
            cudaMemcpyHostToDevice
        ));
    }
private:
    T m_color1, m_color2;
    float m_scale;
#endif // __CUDACC__

};

}