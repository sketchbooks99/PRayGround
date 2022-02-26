#pragma once

#include <prayground/core/texture.h>

namespace prayground {

template <typename T>
struct ConstantTextureData {
    T color;
};

template <typename T>
class ConstantTexture_ final : public Texture {
public:
    using ColorType = T;
    struct Data
    {
        T color;
    };

#ifndef __CUDACC__
    ConstantTexture_(const T& c, int prg_id)
        : Texture(prg_id), m_color(c)
    {}

    void setColor(const T& c)
    {
        m_color = c;
    }
    T color() const
    {
        return m_color;
    }

    void copyToDevice() override
    {
        Data data = { m_color };

        if (!d_data)
            CUDA_CHECK(cudaMalloc(&d_data, sizeof(Data)));
        CUDA_CHECK(cudaMemcpy(
            d_data,
            &data, sizeof(Data),
            cudaMemcpyHostToDevice
        ));
    }
    
private:
    T m_color;
#endif // __CUDACC__
};

}