#pragma once 

#include "core_util.h"
#include "cudabuffer.h"
#include "object.h"

namespace pt {

enum class TextureType {
    Constant,
    Checker, 
    Image
};

#if !defined(__CUDACC__)
inline std::ostream& operator<<(std::ostream& out, TextureType type) {
    switch (type) {
    case TextureType::Constant:
        return out << "TextureType::Constant";
    case TextureType::Checker:
        return out << "TextureType::Checker";
    case TextureType::Image:
        return out << "TextureType::Image";
    default:
        return out << "";
    }
}
#endif

class Texture : public Object {
    virtual HOSTDEVICE float3 eval(float2 coord, float dpdu, float dpdv) const = 0;
    virtual HOST_ONLY TextureType type() const = 0;
}

class ConstantTexture final : public Texture {
private:
    float3 albedo;

public:
    explicit ConstantTexture(float3 a) : albedo(a) {}

    HOSTDEVICE float3 eval(float2 /* coord */, float /* dpdu */, float /* dpdv */) const override {
        return albedo;
    }

    HOST_ONLY TextureType type() const override { return TextureType::Constant; }
private:
    HOST_ONLY setup_env_on_device() override; 
    HOST_ONLY delete_env_on_device() override;
};

class CheckerTexture : public Texture {
private:
    float3 color1, color2;
    float scale;

public:
    CheckerTexture(float3 c1, float3 c2, float s=5)
    : color1(c1), color2(c2), scale(s) {}

    HOSTDEVICE float3 eval(float2 coord, float /* dpdu */, float /* dpdu */) override {
        bool is_odd = sin(scale*coord.x) * sin(scale*coord.y) < 0;
        return is_odd ? color1 : color2;
    }

    HOST_ONLY TextureType type() const override { return TextureType::Checker; }
private: 
    HOST_ONLY setup_env_on_device() override; 
    HOST_ONLY delete_env_on_device() override;
};

class ImageTexture {
private:
    float3* data;
    unsigned int width, height;

public:
    explicit ImageTexture(unsigned int w, unsigned int h) : width(w), height(h) {
        CUDABuffer<float3> data_buffer;
        data_buffer.allocate(width*height*sizeof(float3));
        data = data_buffer.data();
    }
    
private:
    HOST_ONLY setup_env_on_device() override; 
    HOST_ONLY delete_env_on_device() override;
}

}
