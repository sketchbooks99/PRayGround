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

class Texture : public DeviceCallableObject {
    virtual HOSTDEVICE float3 eval(float2 coord, float dpdu, float dpdv) const = 0;
    virtual HOST TextureType type() const = 0;
}

class ConstantTexture final : public Texture {
private:
    float3 albedo;

public:
    explicit HOSTDEVICE ConstantTexture(float3 a) : albedo(a) {}
    
    // Copy constructor
    explicit HOSTDEVICE ConstantTexture(const ConstantTexture& constant)
    : albedo(constant.get_albedo()) {}

    HOSTDEVICE float3 eval(float2 /* coord */, float /* dpdu */, float /* dpdv */) const override {
        return albedo;
    }

    HOSTDEVICE float3 get_albedo() { return albedo; }

    HOST TextureType type() const override { return TextureType::Constant; }
private:
    HOST setup_on_device() override;
    HOST delete_on_device() override;
};

class CheckerTexture final : public Texture {
private:
    float3 color1, color2;
    float scale;

public:
    HOSTDEVICE CheckerTexture(float3 c1, float3 c2, float s=5)
    : color1(c1), color2(c2), scale(s) {}

    // Copy constructor
    HOSTDEVICE CheckerTexture(const CheckerTexture& checker)
    : color1(checker.get_color1()), color2(checker.get_color2()), scale(checker.get_scale()) {}

    HOSTDEVICE float3 eval(float2 coord, float /* dpdu */, float /* dpdu */) override {
        bool is_odd = sin(scale*coord.x) * sin(scale*coord.y) < 0;
        return is_odd ? color1 : color2;
    }

    HOST TextureType type() const override { return TextureType::Checker; }

    HOSTDEVICE float3 get_color1() { return color1; }
    HOSTDEVICE float3 get_color2() { return color2; }
    HOSTDEVICE float scale() { return scale; }
private: 
    HOST setup_on_device() override; 
    HOST delete_on_device() override;
};

class ImageTexture final : public Texture {
private:
    float3* data;
    unsigned int width, height;

public:
    explicit HOST ImageTexture(const std::string& filename) {}

    explicit HOSTDEVICE ImageTexture(unsigned int w, unsigned int h) : width(w), height(h) {
#if !defined(__CUDACC__)
        CUDABuffer<float3> data_buffer;
        data_buffer.allocate(width*height*sizeof(float3));
        data = data_buffer.data();
#endif
    }

    explicit HOSTDEVICE ImageTexture(const ImageTexture& image)
    : 

    HOST TextureType type() const override { return TextureType::Image; }
    
private:
    HOST setup_on_device() override; 
    HOST delete_on_device() override;
}

}
