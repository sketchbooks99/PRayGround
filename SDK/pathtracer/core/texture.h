#pragma once 

#include "core_util.h"
#include "cudabuffer.h"

namespace pt {

struct ConstantTexture {
    ConstantTexture(float3 a) : albedo(a) {}
    float3 albedo;
};

struct CheckerTexture {
    CheckerTexture(float3 c1, float3 c2, float s)
    : color1(c1), color2(c2), scale(s) {}
    float3 color1, color2;
    float scale;
};

struct ImageTexture {
    ImageTexture(unsigned int w, unsigned int h) { 
        /// TODO: allocate image with specified dimensions.
    }
    CUDABuffer<float3> image;
    unsigned int width, height;
}

}
