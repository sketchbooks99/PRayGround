#pragma once 

#include <optix.h>
#include <sutil/vec_math.h>

enum class TexType {
    CONATANT = 1u << 0,
    CHECKER = 1u << 1,
    IMAGE = 1u << 2  
};

// This is abstract class for readability
class Texture {
public:
    Texture(TexType textype) : textype(textype) {}
    const bool isEqualType(TexType textype) {
        return this->textype == textype;
    }
private:
    TexType textype;
};

// Constant Texture 
class ConstantTexture : public Texture {
public:
    ConstantTexture(float3 mat_color = make_float3(0.8f)) : mat_color(mat_color) {}
private:
    float3 mat_color;
};

class CheckerTexture : public Texture {
public:
    CheckerTexture(float3 color1, float3 color2) : color1(color1), color2(color2) {}
private:
    float3 color1, color2;
};

class ImageTexture : public Texture {
public:
    ImageTexture(const std::string& filename);
private:
    
}