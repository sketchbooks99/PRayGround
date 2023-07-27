#pragma once 

#include <optix.h>
#include <prayground/math/vec.h>
#include <prayground/math/matrix.h>
#include <prayground/optix/sbt.h>
#include <prayground/core/camera.h>
#include <prayground/core/interaction.h>
#include <prayground/texture/constant.h>
#include <prayground/texture/checker.h>

using namespace prayground;

using ConstantTexture = ConstantTexture_<Vec3f>;
using CheckerTexture = CheckerTexture_<Vec3f>;

constexpr int32_t NUM_CANDIDATES = 8;

enum class RayType : uint32_t
{
    Radiance = 0,
    Shadow = 1,
    Count = 2
};

struct Triangle {
    Vec3f v0;
    Vec3f v1;
    Vec3f v2;
    Vec3f n;
};

// Should be polygonal light source?
struct LightInfo {
    Triangle triangle;
    Vec3f emission;
};

struct Reservoir {
    int y;          // The output sample (the index of light)
    Vec3f p;        // The position of the light
    int M;          // The number of samples seen so far
    float wsum;     // The sum of weights
    float W;        // Probabilistic weight

    void init()
    {
        y = 0;
        M = 0; 
        wsum = 0.0f;
        W = 0.0f;
    }

    void update(int i, float weight, uint32_t& seed)
    {
        if (weight == 0.0f)
            return;
        wsum += weight;
        M++;
        if (rnd(seed) < (weight / wsum)) {
            y = i;
            W = weight;
        }
    }
};

struct LaunchParams {
    uint32_t width;
    uint32_t height;
    uint32_t samples_per_launch;
    uint32_t max_depth;
    int frame;

    float white;

    Vec4u* result_buffer;
    Vec4f* accum_buffer;
    
    OptixTraversableHandle handle;

    LightInfo* lights;
    int num_lights;
    
    // Streamed reservoir
    // Should prepare the reservoirs for each pixels
    Reservoir* reservoirs;

    Reservoir* prev_reservoirs;
};