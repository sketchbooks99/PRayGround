#pragma once

#include <optix.h>
#include <prayground/math/vec.h>
#include <prayground/texture/constant.h>
#include <prayground/texture/checker.h>
#include <prayground/physics/tree.h>
#include "envmap.cuh"
#include "textures.h"

using namespace prayground;

using ConstantTexture = ConstantTexture_<Vec4f>;
using CheckerTexture = CheckerTexture_<Vec4f>;
using ProceduralWoodenTexture = ProceduralWoodenTexture_<Vec4f>;
using StarNightTexture = StarNightTexture_<Vec4f>;
using LeafTexture = LeafTexture_<Vec4f>;

struct AreaEmitterInfo
{
    void* shape_data;
    SurfaceInfo* surface_info;

    Matrix4f objToWorld;
    Matrix4f worldToObj;

    uint32_t sample_id;
    uint32_t pdf_id;
};

struct LightInteraction
{
    // A surface point on the light source in world coordinates
    Vec3f p;
    // Surface normal on the light source in world coordinates
    Vec3f n;
    // Texture coordinates on light source
    Vec2f uv;
    // Area of light source
    float area;
    // PDF of light source
    float pdf;
};

// Texture baking parameters
struct BakeTextureParams {
    uint32_t width;
    uint32_t height;
    uint32_t texture_id;
    void* texture_data;
    Vec3f* output_buffer;  // RGB float buffer
};

struct LaunchParams {
    uint32_t width;
    uint32_t height;
    uint32_t samples_per_launch;
    int32_t frame;
    uint32_t max_depth;
    
    // Elapsed time from the start of rendering (in seconds)
    float elapsed_time;

    Vec4u* result_buffer;
    Vec4f* accum_buffer;
    OptixTraversableHandle handle;

    AreaEmitterInfo* lights;
    uint32_t n_lights;

    // Environment map importance sampling data
    EnvmapSamplingData* envmap_sampling_data;
    uint32_t envmap_sample_id;
    uint32_t envmap_pdf_id;
    uint32_t envmap_texture_id;
    void* envmap_texture_data;

    float white;
};

struct ProceduralTreeData {
    float gravity;           // Downward force (bending down)
    float coverage;          // Horizontal spreading force
    float vertically;        // Upward vertical force (resistance to gravity)
    float twist;             // Twist amount for spiral effect
    bool trunk_mode;         // If true, main trunk continues straight with side branches
    float scale;
    int depth;
    float radius;
    
    // Leaf parameters
    bool has_leaves;         // Whether to generate leaves
    float leaf_density;      // Density of leaves (0.0 - 1.0)
    int leaf_start_gen;      // Generation to start adding leaves (e.g., 2 = only on thin branches)
    float leaf_size;         // Size of each leaf
};