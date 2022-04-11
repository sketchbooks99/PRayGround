#pragma once 

#include <prayground/math/matrix.h>
#include <prayground/math/vec.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace prayground;

using ConstantTexture = ConstantTexture_<Vec3f>;
using CheckerTexture = CheckerTexture_<Vec3f>;

struct AreaEmitterInfo
{
    void* shape_data;
    SurfaceInfo surface_info;
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

struct PathVertex {
    // Hit position
    Vec3f       p;
    // Path throughput
    Vec3f       throughput;
    // Path length between source and vertex
    uint32_t    path_length;

    // Surface infomation on a vertex
    SurfaceInfo surface_info;
    bool from_light;

    // MIS quantities
    float dVCM;
    float dVM; 
    float dVC;
};

struct VCM {
    float base_radius;
    float radius;
    float radius_alpha;
    // Iterator
    int iteration;
    float vm_normalization;

    // MIS factor
    float mis_vm_weight_factor;
    float mis_vc_weight_factor;

    // Light vertices
    thrust::device_vector<PathVertex> light_vertices;
    thrust::host_vector<int> path_ends;
};

struct LaunchParams {
    uint32_t width; 
    uint32_t height;
    uint32_t samples_per_launch;
    uint32_t max_depth;
    int frame;

    Vec4f* accum_buffer;
    Vec4u* result_buffer;

    AreaEmitterInfo* lights;
    uint32_t num_lights;

    OptixTraversableHandle handle;

    VCM vcm;
};