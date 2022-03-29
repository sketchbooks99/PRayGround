#pragma once 

#include <prayground/math/matrix.h>
#include <prayground/math/vec.h>

using namespace prayground;

struct AreaEmitterInfo
{
    void* shape_data;
    Matrix4f objToWorld;
    Matrix4f worldToObj;

    uint32_t sample_id;
    uint32_t pdf_id;
};

struct LaunchParams {
    uint32_t width; 
    uint32_t height;
    uint32_t samples_per_launch;
    int frame;

    Vec4f* accum_buffer;
    Vec4u* result_buffer;

    AreaEmitterInfo lights;
};

