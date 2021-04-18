#pragma once

#include <optix.h>
#include "../core/material.h"
#include "../core/util.h"

namespace oprt {

struct EmptyData {};

struct MissData {
    float4 bg_color;
};

struct HitGroupData {
    /// Pointer that stores geometries data (e.g. \c oprt::MeshData )
    void* shapedata;
    void* matdata;

    // Index of direct callables function to sample bsdf properties.
    unsigned int sample_func_idx;
};

#ifndef __CUDACC__
template <typename T>
struct Record 
{
    __align__ (OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using RayGenRecord = Record<EmptyData>;
using MissRecord = Record<MissData>;
using HitGroupRecord = Record<HitGroupData>;
using CallableRecord = Record<EmptyData>;

#endif

}