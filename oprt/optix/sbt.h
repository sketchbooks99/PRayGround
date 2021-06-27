#pragma once

#include <optix.h>
#include "../core/material.h"
#include "../core/util.h"
#include "../optix/util.h"

namespace oprt {

struct EmptyData {};

struct MissData {
    void* env_data;
};

struct HitGroupData {
    /// Pointer that stores geometries data (e.g. \c oprt::MeshData )
    void* shape_data;
    void* surface_data;

    // Index of direct callables function to sample bsdf properties.
    unsigned int surface_func_base_id;
    SurfaceType surface_type;
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