#pragma once

#include <optix.h>
#include <include/core/material.h>
#include <include/core/util.h>

namespace pt {

struct RayGenData {};

struct MissData {
    float4 bg_color;
};

struct HitGroupData {
    /// Pointer that stores geometries data (e.g. \c pt::MeshData )
    void* shapedata;
    Material* matptr;
};

struct HitGroupData2 {
    float3* vertices; 
    float3* normals; 
    int3* indices;
    float3 emission; 
    float3 albedo;
};

#ifndef __CUDACC__
template <typename T>
struct Record 
{
    __align__ (OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using RayGenRecord = Record<RayGenData>;
using MissRecord = Record<MissData>;
using HitGroupRecord = Record<HitGroupData>;
using HitGroupRecord2 = Record<HitGroupData2>;

#endif

}