#pragma once

#include <optix.h>
#include <core/util.h>
#include <core/material.h>

namespace pt {

struct RayGenData {};

struct MissData {
    float4 bg_color;
};

struct HitGroupData {
    /// Pointer that stores geometries data (e.g. \c pt::MeshData )
    CUdeviceptr shapedata;
    MaterialPtr matptr;
};

template <typename T>
struct Record 
{
    __align__ (OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using RayGenRecord = Record<RayGenData>;
using MissRecord = Record<MissData>;
using HitGroupRecord = Record<HitGroupData>;

}