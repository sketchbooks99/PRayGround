#pragma once

#include <optix.h>
#include <core/util.h>

namespace pt {

struct MissData {
    float4 bg_color;
};

struct ShapeData {
    void* data;
};

template <typename T>
struct Record 
{
    __align__ (OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct EmptyRecord
{
    __align__ (OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

using RayGenRecord = EmptyRecord;
using MissRecord = Record<MissData>;
using HitGroupRecord = Record<ShapeData>;

}