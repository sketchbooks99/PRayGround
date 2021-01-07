//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include "Material.h"
#include "Shape.h"

struct RadiancePRD
{
    float3 result;
    unsigned int depth;
    unsigned int seed;
};

enum RayType {
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT
};

struct Params
{
    unsigned int subframe_index;
    float4* accum_buffer;
    uchar4* frame_buffer;
    unsigned int width;
    unsigned int height;
    unsigned int samples_per_launch;

    unsigned int max_depth;

    float3 eye;
    float3 U;
    float3 V;
    float3 W;

    OptixTraversableHandle handle;
};

enum HitType
{
    HIT_OUTSIDE_FROM_OUTSIDE = 1u << 0,
    HIT_OUTSIDE_FROM_INSIDE = 1u << 1,
    HIT_INSIDE_FROM_OUTSIDE = 1u << 2,
    HIT_INSIDE_FROM_INSIDE = 1u << 3
};

struct RayGenData 
{
};

struct MissData
{
    float4 bg_color;
};

struct MeshData{
    float4* vertices;
    float4* normals;
    int3* indices;
};

struct DiffuseData {
    float3 mat_color;
    bool is_normal;
};

struct MetalData {
    float3 mat_color;
    float reflection;
};

struct DielectricData {
    float3 mat_color;
    float ior;
};

struct EmissionData {
    float3 color;
};

// member in union must not have any constructor
struct HitGroupData {
    MeshData mesh;
    union {
        DiffuseData diffuse;
        MetalData metal;
        DielectricData dielectric;
        EmissionData emission;
    } shading;
};