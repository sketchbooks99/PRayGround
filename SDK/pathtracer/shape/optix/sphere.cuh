#pragma once 

#include <optix.h>
#include <sutil/vec_math.h>
#include <core/transform.h>
#include <core/material.h>

struct SphereHitGroupData {
    float3 center;
    float radius;
    sutil::Transform transform;
    MaterialPtr matptr;
};

CALLABLE_FUNC void IS_FUNC(sphere) {
    
}