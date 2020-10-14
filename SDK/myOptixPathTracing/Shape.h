#pragma once

#include <sutil/vec_math.h>

// Abstract class for readability
struct Shape {
};

// Mesh Data
struct Mesh : public Shape {
    float4* vertices;
    float4* normals;
    int3* indices;
};