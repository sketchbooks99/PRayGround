#ifndef TRIANGLEMESH_H
#define TRIANGLEMESH_H

#include <fstream>
#include <vector>
#include <sutil/vec_math.h>
#include <iostream>
#include <assert.h>

// TODO: I have to replace all float4 to Vertex ... Fuuuuuuuuuuck!!!

struct Vertex {
    Vertex() : x(0.0f), y(0.0f), z(0.0f), pad(0.0f) {}
    Vertex(float x, float y, float z, float pad) : x(x), y(y), z(z), pad(pad) {}
    Vertex(float t) : x(t), y(t), z(t), pad(0.0f) {};

    Vertex operator-() const { return Vertex(-x, -y, -z, 0.0f); }
    float operator[](int idx) const {
        assert(idx <= 3);
        if      (idx == 0) return x;
        else if (idx == 1) return y;
        else if (idx == 2) return z;
        else if (idx == 3) return pad;
    }
    float& operator[](int idx) {
        assert(idx <= 3);
        if      (idx == 0) return x;
        else if (idx == 1) return y;
        else if (idx == 2) return z;
        else if (idx == 3) return pad;
    }

    Vertex& operator+=(const Vertex &v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    Vertex& operator*=(const float t)
    {
        x *= t;
        y *= t;
        z *= t;
        return *this;
    }

    Vertex& operator/=(const float t)
    {
        assert(t != 0.0f);
        return *this *= 1 / t;
    }

    float length() const { return sqrt(length_squared()); }
    float length_squared() const { return x * x + y * y + z * z; }
    
    float x, y, z, pad;
};

typedef Vertex Normal;

inline std::ostream& operator<<(std::ostream& out, const Vertex& v)
{
    return out << v.x << ' ' << v.y << ' ' << v.z;
}

inline Vertex operator+(const Vertex& v1, const Vertex& v2)
{
    return Vertex(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, 0.0f);
}

inline Vertex operator+(const Vertex& v, const float3 f)
{
    return Vertex(v.x + f.x, v.y + f.y, v.z + f.z, 0.0f);
}

inline Vertex operator-(const Vertex& v1, const Vertex& v2) 
{
    return v1 + (-v2);
}

inline Vertex operator*(const Vertex& v1, const Vertex& v2)
{
    return Vertex(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z, 0.0f);
}

inline Vertex operator*(const Vertex& v, const float t)
{
    return Vertex(v.x * t, v.y * t, v.z * t, 0.0f);
}

inline Vertex operator/(const Vertex& v, const float t)
{
    assert(t != 0);
    return v * (1 / t);
}

inline float dot(const Vertex& v1, const Vertex& v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

inline Vertex cross(const Vertex& v1, const Vertex& v2)
{
    return Vertex(
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x,
        0.0f
    );
}

inline Vertex normalize(Vertex v)
{
    return v / v.length();
}
 
struct TriangleMesh {
    TriangleMesh() {}
    TriangleMesh(const std::string& filename, float3 position, float size, float3 axis);
    TriangleMesh(std::vector<Vertex> vertices, std::vector<int3> faces, std::vector<Normal> normals);
    std::vector<Vertex> vertices;
    std::vector<Normal> normals;
    std::vector<int3> indices;
};

#endif