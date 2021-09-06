#pragma once 

#include <prayground/shape/trianglemesh.h>

namespace prayground {

class PrimitiveMesh {
public:
    TriangleMesh getMesh();
private:
    std::vector<float3> m_vertices;
    std::vector<Face> m_faces;
    std::vector<float3> m_normals;
    std::vector<float2> m_texcoords;
};

class IcoSphereMesh final : public PrimitiveMesh {
public:
    IcoSphereMesh(float radius=1, int subdivisions=1);

    void smooth();
    void splitVertices();
private:
    float m_radius; 
    int m_subdivisions;
};

class UVSphereMesh final : public PrimitiveMesh {
public:
    UVSphereMesh(float radius=1, int2 res = {2,2});

    void smooth();
private:
    float m_radius;
    int2 m_resolution;
};

class CylinderMesh final : public PrimitiveMesh {
public:
    CylinderMesh(float radius = 1, float height = 2);
private:
    float m_radius; 
    float m_height;
};

class PlaneMesh final : public PrimitiveMesh {
public: 
    PlaneMesh(float2 size = {10,10}, int2 res = {2,2});
private:
    float2 m_size;
    int2 m_resolution;
};

} // ::prayground
