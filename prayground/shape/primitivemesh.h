#pragma once 

#include <prayground/shape/trianglemesh.h>

namespace prayground {

class IcoSphereMesh final : public TriangleMesh {
public:
    IcoSphereMesh(float radius=1, int level=2);

    void smooth() override;
    void splitVertices();
private:
    float m_radius; 
    int m_level;
    std::vector<int> share_count; // 各頂点の共有数を管理する
};

class UVSphereMesh final : public TriangleMesh {
public:
    UVSphereMesh(float radius=1, int2 res = {2,2});
private:
    float m_radius;
    int2 m_resolution;
};

class CylinderMesh final : public TriangleMesh {
public:
    CylinderMesh(float radius = 1, float height = 2);
private:
    float m_radius; 
    float m_height;
};

class PlaneMesh final : public TriangleMesh {
public: 
    PlaneMesh(float2 size = {10,10}, int2 res = {2,2}, Axis axis=Axis::Y);
private:
    float2 m_size;
    int2 m_resolution;
};

} // ::prayground
