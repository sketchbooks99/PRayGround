#pragma once 

#include <prayground/shape/trianglemesh.h>

namespace prayground {

class IcoSphereMesh final : public TriangleMesh {
public:
    IcoSphereMesh(float radius = 1, int level = 2);

    void smooth() override;
    void splitVertices();
private:
    float m_radius;
    int m_level;
    std::vector<int> share_count; // Count the 
};

class UVSphereMesh final : public TriangleMesh {
public:
    UVSphereMesh(float radius = 1, const Vec2ui& resolution = {2,2});
private:
    float m_radius;
    Vec2ui m_resolution;
};

class CylinderMesh final : public TriangleMesh {
public:
    CylinderMesh(float radius = 1, float height = 2, const Vec2ui& resolution = {10, 5});
private:
    float m_radius;
    float m_height;
    Vec2ui m_resolution;
};

class PlaneMesh final : public TriangleMesh {
public: 
    PlaneMesh(float2 size = {10,10}, int2 res = {2,2}, Axis axis=Axis::Y);
private:
    Vec2f m_size;
    Vec2ui m_resolution;
};

} // ::prayground
