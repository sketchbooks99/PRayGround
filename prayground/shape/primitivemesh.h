#pragma once 

#include <prayground/shape/trianglemesh.h>

namespace prayground {

#ifndef __CUDACC__

    class IcoSphereMesh final : public TriangleMesh {
    public:
        IcoSphereMesh(float radius = 1, int level = 2);

        void subdivide(const float level);
        void smooth() override;
        void splitVertices();
    private:
        float m_radius;
        int m_level;
        std::vector<int> share_count; // Count the number of sharing faces
    };

    class UVSphereMesh final : public TriangleMesh {
    public:
        UVSphereMesh(float radius = 1, const Vec2ui& resolution = {2,2});

        float radius() const;
        void setRadius(const float radius);

        const Vec2ui& resolution() const;
        void setResolution(const Vec2ui& resolution);
    private:
        float m_radius;
        Vec2ui m_resolution;
    };

    class CylinderMesh final : public TriangleMesh {
    public:
        CylinderMesh(float radius = 1, float height = 2, const Vec2ui& resolution = {10, 5});

        float radius() const;
        void setRadius(const float radius);

        float height() const;
        void setHeight(const float height);

        const Vec2ui& resolution() const;
        void setResolution(const Vec2ui& resolution);
    private:
        float m_radius;
        float m_height;
        Vec2ui m_resolution;
    };

    class PlaneMesh final : public TriangleMesh {
    public: 
        PlaneMesh(const Vec2f& size = {10,10}, const Vec2ui& resolution = {2,2}, Axis axis=Axis::Y);

        void init();

        const Vec2f& size() const;
        void setSize(const Vec2f& size);

        const Vec2ui& resolution() const;
        void setResolution(const Vec2ui& resolution);
    private:
        Vec2f m_size;
        Vec2ui m_resolution;
        Axis m_axis;
    };

#endif

} // namespace prayground
