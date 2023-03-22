#pragma once 

#include <prayground/shape/trianglemesh.h>
#ifndef __CUDACC__
#include <map>
#endif

namespace prayground {

#ifndef __CUDACC__

    class IcoSphereMesh final : public TriangleMesh {
    public:
        IcoSphereMesh(float radius = 1, int level = 2);

        void subdivide(const float level);
        void splitVertices();
    private:
        /* Return index if the same vertex position has already existed.
        *  If there is no same vertex, it will return -1. */
        int32_t findOrAddVertex(const Vec3f& v, const Vec2f& texcoord);

        float m_radius;
        int m_level;
        /* Map to search an index correnponds to a texcoord */
        std::map<std::pair<float, float>, int32_t> m_indices;
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
