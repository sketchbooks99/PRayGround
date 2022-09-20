#pragma once

#ifndef __CUDACC__
#include <prayground/core/shape.h>
#include <prayground/core/cudabuffer.h>
#endif

namespace prayground {

    class Curves final : public Shape {
    public:
        struct Data {
            Vec3f* vertices;
            int32_t* indices;
            Vec3f* normals;
            float* widths;
        };

        enum class Type : uint32_t {
            Linear = OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR,
            QuadlicBspline = OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE,
            CubicBspline = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE
        };

#ifndef __CUDACC__

        Curves();
        Curves(
            const std::vector<Vec3f>& vertices, 
            const std::vector<int32_t>& indices, 
            const std::vector<Vec3f>& normals, 
            const std::vector<float>& widths );

        constexpr ShapeType type() override;

        void copyToDevice() override;
        void free() override;

        OptixBuildInput createBuildInput() override;

        uint32_t numPrimitives() const override;

        AABB bound() const override;

        Data getData();

        void addVertices(const std::vector<Vec3f>& vertices);
        void addIndices(const std::vector<int32_t>& indices);
        void addNormals(const std::vector<Vec3f>& normals);
        void addWidths(const std::vector<float>& widths);

        void addVertex(const Vec3f& v);
        void addIndex(int32_t i);
        void addNormal(const Vec3f& n);
        void addWidth(float w);

        void load(const std::filesystem::path& filename);

        const std::vector<Vec3f>& vertices() const { return m_vertices; }
        const std::vector<int32_t>& indices() const { return m_indices; }
        const std::vector<Vec3f>& normals() const { return m_normals; }
        const std::vector<float>& widths() const { return m_widths; }

        CUdeviceptr deviceVertices() const { return d_vertices; }
        CUdeviceptr deviceIndices() const { return d_indices; }
        CUdeviceptr deviceNormals() const { return d_normals; }
        CUdeviceptr deviceWidths() const { return d_widths; }

    private:
        std::vector<Vec3f> m_vertices;
        std::vector<int32_t> m_indices;
        std::vector<Vec3f> m_normals;
        std::vector<float> m_widths;

        CUdeviceptr d_vertices { 0 };
        CUdeviceptr d_indices { 0 };
        CUdeviceptr d_normals { 0 };
        CUdeviceptr d_widths { 0 };
#endif
    };

} // namespace prayground