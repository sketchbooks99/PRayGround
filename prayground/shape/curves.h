#pragma once

#ifndef __CUDACC__
#include <prayground/core/shape.h>
#include <prayground/core/cudabuffer.h>
#include <filesystem>
#endif

namespace prayground {

    class Curves final : public Shape {
    public:
        struct Data {
            Vec3f* vertices;
            int32_t* indices;
            float* widths;
            Vec3f* normals;
        };

        enum class Type : uint32_t {
            QuadraticBSpline = OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE,
            CubicBSpline = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE,
            Linear = OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR,
#if OPTIX_VERSION >= 70400
            CatmullRom = OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM
#endif
        };

#ifndef __CUDACC__

        Curves(Curves::Type curve_type);
        Curves(
            Curves::Type curve_type,
            const std::vector<Vec3f>& vertices, 
            const std::vector<int32_t>& indices,
            const std::vector<float>& widths,
            const std::vector<Vec3f>& normals);

        constexpr ShapeType type() override;

        void copyToDevice() override;
        void free() override;

        OptixBuildInput createBuildInput() override;

        uint32_t numPrimitives() const override;

        AABB bound() const override;

        Data getData();

        void addVertices(const std::vector<Vec3f>& vertices);
        void addIndices(const std::vector<int32_t>& indices);
        void addWidths(const std::vector<float>& widths);
        void addNormals(const std::vector<Vec3f>& normals);

        void addVertex(const Vec3f& v);
        void addIndex(int32_t i);
        void addWidth(float w);
        void addNormal(const Vec3f& n);

        void load(const std::filesystem::path& filename);

        Curves::Type curveType() const { return m_curve_type; }

        const std::vector<Vec3f>& vertices() const { return m_vertices; }
        const std::vector<int32_t>& indices() const { return m_indices; }
        const std::vector<float>& widths() const { return m_widths; }
        const std::vector<Vec3f>& normals() const { return m_normals; }

        CUdeviceptr deviceVertices() const { return d_vertices; }
        CUdeviceptr deviceIndices() const { return d_indices; }
        CUdeviceptr deviceWidths() const { return d_widths; }
        CUdeviceptr deviceNormals() const { return d_normals; }

        static uint32_t getNumVertexPerSegment(Curves::Type curves_type);

    private:
        Curves::Type m_curve_type;

        std::vector<Vec3f> m_vertices;
        std::vector<int32_t> m_indices;
        std::vector<float> m_widths;
        std::vector<Vec3f> m_normals;

        CUdeviceptr d_vertices { 0 };
        CUdeviceptr d_indices { 0 };
        CUdeviceptr d_widths { 0 };
        CUdeviceptr d_normals{ 0 };

#endif
    };

} // namespace prayground