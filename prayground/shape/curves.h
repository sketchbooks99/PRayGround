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

#ifndef __CUDACC__

        Curves();
        Curves(
            const std::vector<Vec3f>& vertices, 
            const std::vector<int32_t>& indices, 
            const std::vector<Vec3f>& normals, 
            const std::vector<float>& widths, 
            const std::vector<uint32_t>& sbt_indices = std::vector<uint32_t>() );

        constexpr ShapeType type() override;

        void copyToDevice() override;
        void free() override;

        OptixBuildInput createBuildInput() override;

        uint32_t numPrimitives() const override;

        AABB bound() const override;

        Data getData() const;

        void addVertices(const std::vector<Vec3f>& vertices);
        void addIndices(const std::vector<int32_t>& indices);
        void addNormals(const std::vector<Vec3f>& normals);
        void addWidths(const std::vector<float>& widths);

        void addVertex(const Vec3f& v);
        void addIndex(int32_t i);
        void addNormal(const Vec3f& n);
        void addWidth(float w);

        void load(const std::filesystem::path& filename);

        void addSbtIndices(const std::vector<uint32_t>& sbt_indices);
        void offsetSbtIndex(uint32_t sbt_base);
        uint32_t numMaterials() const;

        const std::vector<Vec3f>& vertices() const { return m_vertices; }
        const std::vector<int32_t>& indices() const { return m_indices; }
        const std::vector<Vec3f>& normals() const { return m_normals; }
        const std::vector<Vec2f>& widths() const { return m_widths; }
        const std::vector<uint32_t>& sbtIndices() const { return m_sbt_indices; }

        CUdeviceptr deviceVertices() const { return d_vertices; }
        CUdeviceptr deviceIndices() const { return d_indices; }
        CUdeviceptr deviceNormals() const { return d_normals; }
        CUdeviceptr deviceWidths() const { return d_widths; }
        CUdeviceptr deviceSbtIndices() const { return d_sbt_indices; }

    private:
        std::vector<Vec3f> m_vertices;
        std::vector<int32_t> m_indices;
        std::vector<Vec3f> m_normals;
        std::vector<float> m_widths;
        std::vector<uint32_t> m_sbt_indices;

        CUdeviceptr d_vertices { 0 };
        CUdeviceptr d_normals { 0 };
        CUdeviceptr d_normals { 0 };
        CUdeviceptr d_widths { 0 };
        CUdeviceptr d_sbt_indices { 0 };
#endif
    };

} // namespace prayground