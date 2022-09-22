#include "curves.h"
#include <prayground/core/cudabuffer.h>
#include <prayground/core/file_util.h>
#include <prayground/math/util.h>

namespace prayground {

    namespace fs = std::filesystem;

    Curves::Curves() {}

    Curves::Curves(
        const std::vector<Vec3f>& vertices, 
        const std::vector<int32_t>& indices, 
        const std::vector<Vec3f>& normals, 
        const std::vector<float>& widths) 
        : m_vertices(vertices)
        , m_indices(indices)
        , m_normals(normals)
        , m_widths(widths)
    {

    }

    constexpr ShapeType Curves::type() 
    {
        return ShapeType::Curves;
    }

    void Curves::copyToDevice()
    {
        Data data = this->getData();

        if (!d_data)
            CUDA_CHECK(cudaMalloc(&d_data, sizeof(Data)));
        CUDA_CHECK(cudaMemcpy(
            d_data,
            &data, sizeof(Data),
            cudaMemcpyHostToDevice
        ));
    }

    void Curves::free()
    {
        Shape::free();
        cuda_frees(d_vertices, d_normals, d_indices, d_widths);
    }

    OptixBuildInput Curves::createBuildInput()
    {
        OptixBuildInput bi = {};

        bi.type = static_cast<OptixBuildInputType>(this->type());
        bi.curveArray.vertexBuffers = &d_vertices;
        bi.curveArray.vertexStrideInBytes = sizeof(Vec3f);
        bi.curveArray.numVertices = static_cast<uint32_t>(m_vertices.size());
        bi.curveArray.widthBuffers = &d_widths;
        bi.curveArray.widthStrideInBytes = sizeof(float);
        bi.curveArray.normalBuffers = &d_normals;
        bi.curveArray.normalStrideInBytes = sizeof(Vec3f);
        bi.curveArray.indexBuffer = d_indices;
        bi.curveArray.indexStrideInBytes = sizeof(int32_t);
        bi.curveArray.flag = OPTIX_GEOMETRY_FLAG_NONE;
        bi.curveArray.primitiveIndexOffset = 0;
        return bi;
    }

    uint32_t Curves::numPrimitives() const
    {
        return m_indices.size();
    }

    AABB Curves::bound() const
    {
        return AABB{};
    }

    Curves::Data Curves::getData()
    {
        CUDABuffer<Vec3f> d_vertices_buf;
        CUDABuffer<int> d_indices_buf;
        CUDABuffer<Vec3f> d_normals_buf;
        CUDABuffer<float> d_widths_buf;

        d_vertices_buf.copyToDevice(m_vertices);
        d_indices_buf.copyToDevice(m_indices);
        d_normals_buf.copyToDevice(m_normals);
        d_widths_buf.copyToDevice(m_widths);

        d_vertices = d_vertices_buf.devicePtr();
        d_indices = d_indices_buf.devicePtr();
        d_normals = d_normals_buf.devicePtr();
        d_widths = d_widths_buf.devicePtr();

        // Device side pointer of curve mesh data
        Data data = {
            .vertices = d_vertices_buf.deviceData(),
            .indices = d_indices_buf.deviceData(),
            .normals = d_normals_buf.deviceData(),
            .widths = d_widths_buf.deviceData(),
        };

        return data;
    }

    void Curves::addVertices(const std::vector<Vec3f>& vertices)
    {
        std::copy(vertices.begin(), vertices.end(), std::back_inserter(m_vertices));
    }

    void Curves::addIndices(const std::vector<int32_t>& indices)
    {
        std::copy(indices.begin(), indices.end(), std::back_inserter(m_indices));
    }

    void Curves::addNormals(const std::vector<Vec3f>& normals)
    {
        std::copy(normals.begin(), normals.end(), std::back_inserter(m_normals));
    }

    void Curves::addWidths(const std::vector<float>& widths)
    {
        std::copy(widths.begin(), widths.end(), std::back_inserter(m_widths));
    }

    void Curves::addVertex(const Vec3f& v)
    {
        m_vertices.emplace_back(v);
    }

    void Curves::addIndex(int32_t i)
    {
        m_indices.emplace_back(i);
    }

    void Curves::addNormal(const Vec3f& n)
    {
        m_normals.emplace_back(n);
    }

    void Curves::addWidth(float w)
    {
        m_widths.emplace_back(w);
    }

    void Curves::load(const std::filesystem::path& filename)
    {
    }

} // namespace prayground