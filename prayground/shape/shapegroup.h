#pragma once 

#ifndef __CUDACC__

#include <prayground/core/util.h>
#include <prayground/core/cudabuffer.h>
#include <prayground/core/shape.h>
#include <prayground/shape/trianglemesh.h>
#include <vector>
#include <memory>
#include <concepts>

namespace prayground {

template <class T>
concept DerivedShape = requires(T x)
{
    std::derived_from<T, Shape>; // Shapeの派生クラス
    x.getData();                 // getData()呼び出しが可能
    typename T::Data;            // Data構造体を持っている
};

// 力技感が否めない...
// ShapeTの型が決まった時点でShapeTypeも確定させる(ある種コンパイル時にShapeTからShapeTypeを抽出する？)ことはできる？
template <DerivedShape ShapeT, ShapeType Type>
class ShapeGroup final : public Shape {
public:
    ShapeGroup() {}
    ~ShapeGroup() override {}

    constexpr ShapeType type() override
    {
        return Type;
    }

    OptixBuildInput createBuildInput() override
    {
        OptixBuildInput bi = {};
        std::vector<uint32_t> sbt_indices;
        std::vector<uint32_t> sbt_counter;

        if constexpr (Type == ShapeType::Mesh)
        {
            for (auto& mesh : m_shapes)
            {
                // 重複しないindexの数を数える
                uint32_t sbt_idx = mesh.sbtIndex();
                auto itr = std::find(sbt_counter.begin(), sbt_counter.end(), sbt_idx);
                if (sbt_counter.empty() || itr == sbt_counter.end())
                    sbt_counter.push_back(sbt_idx);
                
                std::vector<uint32_t> mesh_sbt_index(mesh.faces().size(), sbt_idx);
                std::copy(mesh_sbt_index.begin(), mesh_sbt_index.end(), std::back_inserter(sbt_indices));
            }
            std::vector<uint32_t> triangle_input_flags(m_shapes.size(), OPTIX_GEOMETRY_FLAG_NONE);

            CUDABuffer<uint32_t> d_sbt_indices;
            d_sbt_indices.copyToDevice(sbt_indices);

            bi.type = static_cast<OptixBuildInputType>(Type);
            bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            bi.triangleArray.vertexStrideInBytes = sizeof(float3);
            bi.triangleArray.numVertices = m_mesh_input.num_vertices;
            bi.triangleArray.vertexBuffers = m_mesh_input.d_vertices;
            bi.triangleArray.flags = triangle_input_flags.data();
            bi.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            bi.triangleArray.indexStrideInBytes = sizeof(Face);
            bi.triangleArray.numIndexTriplets = m_mesh_input.num_faces;
            bi.triangleArray.numSbtRecords = static_cast<uint32_t>(sbt_counter.size());
            bi.triangleArray.sbtIndexOffsetBuffer = d_sbt_indices.devicePtr();
            bi.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
            bi.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);
        }
        else if constexpr (Type == ShapeType::Custom)
        {
            uint32_t* input_flags = new uint32_t[m_shapes.size()];
            for (int i = 0; auto& custom : m_shapes)
            {
                // 重複しないindexの数を数える
                uint32_t sbt_idx = custom.sbtIndex();
                auto itr = std::find(sbt_counter.begin(), sbt_counter.end(), sbt_idx);
                if (sbt_counter.empty() || itr == sbt_counter.end())
                    sbt_counter.push_back(sbt_idx);

                sbt_indices.push_back(sbt_idx);
                input_flags[i++] = OPTIX_GEOMETRY_FLAG_NONE;
            }

            CUDABuffer<uint32_t> d_sbt_indices;
            d_sbt_indices.copyToDevice(sbt_indices);

            OptixAabb aabb = static_cast<OptixAabb>(this->bound());
            std::vector<OptixAabb> aabbs;
            std::transform(m_shapes.begin(), m_shapes.end(), std::back_inserter(aabbs),
                [](const ShapeT& custom) { return static_cast<OptixAabb>(custom.bound()); });
            CUDABuffer<OptixAabb> d_aabb;
            d_aabb.copyToDevice(aabbs);
            d_aabb_buffer = d_aabb.devicePtr();

            bi.type = static_cast<OptixBuildInputType>(Type);
            bi.customPrimitiveArray.aabbBuffers = &d_aabb_buffer;
            bi.customPrimitiveArray.numPrimitives = static_cast<uint32_t>(m_shapes.size());
            bi.customPrimitiveArray.flags = input_flags;
            bi.customPrimitiveArray.numSbtRecords = static_cast<uint32_t>(sbt_counter.size());
            bi.customPrimitiveArray.sbtIndexOffsetBuffer = d_sbt_indices.devicePtr();
            bi.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
            bi.customPrimitiveArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);
        }
        else if constexpr (Type == ShapeType::Curves)
        {
            UNIMPLEMENTED();
        }

        return bi;
    }

    void copyToDevice() override
    {
        if constexpr (Type == ShapeType::Mesh)
        {
            std::vector<float3> vertices;
            std::vector<float3> normals;
            std::vector<float2> texcoords;
            std::vector<Face> faces;
            int32_t num_verts = 0;
            int32_t num_normals = 0;
            int32_t num_texcoords = 0;
            for (auto& mesh : m_shapes)
            {
                std::copy(mesh.vertices().begin(), mesh.vertices().end(), std::back_inserter(vertices));
                std::copy(mesh.normals().begin(), mesh.normals().end(), std::back_inserter(normals));
                std::copy(mesh.texcoords().begin(), mesh.texcoords().end(), std::back_inserter(texcoords));
                std::transform(mesh.faces().begin(), mesh.faces().end(), std::back_inserter(faces),
                    [&](Face face) { return Face{
                        .vertex_id = face.vertex_id + make_int3(num_verts),
                        .normal_id = face.normal_id + make_int3(num_normals),
                        .texcoord_id = face.texcoord_id + make_int3(num_texcoords),
                    };  });
                
                num_verts += (uint32_t)mesh.vertices().size();
                num_normals += (uint32_t)mesh.normals().size();
                num_texcoords += (uint32_t)mesh.texcoords().size();
            }
            
            // GPU側にMeshのデータをコピー
            // tmp_mesh.copyToDevice()としても、tmp_meshの保持するd_dataの寿命が切れる可能性があるため、それは避ける
            TriangleMesh tmp_mesh{vertices, faces, normals, texcoords};
            MeshData data = tmp_mesh.getData();
            if (!d_data) 
                CUDA_CHECK(cudaMalloc(&d_data, sizeof(MeshData)));
            CUDA_CHECK(cudaMemcpy(
                d_data,
                &data, sizeof(MeshData),
                cudaMemcpyHostToDevice
            ));

            m_mesh_input = 
            {
                .d_vertices = tmp_mesh.deviceVertices(),
                .d_faces = tmp_mesh.deviceFaces(),
                .num_vertices = static_cast<uint32_t>(tmp_mesh.vertices().size()),
                .num_faces = static_cast<uint32_t>(tmp_mesh.faces().size())
            };
        }
        else if constexpr (Type == ShapeType::Custom)
        {
            std::vector<typename ShapeT::Data> device_data;
            for (auto& custom : m_shapes)
                device_data.push_back(custom.getData());

            if (!d_data)
                CUDA_CHECK(cudaMalloc(&d_data, sizeof(typename ShapeT::Data) * device_data.size()));
            CUDA_CHECK(cudaMemcpy(
                d_data, 
                device_data.data(), sizeof(typename ShapeT::Data) * device_data.size(),
                cudaMemcpyHostToDevice
            ));
        }
        else if constexpr (Type == ShapeType::Curves)
        {
            UNIMPLEMENTED();
        }
    }

    void free() override
    {
        for (auto& shape : m_shapes)
            shape.free();
        cuda_free(d_aabb_buffer);
    }

    AABB bound() const override
    {
        AABB aabb{};
        if constexpr (Type == ShapeType::Custom)
        {
            for (auto& shape : m_shapes)
                aabb = AABB::merge(aabb, shape.bound());
        }
        return aabb;
    }

    void addShape(const ShapeT& shape)
    {
        m_shapes.push_back(shape);
    }

    uint32_t numShapes() const
    {
        return static_cast<uint32_t>(m_shapes.size());
    }
private:
    struct MeshBuildInputData
    {
        CUdeviceptr d_vertices;
        CUdeviceptr d_faces;
        uint32_t num_vertices;
        uint32_t num_faces;
    };

    MeshBuildInputData m_mesh_input; // Mesh用
    CUdeviceptr d_aabb_buffer;       // Custom primitive用

    std::vector<ShapeT> m_shapes;
};

} // ::prayground

#endif // __CUDACC__