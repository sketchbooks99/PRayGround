#include "trianglemesh.h"
#include <prayground/core/cudabuffer.h>
#include <prayground/core/load3d.h>
#include <prayground/core/file_util.h>
#include <prayground/math/util.h>

namespace prayground {

namespace fs = std::filesystem;

namespace {

float2 getSphereUV(const float3& p)
{
    float phi = atan2(p.z, p.x);
    float theta = asin(p.y);
    float u = 1.0f - (phi + constants::pi) / (constants::two_pi);
    float v = 1.0f - (theta + constants::pi / 2.0f) / constants::pi;
    return make_float2(u, v);
}

} // ::nonamed namespace

/** @note At present, only .obj file format is supported. */
// ------------------------------------------------------------------
TriangleMesh::TriangleMesh(const fs::path& filename)
{
    load(filename);
}

TriangleMesh::TriangleMesh(
    std::vector<float3> vertices, 
    std::vector<Face> faces, 
    std::vector<float3> normals, 
    std::vector<float2> texcoords) 
    : m_vertices(vertices), 
      m_faces(faces), 
      m_normals(normals),
      m_texcoords(texcoords)
{
    
}

// ------------------------------------------------------------------
ShapeType TriangleMesh::type() const
{
    return ShapeType::Mesh;
}

// ------------------------------------------------------------------
void TriangleMesh::copyToDevice() {

    CUDABuffer<float3> d_vertices_buf;
    CUDABuffer<Face> d_faces_buf;
    CUDABuffer<float3> d_normals_buf;
    CUDABuffer<float2> d_texcoords_buf;
    d_vertices_buf.copyToDevice(m_vertices);
    d_faces_buf.copyToDevice(m_faces);
    d_normals_buf.copyToDevice(m_normals);
    d_texcoords_buf.copyToDevice(m_texcoords);

    // device side pointer of mesh data
    MeshData data = {
        .vertices = d_vertices_buf.deviceData(),
        .faces = d_faces_buf.deviceData(), 
        .normals = d_normals_buf.deviceData(),
        .texcoords = d_texcoords_buf.deviceData()
    };

    if (!d_data) 
        CUDA_CHECK(cudaMalloc(&d_data, sizeof(MeshData)));
    CUDA_CHECK(cudaMemcpy(
        d_data,
        &data, sizeof(MeshData),
        cudaMemcpyHostToDevice
    ));

    d_vertices = d_vertices_buf.devicePtr();
    d_faces = d_faces_buf.devicePtr();
    d_normals = d_normals_buf.devicePtr();
    d_texcoords = d_texcoords_buf.devicePtr();
}

// ------------------------------------------------------------------
OptixBuildInput TriangleMesh::createBuildInput() 
{
    OptixBuildInput bi = {};
    CUDABuffer<uint32_t> d_sbt_faces;
    std::vector<uint32_t> sbt_faces(m_faces.size(), m_sbt_index);
    d_sbt_faces.copyToDevice(sbt_faces);

    unsigned int* triangle_input_flags = new unsigned int[1];
    triangle_input_flags[0] = OPTIX_GEOMETRY_FLAG_NONE;
    
    bi.type = static_cast<OptixBuildInputType>(this->type());
    bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    bi.triangleArray.vertexStrideInBytes = sizeof(float3);
    bi.triangleArray.numVertices = static_cast<uint32_t>(m_vertices.size());
    bi.triangleArray.vertexBuffers = &d_vertices;
    bi.triangleArray.flags = triangle_input_flags;
    bi.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    bi.triangleArray.indexStrideInBytes = sizeof(Face);
    bi.triangleArray.numIndexTriplets = static_cast<uint32_t>(m_faces.size());
    bi.triangleArray.indexBuffer = d_faces;
    bi.triangleArray.numSbtRecords = 1;
    bi.triangleArray.sbtIndexOffsetBuffer = d_sbt_faces.devicePtr();
    bi.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    bi.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);
    return bi;
}

// ------------------------------------------------------------------
void TriangleMesh::free()
{
    Shape::free();
    cuda_frees(d_vertices, d_normals, d_faces, d_texcoords);
}

AABB TriangleMesh::bound() const 
{
    return AABB{};
}

// ------------------------------------------------------------------
void TriangleMesh::addVertex(const float3& v)
{
    TODO_MESSAGE();
}

void TriangleMesh::addFace(const Face& face)
{
    TODO_MESSAGE();
}

void TriangleMesh::addNormal(const float3& n)
{
    TODO_MESSAGE();
}

void TriangleMesh::addTexcoord(const float2& texcoord)
{
    TODO_MESSAGE();
}

// ------------------------------------------------------------------
void TriangleMesh::load(const fs::path& filename)
{
    if (filename.string().substr(filename.string().length() - 4) == ".obj") {
        std::optional<fs::path> filepath = findDataPath(filename);
        Assert(filepath, "The OBJ file '" + filename.string() + "' is not found.");

        Message(MSG_NORMAL, "Loading OBJ file '" + filepath.value().string() + "' ...");
        loadObj(filepath.value(), m_vertices, m_normals, m_faces, m_texcoords);
    }
    else if (filename.string().substr(filename.string().length() - 4) == ".ply") {
        std::optional<fs::path> filepath = findDataPath(filename);
        Assert(filepath, "The OBJ file '" + filename.string() + "' is not found.");
            
        Message(MSG_NORMAL, "Loading PLY file '" + filepath.value().string() + "' ...");
        loadPly(filepath.value(), m_vertices, m_normals, m_faces, m_texcoords);
    }

    // Calculate normals if they are empty.
    if (m_normals.empty())
    {
        m_normals.resize(m_faces.size());
        for(size_t i=0; i<m_faces.size(); i++)
        {
            m_faces[i].normal_id = make_int3(i);

            auto p0 = m_vertices[m_faces[i].vertex_id.x];
            auto p1 = m_vertices[m_faces[i].vertex_id.y];
            auto p2 = m_vertices[m_faces[i].vertex_id.z];
            auto N = normalize(cross(p2 - p0, p1 - p0));

            m_normals[i] = N;
            m_normals[i] = N;
            m_normals[i] = N;
        }
    }

    if (m_texcoords.empty()) {
        m_texcoords.resize(m_vertices.size());
    }
}

void TriangleMesh::smooth()
{
    m_normals.clear();
    m_normals = std::vector<float3>(m_vertices.size(), make_float3(0.0f));
    auto counts = std::vector<int>(m_vertices.size(), 0);
    for (size_t i = 0; i < m_faces.size(); i++)
    {
        m_faces[i].normal_id = m_faces[i].vertex_id;

        auto p0 = m_vertices[m_faces[i].vertex_id.x];
        auto p1 = m_vertices[m_faces[i].vertex_id.y];
        auto p2 = m_vertices[m_faces[i].vertex_id.z];
        auto N = normalize(cross(p2 - p0, p1 - p0));

        auto idx = m_faces[i].vertex_id.x;
        m_normals[idx] += N;
        counts[idx]++;

        idx = m_faces[i].vertex_id.y;
        m_normals[idx] += N;
        counts[idx]++;

        idx = m_faces[i].vertex_id.z;
        m_normals[idx] += N;
        counts[idx]++;
    }
    for (size_t i = 0; i < m_vertices.size(); i++)
    {
        m_normals[i] /= counts[i];
        m_normals[i] = normalize(m_normals[i]);
    }
}

// ------------------------------------------------------------------
std::shared_ptr<TriangleMesh> createQuadMesh(
    const float u_min, const float u_max,
    const float v_min, const float v_max, 
    const float k, Axis axis) 
{
    // Prepare vertices and texcoords
    std::vector<std::array<float, 3>> temp_vertices;
    std::vector<float2> texcoords;

    for (int u=0; u<2; u++) 
    {
        /**
         * @note
         * - Axis::X ... u_axis = Y, v_axis = Z
         * - Axis::Y ... u_axis = X, v_axis = Z
         * - Axis::Z ... u_axis = X, v_axis = Y
         */
        int u_axis = ((int)axis + 1) % 3;
        int v_axis = ((int)axis + 2) % 3;
        if (axis == Axis::Y) 
            std::swap(u_axis, v_axis);

        for (int v=0; v<2; v++) 
        {
            std::array<float, 3> vertex { 0.0f, 0.0f, 0.0f };
            vertex[u_axis] = u_min + (float)u * (u_max - u_min);
            vertex[v_axis] = v_min + (float)v * (v_max - v_min);
            vertex[(int)axis] = k;
            texcoords.push_back(make_float2(u, v));
            temp_vertices.push_back(vertex);      
        }
    }

    // Transform vertices data from std::array<float, 3> to float3
    std::vector<float3> vertices;
    std::transform(temp_vertices.begin(), temp_vertices.end(), std::back_inserter(vertices), 
        [](const std::array<float, 3>& v) { return make_float3(v[0], v[1], v[2]); } );

    // Face indices
    std::vector<Face> faces;
    Face face1{make_int3(0, 1, 2), make_int3(0, 1, 2), make_int3(0, 1, 2)};
    Face face2{make_int3(2, 1, 3), make_int3(2, 1, 3), make_int3(2, 1, 3)};
    faces.push_back(face1);
    faces.push_back(face2);
    
    // Normals
    float n[3] = {0.0f, 0.0f, 0.0f};
    n[(int)axis] = 1.0f;
    std::vector<float3> normals(4, make_float3(n[0], n[1], n[2]));

    return make_shared<TriangleMesh>(vertices, faces, normals, texcoords, false);
}

// ------------------------------------------------------------------
std::shared_ptr<TriangleMesh> createPlaneMesh(
    const float2& min, const float2& max, const float2& res, Axis axis
)
{
    
}

// ------------------------------------------------------------------
std::shared_ptr<TriangleMesh> createSphereMesh(
    const float radius, const float2& res
)
{

}

// ------------------------------------------------------------------
// Icosphere
// ref: http://www.songho.ca/opengl/gl_sphere.html
std::shared_ptr<TriangleMesh> createIcoSphereMesh(
    const float radius, const int subdivisions
)
{
    Assert(subdivisions >= 0 && subdivisions < 10, 
        "prayground::createIsoSphereMesh(): The number of subdivision must be 0 ~ 9.");
    
    const int num_vertices = 2 + 10 * (2 * subdivisions - 1);

    std::vector<float3> vertices(12);
    std::vector<Face> faces(20);
    std::vector<float3> normals(20);
    std::vector<float2> texcoords(12);
    
    const float h_angle = radians(72.0f);
    const float v_angle = atanf(1.0f / 2.0f);

    // When subdivision level = 1
    vertices[0] = make_float3(0.0f, radius, 0.0f);   // top vertex
    vertices[11] = make_float3(0.0f, -radius, 0.0f); // bottom vertex
    texcoords[0] = getSphereUV(normalize(vertices[0]));
    texcoords[11] = getSphereUV(normalize(vertices[11]));

    for (int i = 1; i <= 5; i++)
    {
        int i00 = i;
        int i01 = i % 5 + 1;
        int i10 = i + 5;
        int i11 = i == 1 ? 10 : (i + 4) % 5 + 5;

        float3 v00 =
        {
            .x = radius * cosf(v_angle) * cosf(i * h_angle),
            .y = radius * sinf(v_angle),
            .z = radius * cosf(v_angle) * sinf(i * h_angle)
        };

        float3 v01 =
        {
            .x = radius * cosf(v_angle) * cosf((i+1) * h_angle),
            .y = radius * sinf(v_angle),
            .z = radius * cosf(v_angle) * sinf((i+1) * h_angle)
        };

        float3 v10 =
        {
            .x =  radius * cosf(v_angle) * cosf(i * h_angle + h_angle / 2.0f),
            .y = -radius * sinf(v_angle),
            .z =  radius * cosf(v_angle) * sinf(i * h_angle + h_angle / 2.0f)
        };

        float3 v11 =
        {
            .x =  radius * cosf(v_angle) * cosf((i + 1) * h_angle),
            .y = -radius * sinf(v_angle),
            .z =  radius * cosf(v_angle) * sinf((i + 1) * h_angle)
        };

        // upper row
        vertices[i00] = v00;
        texcoords[i00] = getSphereUV(normalize(vertices[i00]));

        // lower row
        vertices[i10] = v10;
        texcoords[i10] = getSphereUV(normalize(vertices[i10]));

        // Faces and normals
        int f0_id = i * 4;
        int f1_id = i * 4 + 1;
        int f2_id = i * 4 + 2;
        int f3_id = i * 4 + 3;

        Face f0 = {
            .vertex_id = {0, i00, i01},
            .normal_id = {f0_id, f0_id, f0_id},
            .texcoord_id = {0, i00, i01}
        };
        normals[f0_id] = normalize(cross(v00 - vertices[0], v01 - vertices[0]));

        Face f1 = {
            .vertex_id = {i00, i10, i01},
            .normal_id = {f1_id, f1_id, f1_id},
            .texcoord_id = {i00, i10, i01}
        };
        normals[f1_id] = normalize(cross(v10 - v00, v01 - v00));

        Face f2 = {
            .vertex_id = {i10, i00, i11},
            .normal_id = {f2_id, f2_id, f2_id},
            .texcoord_id = {i10, i00, i11}
        };
        normals[f2_id] = normalize(cross(v00 - v10, v11 - v10));

        Face f3 = {
            .vertex_id = {11, i10, i11},
            .normal_id = {f3_id, f3_id, f3_id},
            .texcoord_id = {11, i10, i11}
        };
        normals[f3_id] = normalize(cross(v10 - vertices[11], v11 - vertices[11]));
    }

    for (int i = 0; i < subdivisions; i++)
    {
        
    }
}

} // ::prayground