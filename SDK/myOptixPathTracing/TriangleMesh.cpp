#include "TriangleMesh.h"

// At present, only ".obj" format is supported.
TriangleMesh::TriangleMesh(const std::string& filename, float3 position, float size, float3 axis)
{
    if(filename.substr(filename.length() - 4) == ".obj")
    {
        std::ifstream ifs(filename, std::ios::in);
        for(std::string buffer; ifs >> buffer;)
        {
            if(buffer == "v")
            {
                float x, y, z;
                ifs >> x >> y >> z;
                x *= axis.x;
                y *= axis.y;
                z *= axis.z;
                vertices.emplace_back(x, y, z, 0.0f);
            } 
            else if(buffer == "f")
            {
                int i0, i1, i2;
                ifs >> i0 >> i1 >> i2;
                indices.emplace_back(make_int3(i0-1, i1-1, i2-1));
            }
        }
        ifs.close();
    }

    // Transformation
    Vertex center;
    auto min = vertices.front(), max = vertices.front();
    for (auto& vertex : vertices)
    {
        center += vertex / vertices.size();
        for (int i = 0; i < 3; i++)
        {
            if (vertex[i] < min[i]) min[i] = vertex[i];
            if (vertex[i] > max[i]) max[i] = vertex[i];
        }
    }

    auto scale = std::numeric_limits<float>::max();
    for (int i = 0; i < 3; i++)
    {
        float d = 1e-6f;    
        auto ratio = size / (max[i] - min[i] + d);
        if (ratio < scale)
            scale = ratio;
    }

    for (auto& vertex : vertices) {
        vertex = (vertex - center) * scale + position;
    }

    // Mesh smoothing
    if(indices.size() > 32)
    {
        normals = std::vector<Normal>(vertices.size(), Normal());
        auto counts = std::vector<int>(vertices.size(), 0);
        for(int i=0; i<indices.size(); i++)
        {
            auto p0 = vertices[indices[i].x];
            auto p1 = vertices[indices[i].y];
            auto p2 = vertices[indices[i].z];
            auto p0_f3 = make_float3(p0.x, p0.y, p0.z);
            auto p1_f3 = make_float3(p1.x, p1.y, p1.z);
            auto p2_f3 = make_float3(p2.x, p2.y, p2.z);
            auto N = normalize(cross(p2_f3 - p0_f3, p1_f3 - p0_f3));

            auto idx = indices[i].x;
            normals[idx] += Normal(N.x, N.y, N.z, 0.0f);
            counts[idx]++;

            idx = indices[i].y;
            normals[idx] += Normal(N.x, N.y, N.z, 0.0f);
            counts[idx]++;

            idx = indices[i].z;
            normals[idx] += Normal(N.x, N.y, N.z, 0.0f);
            counts[idx]++;
        }
        for(int i=0; i<vertices.size(); i++)
        {
            normals[i] /= counts[i];
            normals[i] = normalize(normals[i]);
        }
    }
}

TriangleMesh::TriangleMesh(std::vector<Vertex> vertices, std::vector<int3> indices, std::vector<Normal> normals) :
    vertices(vertices), 
    indices(indices), 
    normals(normals)
{
    assert(vertices.size() == normals.size());

    // Mesh smoothing
    if (indices.size() > 32)
    {
        normals = std::vector<Normal>(vertices.size(), Normal());
        auto counts = std::vector<int>(vertices.size(), 0);
        for (int i = 0; i < indices.size(); i++)
        {
            auto p0 = vertices[indices[i].x];
            auto p1 = vertices[indices[i].y];
            auto p2 = vertices[indices[i].z];
            auto p0_f3 = make_float3(p0.x, p0.y, p0.z);
            auto p1_f3 = make_float3(p1.x, p1.y, p1.z);
            auto p2_f3 = make_float3(p2.x, p2.y, p2.z);
            auto N = normalize(cross(p2_f3 - p0_f3, p1_f3 - p0_f3));

            auto idx = indices[i].x;
            normals[idx] += Normal(N.x, N.y, N.z, 0.0f);
            counts[idx]++;

            idx = indices[i].y;
            normals[idx] += Normal(N.x, N.y, N.z, 0.0f);
            counts[idx]++;

            idx = indices[i].z;
            normals[idx] += Normal(N.x, N.y, N.z, 0.0f);
            counts[idx]++;
        }
        for (int i = 0; i < vertices.size(); i++)
        {
            normals[i] /= counts[i];
            normals[i] = normalize(normals[i]);
        }
    }
}