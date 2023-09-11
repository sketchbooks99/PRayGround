#include "primitivemesh.h"
#include <algorithm>

namespace prayground {

    namespace {

        Vec2f getSphereUV(const Vec3f& p)
        {
            float phi = atan2(p.z(), p.x());
            float theta = asin(p.y());
            float u = 1.0f - (phi + math::pi) / (math::two_pi);
            float v = 1.0f - (theta + math::pi / 2.0f) / math::pi;
            return Vec2f(u, v);
        }

    } // nonamed namespace

    // ---------------------------------------------------------
    IcoSphereMesh::IcoSphereMesh(float radius, int level)
    : m_radius(radius), m_level(level)
    {
        ASSERT(level >= 0 && level < 20, "The level of subdivision must be 0 to 19.");

        const float H_ANGLE = math::pi / 180.0f * 72.0f;
        const float V_ANGLE = atanf(1.0f / 2.0f);

        m_vertices.resize(12);
        m_texcoords.resize(12);
        m_faces.resize(20);

        float h_angle1 = -math::pi / 2.0f - H_ANGLE / 2.0f;
        float h_angle2 = -math::pi / 2.0f;

        // Top vertex
        m_vertices[0] = Vec3f(0, radius, 0);
        m_texcoords[0] = Vec2f(0, 0);
        m_indices.emplace(std::make_pair<float, float>(0, 0), 0);

        // 10 vertices at 2nd and 3rd rows
        for (int32_t i = 1; i <= 5; i++)
        {
            int32_t i0 = i;
            int32_t i1 = i + 5;

            const float y = m_radius * sinf(V_ANGLE);
            const float xz = m_radius * cosf(V_ANGLE);

            const Vec3f v0(xz * cosf(h_angle1), y, xz * sinf(h_angle1));
            const Vec3f v1(xz * cosf(h_angle2), -y, xz * sinf(h_angle2));
            m_vertices[i0] = v0;
            m_vertices[i1] = v1;
            const Vec2f uv0 = getSphereUV(normalize(v0));
            const Vec2f uv1 = getSphereUV(normalize(v1));
            std::pair<float, float> key0(uv0.x(), uv0.y());
            std::pair<float, float> key1(uv1.x(), uv1.y());
            m_indices.emplace(key0, i0);
            m_indices.emplace(key1, i1);
            m_texcoords[i0] = uv0;
            m_texcoords[i1] = uv1;

            // Top face
            Face f0{ {0, i0, i0 % 5 + 1}, {0, 0, 0}, {0, i0, i0 % 5 + 1} };
            // Middle face
            Face f1{ {i0, i1, i0 % 5 + 1}, {0, 0, 0}, {i0, i1, i1 % 5 + 1} };
            Face f2{ {i1, i1 % 5 + 6, i0 % 5 + 1}, {0, 0, 0}, {i1, i1 % 5 + 6, i0 % 5 + 1} };
            // Bottom face
            Face f3{ {i1, 11, i1 % 5 + 6}, {0, 0, 0}, {i1, 11, i1 % 5 + 6} };

            m_faces[i - 1] = f0;
            m_faces[5 + (i - 1) * 2 + 0] = f1;
            m_faces[5 + (i - 1) * 2 + 1] = f2;
            m_faces[15 + (i - 1)] = f3;

            // Move to next angles
            h_angle1 += H_ANGLE;
            h_angle2 += H_ANGLE;
        }

        // Bottom vertex
        m_vertices[11] = Vec3f(0, -radius, 0);
        m_texcoords[11] = Vec2f(0, 1);
        m_indices.emplace(std::make_pair<float, float>(0, 1), 11);

        int32_t n_id = 0;
        for (auto& f : m_faces)
        {
            const Vec3f& v0 = m_vertices[f.vertex_id.x()];
            const Vec3f& v1 = m_vertices[f.vertex_id.y()];
            const Vec3f& v2 = m_vertices[f.vertex_id.z()];

            const Vec3f n = normalize(cross(v1 - v0, v2 - v0));

            m_normals.emplace_back(n);
            m_normals.emplace_back(n);
            m_normals.emplace_back(n);

            f.normal_id = Vec3i(n_id, n_id + 1, n_id + 2);
            n_id += 3;
        }

        subdivide(m_level);
    }

    // Subdivide each faces to 4 faces
    // Prohibit m_level(current level) + level exceeds 20 due to decline of performance
    void IcoSphereMesh::subdivide(const float level)
    {
        PG_LOG("Processing subdivision of Icosphere (subdivision level =", level, ") ...");

        const int32_t total_level = m_level + level;
        ASSERT(total_level >= 0 && total_level < 20, "The level of subdivision must be 0 to 19.");

        std::vector<Face> temp_faces;
        int32_t index = 0;

        for (int32_t i = 1; i <= total_level; i++)
        {
            // Copy previous vertices/faces and clear it
            temp_faces = m_faces;
            m_faces.clear();
            index = 0;

            // Perform subdivision for each triangle
            for (int32_t j = 0; j < temp_faces.size(); j++)
            {
                const int32_t i0 = temp_faces[j].vertex_id.x();
                const int32_t i1 = temp_faces[j].vertex_id.y();
                const int32_t i2 = temp_faces[j].vertex_id.z();
                const Vec3f v0 = m_vertices[i0];
                const Vec3f v1 = m_vertices[i1];
                const Vec3f v2 = m_vertices[i2];

                /**
                 * Add 3 vertices between each edges, 4 triangle faces 
                 * 
                 *              v0 *
                 *                / \
                 *        new_v0 * - * new_v2
                 *              / \ / \
                 *          v1 * - * - * v2
                 *                 new_v1
                 */

                Vec3f new_v0 = lerp(v0, v1, 0.5f);
                Vec3f new_v1 = lerp(v1, v2, 0.5f);
                Vec3f new_v2 = lerp(v2, v0, 0.5f);
                new_v0 = normalize(new_v0) * m_radius;
                new_v1 = normalize(new_v1) * m_radius;
                new_v2 = normalize(new_v2) * m_radius;
                Vec2f new_uv0 = getSphereUV(new_v0 * (1.0f / m_radius));
                Vec2f new_uv1 = getSphereUV(new_v1 * (1.0f / m_radius));
                Vec2f new_uv2 = getSphereUV(new_v2 * (1.0f / m_radius));

                int32_t new_i0 = findOrAddVertex(new_v0, new_uv0);
                int32_t new_i1 = findOrAddVertex(new_v1, new_uv1);
                int32_t new_i2 = findOrAddVertex(new_v2, new_uv2);

                Face f0{ {i0, new_i0, new_i2}, {0, 0, 0}, {i0, new_i0, new_i2} };
                Face f1{ {new_i0, i1, new_i1}, {0, 0, 0}, {new_i0, i1, new_i1} };
                Face f2{ {new_i0, new_i1, new_i2}, {0, 0, 0}, {new_i0, new_i1, new_i2} };
                Face f3{ {new_i2, new_i1, i2}, {0, 0, 0}, {new_i2, new_i1, i2} };

                addFace(f0); addFace(f1); addFace(f2); addFace(f3);
            }
        }

        m_normals.clear();
        int32_t n_id = 0;
        for (auto& f : m_faces)
        {
            const Vec3f& v0 = m_vertices[f.vertex_id.x()];
            const Vec3f& v1 = m_vertices[f.vertex_id.y()];
            const Vec3f& v2 = m_vertices[f.vertex_id.z()];

            const Vec3f n = normalize(cross(v2 - v0, v1 - v0));

            addNormal(n);
            addNormal(n);
            addNormal(n);

            f.normal_id = Vec3i(n_id, n_id + 1, n_id + 2);
            n_id += 3;
        }
    }

    // ---------------------------------------------------------
    /**
     * @note 
     * Split shared vertices to independent vertices
     */
    void IcoSphereMesh::splitVertices()
    {
        UNIMPLEMENTED();
    }

    int32_t IcoSphereMesh::findOrAddVertex(const Vec3f& v, const Vec2f& texcoord)
    {
        std::pair<float, float> key(texcoord.x(), texcoord.y());
        auto itr = m_indices.find(key);
        // Found the same vertex and return it's index
        if (itr != m_indices.end())
            return itr->second;
        // There has been no same vertex yet, and add vertex/texcoord
        else
        {
            int32_t index = static_cast<int32_t>(m_indices.size());
            addVertex(v);
            addTexcoord(texcoord);
            m_indices.emplace(key, index);
            return index;
        }
    }

    // ---------------------------------------------------------
    UVSphereMesh::UVSphereMesh(float radius, const Vec2ui& resolution)
        : m_radius(radius), m_resolution(resolution)
    {
        const int32_t total_num_vertices = resolution.x() * (resolution.y() - 1) + 2;

        // Add top vertex
        addVertex(0, radius, 0);
        addTexcoord(0, 0);

        // Add vertices on side
        for (uint32_t v = 1; v < resolution.y(); v++)
        {
            for (uint32_t u = 0; u < resolution.x(); u++)
            {
                const float phi = (math::pi / 2.0f) - ((float)v / resolution.y()) * math::pi;
                const float theta = math::two_pi * ((float)u / resolution.x());
                const float x = cosf(phi) * cosf(theta);
                const float y = sinf(phi);
                const float z = cosf(phi) * sinf(theta);
                Vec3f vertex = Vec3f(x, y, z) * radius;
                addVertex(vertex);
                addTexcoord(getSphereUV(normalize(vertex)));
            }
        }

        // Add bottom vertex
        addVertex(0, -radius, 0);
        addTexcoord(0, 1);

        for (uint32_t v = 0; v < resolution.y(); v++)
        {
            for (uint32_t u = 0; u < resolution.x(); u++)
            {
                if (v == 0)
                {
                    //       i0
                    //       * <- top vertex
                    //      / \
                    // ... * - * ...
                    //    i1   i2

                    const int32_t i0 = 0;
                    const int32_t i1 = u + 1;
                    const int32_t i2 = (u + 1) % resolution.x() + 1;

                    const Vec3f v0 = vertexAt(i0);
                    const Vec3f v1 = vertexAt(i1);
                    const Vec3f v2 = vertexAt(i2);

                    const Vec3f n = normalize(cross(v2 - v0, v1 - v0));

                    // Get base index of normal
                    const int32_t n_base_idx = numNormals();
                    // Add three normals for vertices that constructs triangle face
                    for (int i = 0; i < 3; i++)
                        addNormal(n);

                    const Face f{ {i0, i1, i2}, {n_base_idx, n_base_idx + 1, n_base_idx + 2}, {i0, i1, i2} };
                    addFace(f);
                }
                else if (v == resolution.y() - 1)
                {
                    //      i0   i2
                    // ... - * - * - ...
                    //        \ /
                    //         * <- bottom vertex
                    //         i1

                    const int32_t i0 = (v - 1) * resolution.x() + u + 1;
                    const int32_t i1 = total_num_vertices - 1;
                    const int32_t i2 = (v - 1) * resolution.x() + (u + 1) % resolution.x() + 1;

                    const Vec3f v0 = vertexAt(i0);
                    const Vec3f v1 = vertexAt(i1);
                    const Vec3f v2 = vertexAt(i2);

                    const Vec3f n = normalize(cross(v2 - v0, v1 - v0));

                    // Get base index of normal
                    const int32_t n_base_idx = numNormals();
                    // Add three normals for vertices that construct triangle face
                    for (int i = 0; i < 3; i++)
                        addNormal(n);

                    const Face f{ {i0, i1, i2}, {n_base_idx, n_base_idx + 1, n_base_idx + 2}, {i0, i1, i2} };
                    addFace(f);
                }
                else
                {
                    //      i0   i1
                    // ... - * - * - ...
                    //       | \ | 
                    // ... - * - * - ... 
                    //      i2   i3

                    const int32_t i0 = (v - 1) * resolution.x() + u + 1;
                    const int32_t i1 = (v - 1) * resolution.x() + (u + 1) % resolution.x() + 1;
                    const int32_t i2 = v * resolution.x() + u + 1;
                    const int32_t i3 = v * resolution.x() + (u + 1) % resolution.x() + 1;

                    const Vec3f v0 = vertexAt(i0);
                    const Vec3f v1 = vertexAt(i1);
                    const Vec3f v2 = vertexAt(i2);
                    const Vec3f v3 = vertexAt(i3);

                    const Vec3f n0 = normalize(cross(v3 - v0, v2 - v0));
                    const Vec3f n1 = normalize(cross(v1 - v0, v3 - v0));

                    const int32_t n_base_idx = numNormals();
                    // Add three normals for vertices that construct triangle face
                    for (int i = 0; i < 3; i++)
                        addNormal(n0);
                    for (int i = 0; i < 3; i++)
                        addNormal(n1);

                    Face f0{ {i0, i2, i3}, {n_base_idx, n_base_idx + 1, n_base_idx + 2}, {i0, i2, i3} };
                    Face f1{ {i0, i3, i1}, {n_base_idx + 3, n_base_idx + 4, n_base_idx + 5}, {i0, i3, i1} };

                    addFace(f0); addFace(f1);
                }
            }
        }

        PG_LOG("Dummy");
    }

    float UVSphereMesh::radius() const
    {
        return m_radius;
    }

    void UVSphereMesh::setRadius(const float radius)
    {
        m_radius = radius;
    }

    const Vec2ui& UVSphereMesh::resolution() const
    {
        return m_resolution;
    }

    void UVSphereMesh::setResolution(const Vec2ui& resolution)
    {
        m_resolution = resolution;
    }

    // ---------------------------------------------------------
    CylinderMesh::CylinderMesh(float radius, float height, const Vec2ui& resolution)
        : m_radius(radius), m_height(height), m_resolution(resolution)
    {
        const Vec3f top_center_vertex(0.0f, height / 2.0f, 0.0f);
        const Vec3f bottom_center_vertex(0.0f, -height / 2.0f, 0.0f);
        const int32_t total_num_vertices = m_resolution.x() * (m_resolution.y() + 1) + 2;

        addVertex(top_center_vertex);
        addTexcoord(Vec2f(0, 0));
        addNormal(Vec3f(0, 1, 0));

        int32_t num_top_vertices = m_resolution.x() + 1;

        // Add faces for top surface
        for (uint32_t r = 0; r < m_resolution.x(); r++)
        {
            int32_t i1 = r + 1;
            int32_t i2 = (r + 1) % m_resolution.x() + 1;
            Face f{{0, i1, i2}, {0, i1, i2}, {0, i1, i2}};
            addFace(f);
            addNormal(Vec3f(0, 1, 0));
        }

        // Add vertices/texcoords on the side
        for (uint32_t h = 0; h <= m_resolution.y(); h++)
        {
            for (uint32_t r = 0; r < m_resolution.x(); r++)
            {
                float x = sinf(((float)r / m_resolution.x()) * math::two_pi) * radius;
                float z = cosf(((float)r / m_resolution.x()) * math::two_pi) * radius;
                float y = height / 2.0f - ((float)h / m_resolution.y()) * height;
                // Side faces 
                Vec3f v(x, y, z);
                addVertex(v);
                addTexcoord(Vec2f((float)r / m_resolution.x(), (float)h / m_resolution.y()));
            }
        }

        for (uint32_t h = 0; h < m_resolution.y(); h++)
        {
            for (uint32_t r = 0; r < m_resolution.x(); r++)
            {
                /* i0   i1
                 *  .---.
                 *  | \ |
                 *  .---.
                 * i2   i3 */
                int32_t i0 = h * m_resolution.x() + r + 1;
                int32_t i1 = h * m_resolution.x() + (r + 1) % m_resolution.x() + 1;
                int32_t i2 = (h + 1) * m_resolution.x() + r + 1;
                int32_t i3 = (h + 1) * m_resolution.x() + (r + 1) % m_resolution.x() + 1;

                auto v0 = vertexAt(i0);
                auto v1 = vertexAt(i1);
                auto v2 = vertexAt(i2);
                auto v3 = vertexAt(i3);

                int32_t in0 = numNormals();
                int32_t in1 = numNormals() + 3;

                // Add normal for vertex at i0 and i2 
                Vec3f n0 = normalize(cross(v2 - v0, v3 - v0));
                addNormal(n0); addNormal(n0); addNormal(n0);

                Vec3f n1 = normalize(cross(v3 - v0, v1 - v0));
                addNormal(n1); addNormal(n1); addNormal(n1);

                Face f0 = { {i0, i2, i3}, {in0, in0 + 1, in0 + 2}, {i0, i2, i3} };
                Face f1 = { {i0, i3, i1}, {in1, in1 + 1, in1 + 2}, {i0, i3, i1} };

                addFace(f0); addFace(f1);
            }
        }
        
        int32_t n_base_idx = numNormals();
        // Add faces for bottom surface
        for (uint32_t r = 0; r < m_resolution.x(); r++)
        {
            int32_t base_idx = m_resolution.x() * m_resolution.y() + 1;
            int32_t num_total_normals = numNormals() + num_top_vertices;
            int32_t i0 = total_num_vertices - 1;
            int32_t i1 = base_idx + r;
            int32_t i2 = base_idx + (r + 1) % m_resolution.x();
            Face f{ Vec3i{i0, i1, i2}, 
                Vec3i{num_total_normals - 1, static_cast<int32_t>(n_base_idx + r), static_cast<int32_t>(n_base_idx + (r + 1) % resolution.x())},
                Vec3i{i0, i1, i2}};
            addFace(f);
            addNormal(Vec3f(0, -1, 0));
        }

        addVertex(bottom_center_vertex);
        addTexcoord(Vec2f(0.0f));
        addNormal(Vec3f(0.0f, -1.0f, 0.0f));
    }

    float CylinderMesh::radius() const
    {
        return m_radius;
    }

    void CylinderMesh::setRadius(const float radius)
    {
        m_radius = radius;
    }

    float CylinderMesh::height() const
    {
        return m_height;
    }

    void CylinderMesh::setHeight(const float height)
    {
        m_height = height;
    }

    const Vec2ui& CylinderMesh::resolution() const
    {
        return m_resolution;
    }

    void CylinderMesh::setResolution(const Vec2ui& resolution)
    {
        m_resolution = resolution;
    }

    // ---------------------------------------------------------
    PlaneMesh::PlaneMesh(const Vec2f& size, const Vec2ui& resolution, Axis axis)
        : m_size(size), m_resolution(resolution), m_axis(axis)
    {
        const float u_min = -m_size.x() / 2.0f;
        const float v_max = m_size.y() / 2.0f;
        const float u_step = m_size.x() / (float)m_resolution.x();
        const float v_step = m_size.y() / (float)m_resolution.y();

        for (int v = 0; v <= m_resolution.y(); v++)
        {
            int u_axis = ((int)m_axis + 1) % 3;
            int v_axis = ((int)m_axis + 2) % 3;
            if (m_axis == Axis::Y)
                std::swap(u_axis, v_axis);

            for (int u = 0; u <= m_resolution.x(); u++)
            {
                Vec3f vertex(0.0f);
                vertex[u_axis] = u_min + (float)u * u_step;
                vertex[v_axis] = v_max - (float)v * v_step;
                vertex[(int)m_axis] = 0.0f;
                addTexcoord(Vec2f((float)u / m_resolution.x(), (float)v / m_resolution.y()));
                addVertex(vertex);

                if (u == m_resolution.x() || v == m_resolution.y())
                    continue;

                // i00 - i01 ...
                //  |  \  |  
                // i10 - i11 ...
                //  |  \  |

                int i00 = (m_resolution.x() + 1) * v + u;
                int i01 = i00 + 1;
                int i10 = (m_resolution.x() + 1) * (v + 1) + u;
                int i11 = i10 + 1;
                Face f1{ Vec3i(i00, i10, i11), Vec3i(i00, i10, i11), Vec3i(i00, i10, i11) };
                Face f2{ Vec3i(i00, i11, i01), Vec3i(i00, i11, i01), Vec3i(i00, i11, i01) };
                m_faces.push_back(f1);
                m_faces.push_back(f2);
            }
        }

        Vec3f n{ 0.0f, 0.0f, 0.0f };
        n[(int)m_axis] = 1.0f;
        m_normals.resize(m_vertices.size());
        std::fill(m_normals.begin(), m_normals.end(), n);
    }

    const Vec2f& PlaneMesh::size() const
    {
        return m_size;
    }

    void PlaneMesh::setSize(const Vec2f& size)
    {
        m_size = size;
    }

    const Vec2ui& PlaneMesh::resolution() const
    {
        return m_resolution;
    }

    void PlaneMesh::setResolution(const Vec2ui& resolution)
    {
        m_resolution = resolution;
    }

} // namespace prayground