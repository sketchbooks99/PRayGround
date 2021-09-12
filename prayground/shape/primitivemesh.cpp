#include "primitivemesh.h"
#include <algorithm>

namespace prayground {

namespace {

float2 getSphereUV(const float3& p)
{
    float phi = atan2(p.z, p.x);
    float theta = asin(p.y);
    float u = 1.0f - (phi + math::pi) / (math::two_pi);
    float v = 1.0f - (theta + math::pi / 2.0f) / math::pi;
    return make_float2(u, v);
}

} // ::nonamed namespace

// ---------------------------------------------------------
IcoSphereMesh::IcoSphereMesh(float radius, int level)
: m_radius(radius), m_level(level)
{
    Assert(level >= 0 && level < 20, 
        "prayground::IcoSphereMesh(): The level of details must be 0 ~ 19.");

    const int l2 = (level+1) * (level+1);
    const int num_vertices = 2 + l2 * 10;
    const int num_faces = 20 * l2;

    m_vertices.resize(num_vertices);
    m_faces.resize(num_vertices);
    m_normals.resize(num_vertices);
    m_texcoords.resize(num_vertices);
    
    const float v_ang_center = atanf(1.0f / 2.0f);

    float3 top_v{0.0f, radius, 0.0f};
    float3 bottom_v{0.0f, -radius, 0.0f};

    // top vertex
    m_vertices[0] = top_v;
    m_texcoords[0] = getSphereUV(normalize(top_v));
    // bottom vertex
    m_vertices[num_vertices-1] = bottom_v; 
    m_texcoords[num_vertices-1] = getSphereUV(normalize(bottom_v));

    //            . <- top vertex
    //           / \ 
    //          . - .            upper row
    //         / \ / \           ------------
    //      - . - . - . -   
    // ... \ / \ / \ / \ / ...
    //      . - . - . - .        center row
    // ... / \ / \ / \ / \ ...
    //      - . - . - . -   
    //         \ / \ /           ------------
    //          . - .            lower row
    //           \ /
    //            . <- bottom vertex
    //

    int v_idx = 1;

    // Upper row
    for (int ul = 0; ul < level; ul++)
    {
        float h_ang_per_tri = math::two_pi / (float)((ul+1)*5);
        for (int j = 0; j < (ul+1)*5; j++)
        {
            float h_ang = j * h_ang_per_tri + (float)(ul % 5) * (h_ang_per_tri / 2.0f);
            float v_ang = std::lerp(math::pi / 2.0f, v_ang_center, (1.0f / (float)(level+1)) * (float)(ul+1));

            float x = radius * cosf(v_ang) * cosf(h_ang);
            float y = radius * sinf(v_ang);
            float z = radius * cosf(v_ang) * sinf(h_ang);
            m_vertices[v_idx] = make_float3(x, y, z);

            // 頂点を1つ追加するごとに面を2つ追加する
            // 例) v_idx = 1 (level=1)
            //     0                      
            //    / \   (1,2,0)
            //   1 - 2
            //    \ /   (1,6,2)
            //     6
            //
            //
            for (int k = 0; k < 2; k++)
            {

            }

            v_idx++;
        }
    }

    // Center row
    for (int cl = 0; cl <= level + 1; cl++)
    {
        float h_ang_per_tri = math::two_pi / (float)((level + 1) / 5.0f);
        for (int j = 0; j < (level+1) * 5; j++)
        {
            float h_ang = j * h_ang_per_tri + (cl % 2) * (h_ang_per_tri / 2.0f);
            float v_ang = std::lerp(-v_ang_center, v_ang_center, ((level + 1) - cl) * (1.0f / (float)(level + 1)));

            float x = radius * cosf(v_ang) * cosf(h_ang);
            float y = radius * sinf(v_ang);
            float z = radius * cosf(v_ang) * sinf(h_ang);
            m_vertices[v_idx] = make_float3(x, y, z);
            v_idx++;
        }
    }

    // Lower row
    for (int ll = level-1; ll >= 0; ll--)
    {
        float h_ang_per_tri = math::two_pi / (float)((ll + 1) / 5.0f);
        for (int j = 0; j < (ll+1)*5; j++)
        {
            float h_ang = j * h_ang_per_tri + h_ang_per_tri / 2.0f;
            float v_ang = std::lerp(-v_ang_center, -math::pi / 2.0f, (1.0f / (level+1)) * (float)(level - ll));

            float x = radius * cosf(v_ang) * cosf(h_ang);
            float y = radius * sinf(v_ang);
            float z = radius * cosf(v_ang) * sinf(h_ang);
            m_vertices[v_idx] = make_float3(x, y, z);
            v_idx++;
        }
    }
}

// ---------------------------------------------------------
/**
 * @brief 
 * 法線の平滑化を行う
 * 頂点の共有が外れているか否かで処理を分岐させるかは検討すべき点
 */
void IcoSphereMesh::smooth()
{

}

// ---------------------------------------------------------
/**
 * @brief 
 * 各面における頂点の共有を外す
 * 各三角形ごとの移動などができるようになるが、頂点用のデータ量が増加する
 */
void IcoSphereMesh::splitVertices()
{

}

// ---------------------------------------------------------
UVSphereMesh::UVSphereMesh(float radius, int2 res)
{
    TODO_MESSAGE();
}

// ---------------------------------------------------------
CylinderMesh::CylinderMesh(float radius, float height)
{
    TODO_MESSAGE();
}

// ---------------------------------------------------------
PlaneMesh::PlaneMesh(float2 size, int2 res, Axis axis)
{
    std::vector<std::array<float, 3>> temp_vertices;

    const float u_min = -size.x / 2.0f;
    const float v_min = -size.y / 2.0f;
    const float u_step = size.x / (float)res.x;
    const float v_step = size.y / (float)res.y;

    for (int v = 0; v <= res.y; v++)
    {
        int u_axis = ((int)axis + 1) % 3;
        int v_axis = ((int)axis + 2) % 3;
        if (axis == Axis::Y)
            std::swap(u_axis, v_axis);
        
        for (int u = 0; u <= res.x; u++)
        {
            std::array<float, 3> vertex{ 0.0f, 0.0f, 0.0f };
            vertex[u_axis] = u_min + (float)u * u_step;
            vertex[v_axis] = v_min + (float)v * v_step;
            vertex[(int)axis] = 0.0f;
            m_texcoords.push_back(make_float2((float)u / res.x, (float)v / res.y));
            temp_vertices.push_back(vertex);

            if (u < res.x && v < res.y)
            {
                // i00 - i01 ...
                //  |  \  |  
                // i10 - i11 ...
                //  |  \  |

                int i00 = res.x * v + u;
                int i01 = i00 + 1;
                int i10 = res.x * (v + 1) + u;
                int i11 = i10 + 1;
                Face f1{make_int3(i00, i10, i11), make_int3(i00, i10, i11), make_int3(i00, i10, i11)};
                Face f2{make_int3(i00, i11, i01), make_int3(i00, i11, i01), make_int3(i00, i11, i01)};
                m_faces.push_back(f1); 
                m_faces.push_back(f2);
            }
        }
    }

    std::transform(temp_vertices.begin(), temp_vertices.end(), std::back_insert_iterator(m_vertices),
        [](const std::array<float, 3>& v) { return make_float3(v[0], v[1], v[2]); });

    float n[3] = {0.0f, 0.0f, 0.0f};
    n[(int)axis] = 1.0f;
    m_normals.resize(m_vertices.size());
    std::fill(m_normals.begin(), m_normals.end(), make_float3(n[0], n[1], n[2]));
}

} // ::prayground