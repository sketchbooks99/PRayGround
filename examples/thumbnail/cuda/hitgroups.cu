#include "util.cuh"
#include <prayground/shape/plane.h>
#include <prayground/shape/trianglemesh.h>
#include <prayground/shape/sphere.h>
#include <prayground/shape/cylinder.h>
#include <prayground/shape/box.h>
#include <prayground/core/ray.h>
#include <prayground/core/onb.h>
#include <prayground/core/bsdf.h>

using namespace prayground;

extern "C" __device__ void __closesthit__shadow()
{
    setPayload<0>(1u);
}

// Box -------------------------------------------------------------------------------
static INLINE DEVICE float2 getBoxUV(const float3& p, const float3& min, const float3& max, const int axis)
{
    float2 uv;
    int u_axis = (axis + 1) % 3;
    int v_axis = (axis + 2) % 3;

    // axisがYの時は (u: Z, v: X) -> (u: X, v: Z)へ順番を変える
    if (axis == 1) swap(u_axis, v_axis);

    uv.x = (getByIndex(p, u_axis) - getByIndex(min, u_axis)) / (getByIndex(max, u_axis) - getByIndex(min, u_axis));
    uv.y = (getByIndex(p, v_axis) - getByIndex(min, v_axis)) / (getByIndex(max, v_axis) - getByIndex(min, v_axis));

    return clamp(uv, 0.0f, 1.0f);
}

// Return hit axis
static INLINE DEVICE int hitBox(
    const BoxData* box_data,
    const float3& o, const float3& v,
    const float tmin, const float tmax,
    SurfaceInteraction& si)
{
    float3 min = box_data->min;
    float3 max = box_data->max;

    float _tmin = tmin, _tmax = tmax;
    int min_axis = -1, max_axis = -1;

    for (int i = 0; i < 3; i++)
    {
        float t0, t1;
        if (getByIndex(v, i) == 0.0f)
        {
            t0 = fminf(getByIndex(min, i) - getByIndex(o, i), getByIndex(max, i) - getByIndex(o, i));
            t1 = fmaxf(getByIndex(min, i) - getByIndex(o, i), getByIndex(max, i) - getByIndex(o, i));
        }
        else
        {
            t0 = fminf((getByIndex(min, i) - getByIndex(o, i)) / getByIndex(v, i),
                (getByIndex(max, i) - getByIndex(o, i)) / getByIndex(v, i));
            t1 = fmaxf((getByIndex(min, i) - getByIndex(o, i)) / getByIndex(v, i),
                (getByIndex(max, i) - getByIndex(o, i)) / getByIndex(v, i));
        }
        min_axis = t0 > _tmin ? i : min_axis;
        max_axis = t1 < _tmax ? i : max_axis;

        _tmin = fmaxf(t0, _tmin);
        _tmax = fminf(t1, _tmax);

        if (_tmax < _tmin)
            return -1;
    }

    float3 center = (min + max) / 2.0f;
    if ((tmin < _tmin && _tmin < tmax) && (-1 < min_axis && min_axis < 3))
    {
        float3 p = o + _tmin * v;
        float3 center_axis = p;
        setByIndex(center_axis, min_axis, getByIndex(center, min_axis));
        float3 normal = normalize(p - center_axis);
        float2 uv = getBoxUV(p, min, max, min_axis);
        si.p = p;
        si.n = normal;
        si.uv = uv;
        si.t = _tmin;
        return min_axis;
    }

    if ((tmin < _tmax && _tmax < tmax) && (-1 < max_axis && max_axis < 3))
    {
        float3 p = o + _tmax * v;
        float3 center_axis = p;
        setByIndex(center_axis, max_axis, getByIndex(center, max_axis));
        float3 normal = normalize(p - center_axis);
        float2 uv = getBoxUV(p, min, max, max_axis);
        si.p = p;
        si.n = normal;
        si.uv = uv;
        si.t = _tmax;
        return max_axis;
    }
    return -1;
}

extern "C" __device__ void __intersection__box()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const int prim_id = optixGetPrimitiveIndex();
    BoxData box_data = reinterpret_cast<BoxData*>(data->shape_data)[prim_id];

    Ray ray = getLocalRay();

    SurfaceInteraction si;
    int hit_axis = hitBox(&box_data, ray.o, ray.d, ray.tmin, ray.tmax, si);
    if (hit_axis >= 0) {
        optixReportIntersection(si.t, 0, float3_as_ints(si.n), float2_as_ints(si.uv), hit_axis);
    }
}

extern "C" __device__ void __closesthit__box()
{
    HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());

    Ray ray = getWorldRay();

    float3 local_n = getFloat3FromAttribute<0>();
    float2 uv = getFloat2FromAttribute<3>();
    uint32_t hit_axis = getAttribute<5>();

    float3 world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);

    SurfaceInteraction* si = getSurfaceInteraction();

    si->p = ray.at(ray.tmax);
    si->n = world_n;
    si->t = ray.tmax;
    si->wi = ray.d;
    si->uv = uv;
    si->surface_info = data->surface_info;

    float3 dpdu, dpdv;
    // x
    if (hit_axis == 0)
    {
        dpdu = make_float3(0.0f, 0.0f, 1.0f);
        dpdv = make_float3(0.0f, 1.0f, 0.0f);
    }
    else if (hit_axis == 1)
    {
        dpdu = make_float3(1.0f, 0.0f, 0.0f);
        dpdv = make_float3(0.0f, 0.0f, 1.0f);
    }
    else if (hit_axis == 2)
    {
        dpdu = make_float3(1.0f, 0.0f, 0.0f);
        dpdv = make_float3(0.0f, 1.0f, 0.0f);
    }
    si->dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(dpdu));
    si->dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(dpdv));
}

// Cylinder -------------------------------------------------------------------------------
static INLINE DEVICE float2 getCylinderUV(
    const float3& p, const float radius, const float height, const bool hit_disk
)
{
    if (hit_disk)
    {
        const float r = sqrtf(p.x * p.x + p.z * p.z) / radius;
        const float theta = atan2(p.z, p.x);
        float u = 1.0f - (theta + math::pi / 2.0f) / math::pi;
        return make_float2(u, r);
    }
    else
    {
        const float theta = atan2(p.z, p.x);
        const float v = (p.y + height / 2.0f) / height;
        float u = 1.0f - (theta + math::pi / 2.0f) / math::pi;
        return make_float2(u, v);
    }
}

extern "C" __device__ void __intersection__cylinder()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const CylinderData* cylinder = reinterpret_cast<CylinderData*>(data->shape_data);

    const float radius = cylinder->radius;
    const float height = cylinder->height;

    Ray ray = getLocalRay();

    const float a = dot(ray.d, ray.d) - ray.d.y * ray.d.y;
    const float half_b = (ray.o.x * ray.d.x + ray.o.z * ray.d.z);
    const float c = dot(ray.o, ray.o) - ray.o.y * ray.o.y - radius * radius;
    const float discriminant = half_b * half_b - a * c;

    if (discriminant > 0.0f)
    {
        const float sqrtd = sqrtf(discriminant);
        const float side_t1 = (-half_b - sqrtd) / a;
        const float side_t2 = (-half_b + sqrtd) / a;

        const float side_tmin = fmin(side_t1, side_t2);
        const float side_tmax = fmax(side_t1, side_t2);

        if (side_tmin > ray.tmax || side_tmax < ray.tmin)
            return;

        const float upper = height / 2.0f;
        const float lower = -height / 2.0f;
        const float y_tmin = fmin((lower - ray.o.y) / ray.d.y, (upper - ray.o.y) / ray.d.y);
        const float y_tmax = fmax((lower - ray.o.y) / ray.d.y, (upper - ray.o.y) / ray.d.y);

        float t1 = fmax(y_tmin, side_tmin);
        float t2 = fmin(y_tmax, side_tmax);
        if (t1 > t2 || (t2 < ray.tmin) || (t1 > ray.tmax))
            return;

        bool check_second = true;
        if (ray.tmin < t1 && t1 < ray.tmax)
        {
            float3 P = ray.at(t1);
            bool hit_disk = y_tmin > side_tmin;
            float3 normal = hit_disk
                ? normalize(P - make_float3(P.x, 0.0f, P.z))   // Hit at disk
                : normalize(P - make_float3(0.0f, P.y, 0.0f)); // Hit at side
            float2 uv = getCylinderUV(P, radius, height, hit_disk);
            optixReportIntersection(t1, 0, float3_as_ints(normal), float2_as_ints(uv));
            check_second = false;
        }

        if (check_second)
        {
            if (ray.tmin < t2 && t2 < ray.tmax)
            {
                float3 P = ray.at(t2);
                bool hit_disk = y_tmax < side_tmax;
                float3 normal = hit_disk
                    ? normalize(P - make_float3(P.x, 0.0f, P.z))   // Hit at disk
                    : normalize(P - make_float3(0.0f, P.y, 0.0f)); // Hit at side
                float2 uv = getCylinderUV(P, radius, height, hit_disk);
                optixReportIntersection(t2, 0, float3_as_ints(normal), float2_as_ints(uv));
            }
        }
    }
}

extern "C" __device__ void __closesthit__cylinder()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());

    Ray ray = getWorldRay();

    float3 local_n = getFloat3FromAttribute<0>();

    float2 uv = getFloat2FromAttribute<3>();

    float3 world_n = optixTransformNormalFromObjectToWorldSpace(local_n);

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->n = normalize(world_n);
    si->t = ray.tmax;
    si->wi = ray.d;
    si->uv = uv;
    si->surface_info = data->surface_info;
}

// Plane -------------------------------------------------------------------------------
static __forceinline__ __device__ bool hitPlane(const PlaneData* plane_data, const float3& o, const float3& v, const float tmin, const float tmax, SurfaceInteraction& si)
{
    const float2 min = plane_data->min;
    const float2 max = plane_data->max;
    
    const float t = -o.y / v.y;
    const float x = o.x + t * v.x;
    const float z = o.z + t * v.z;

    if (min.x < x && x < max.x && min.y < z && z < max.y && tmin < t && t < tmax)
    {
        si.uv = make_float2((x - min.x) / (max.x - min.x), (z - min.y) / max.y - min.y);
        si.n = make_float3(0, 1, 0);
        si.t = t;
        si.p = o + t*v;
        return true;
    }
    return false;
}

extern "C" __device__ void __intersection__plane()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const PlaneData* plane_data = reinterpret_cast<PlaneData*>(data->shape_data);

    const float2 min = plane_data->min;
    const float2 max = plane_data->max;

    Ray ray = getLocalRay();

    const float t = -ray.o.y / ray.d.y;

    const float x = ray.o.x + t * ray.d.x;
    const float z = ray.o.z + t * ray.d.z;

    float2 uv = make_float2((x - min.x) / (max.x - min.x), (z - min.y) / (max.y - min.y));

    float3 n = make_float3(0, 1, 0);

    if (min.x < x && x < max.x && min.y < z && z < max.y && ray.tmin < t && t < ray.tmax)
        optixReportIntersection(t, 0, float3_as_ints(n), float2_as_ints(uv));
}

extern "C" __device__ void __closesthit__plane()
{
    HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    
    Ray ray = getWorldRay();

    float3 local_n = make_float3(0, 1, 0);
    float3 world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);
    float2 uv = getFloat2FromAttribute<3>();

    SurfaceInteraction* si = getSurfaceInteraction();

    si->p = ray.at(ray.tmax);
    si->n = world_n;
    si->t = ray.tmax;
    si->wi = ray.d;
    si->uv = uv;
    si->surface_info = data->surface_info;
}

extern "C" __device__ float __continuation_callable__pdf_plane(AreaEmitterInfo area_info, const float3 & origin, const float3 & direction)
{
    const PlaneData* plane_data = reinterpret_cast<PlaneData*>(area_info.shape_data);

    SurfaceInteraction si;
    const float3 local_o = area_info.worldToObj.pointMul(origin);
    const float3 local_d = area_info.worldToObj.vectorMul(direction);

    if (!hitPlane(plane_data, local_o, local_d, 0.01f, 1e16f, si))
        return 0.0f;

    const float3 corner0 = area_info.objToWorld.pointMul(make_float3(plane_data->min.x, 0.0f, plane_data->min.y));
    const float3 corner1 = area_info.objToWorld.pointMul(make_float3(plane_data->max.x, 0.0f, plane_data->min.y));
    const float3 corner2 = area_info.objToWorld.pointMul(make_float3(plane_data->min.x, 0.0f, plane_data->max.y));
    si.n = normalize(area_info.objToWorld.vectorMul(si.n));
    const float area = length(cross(corner1 - corner0, corner2 - corner0));
    const float distance_squared = si.t * si.t;
    const float cosine = fabs(dot(si.n, direction));
    if (cosine < math::eps)
        return 0.0f;
    return distance_squared / (cosine * area);
}

// グローバル空間における si.p -> 光源上の点 のベクトルを返す
extern "C" __device__ float3 __direct_callable__rnd_sample_plane(AreaEmitterInfo area_info, SurfaceInteraction * si)
{
    const PlaneData* plane_data = reinterpret_cast<PlaneData*>(area_info.shape_data);
    // サーフェスの原点をローカル空間に移す
    const float3 local_p = area_info.worldToObj.pointMul(si->p);
    unsigned int seed = si->seed;
    // 平面光源上のランダムな点を取得
    const float3 rnd_p = make_float3(rnd(seed, plane_data->min.x, plane_data->max.x), 0.0f, rnd(seed, plane_data->min.y, plane_data->max.y));
    float3 to_light = rnd_p - local_p;
    to_light = area_info.objToWorld.vectorMul(to_light);
    si->seed = seed;
    return to_light;
}

// Sphere -------------------------------------------------------------------------------
static __forceinline__ __device__ float2 getSphereUV(const float3& p) {
    float phi = atan2(p.z, p.x);
    if (phi < 0) phi += 2.0f * math::pi;
    float theta = acos(p.y);
    float u = phi / (2.0f * math::pi);
    float v = theta / math::pi;
    return make_float2(u, v);
}

static __forceinline__ __device__ bool hitSphere(const SphereData* sphere_data, const float3& o, const float3& v, const float tmin, const float tmax, SurfaceInteraction& si)
{
    const float3 center = sphere_data->center;
    const float radius = sphere_data->radius;

    const float3 oc = o - center;
    const float a = dot(v, v);
    const float half_b = dot(oc, v);
    const float c = dot(oc, oc) - radius * radius;
    const float discriminant = half_b * half_b - a * c;

    if (discriminant <= 0.0f) return false;

    const float sqrtd = sqrtf(discriminant);

    float t = (-half_b - sqrtd) / a;
    if (t < tmin || tmax < t)
    {
        t = (-half_b + sqrtd) / a;
        if (t < tmin || tmax < t)
            return false;
    }

    si.t = t;
    si.p = o + t * v;
    si.n = si.p / radius;
    si.uv = getSphereUV(si.n);
    return true;
}

extern "C" __device__ void __intersection__sphere() {
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const SphereData* sphere_data = reinterpret_cast<SphereData*>(data->shape_data);

    const float3 center = sphere_data->center;
    const float radius = sphere_data->radius;

    Ray ray = getLocalRay();

    const float3 oc = ray.o - center;
    const float a = dot(ray.d, ray.d);
    const float half_b = dot(oc, ray.d);
    const float c = dot(oc, oc) - radius * radius;
    const float discriminant = half_b * half_b - a * c;

    if (discriminant > 0.0f) {
        float sqrtd = sqrtf(discriminant);
        float t1 = (-half_b - sqrtd) / a;
        bool check_second = true;
        if (t1 > ray.tmin && t1 < ray.tmax) {
            float3 normal = normalize((ray.at(t1) - center) / radius);
            const float2 uv = getSphereUV(normal);
            check_second = false;
            optixReportIntersection(t1, 0, float3_as_ints(normal), float2_as_ints(uv));
        }

        if (check_second) {
            float t2 = (-half_b + sqrtd) / a;
            if (t2 > ray.tmin && t2 < ray.tmax) {
                float3 normal = normalize((ray.at(t2) - center) / radius);
                const float2 uv = getSphereUV(normal);
                optixReportIntersection(t2, 0, float3_as_ints(normal), float2_as_ints(uv));
            }
        }
    }
}

extern "C" __device__ void __closesthit__sphere() {
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const SphereData* sphere_data = reinterpret_cast<SphereData*>(data->shape_data);

    Ray ray = getWorldRay();

    float3 local_n = getFloat3FromAttribute<0>();
    float2 uv = getFloat2FromAttribute<3>();
    float3 world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->n = world_n;
    si->t = ray.tmax;
    si->wi = ray.d;
    si->uv = uv;
    si->surface_info = data->surface_info;

    // Calculate partial derivative on texture coordinates
    float phi = atan2(local_n.z, local_n.x);
    if (phi < 0) phi += 2.0f * math::pi;
    const float theta = acos(local_n.y);
    const float u = phi / (2.0f * math::pi);
    const float v = theta / math::pi;
    const float3 dpdu = make_float3(-math::two_pi * local_n.z, 0, math::two_pi * local_n.x);
    const float3 dpdv = math::pi * make_float3(local_n.y * cos(phi), -sin(theta), local_n.y * sin(phi));
    si->dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(dpdu));
    si->dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(dpdv));
}

extern "C" __device__ float __continuation_callable__pdf_sphere(AreaEmitterInfo area_info, const float3 & origin, const float3 & direction)
{
    const SphereData* sphere_data = reinterpret_cast<SphereData*>(area_info.shape_data);
    SurfaceInteraction si;
    const float3 local_o = area_info.worldToObj.pointMul(origin);
    const float3 local_d = area_info.worldToObj.vectorMul(direction);
    
    if (!hitSphere(sphere_data, local_o, local_d, 0.01f, 1e16f, si))
        return 0.0f;

    const float3 center = sphere_data->center;
    const float radius = sphere_data->radius;
    const float cos_theta_max = sqrtf(1.0f - radius * radius / math::sqr(length(center - local_o)));
    const float solid_angle = 2.0f * math::pi * (1.0f - cos_theta_max);
    return 1.0f / solid_angle;
}

extern "C" __device__ float3 __direct_callable__rnd_sample_sphere(AreaEmitterInfo area_info, SurfaceInteraction* si)
{
    const SphereData* sphere_data = reinterpret_cast<SphereData*>(area_info.shape_data);
    const float3 center = sphere_data->center;
    const float3 local_o = area_info.worldToObj.pointMul(si->p);
    const float3 oc = center - local_o;
    float distance_squared = dot(oc, oc);
    Onb onb(normalize(oc));
    unsigned int seed = si->seed;
    float3 to_light = randomSampleToSphere(seed, sphere_data->radius, distance_squared);
    onb.inverseTransform(to_light);
    si->seed = seed;
    return normalize(area_info.objToWorld.vectorMul(to_light));
}

// Triangle mesh -------------------------------------------------------------------------------
extern "C" __device__ void __closesthit__mesh()
{
    HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const MeshData* mesh_data = reinterpret_cast<MeshData*>(data->shape_data);

    Ray ray = getWorldRay();
    
    const int prim_id = optixGetPrimitiveIndex();
    const Face face = mesh_data->faces[prim_id];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    const float3 p0 = mesh_data->vertices[face.vertex_id.x];
    const float3 p1 = mesh_data->vertices[face.vertex_id.y];
    const float3 p2 = mesh_data->vertices[face.vertex_id.z];

    const float2 texcoord0 = mesh_data->texcoords[face.texcoord_id.x];
    const float2 texcoord1 = mesh_data->texcoords[face.texcoord_id.y];
    const float2 texcoord2 = mesh_data->texcoords[face.texcoord_id.z];
    const float2 texcoords = (1-u-v)*texcoord0 + u*texcoord1 + v*texcoord2;

    float3 n0 = mesh_data->normals[face.normal_id.x];
	float3 n1 = mesh_data->normals[face.normal_id.y];
	float3 n2 = mesh_data->normals[face.normal_id.z];

    // Linear interpolation of normal by barycentric coordinates.
    float3 local_n = (1.0f-u-v)*n0 + u*n1 + v*n2;
    float3 world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->n = world_n;
    si->t = ray.tmax;
    si->wi = ray.d;
    si->uv = texcoords;
    si->surface_info = data->surface_info;

    // Calculate partial derivative on texture coordinates
    float3 dpdu, dpdv;
    const float2 duv02 = texcoord0 - texcoord2, duv12 = texcoord1 - texcoord2;
    const float3 dp02 = p0 - p2, dp12 = p1 - p2;
    const float D = duv02.x * duv12.y - duv02.y * duv12.x;
    bool degenerateUV = abs(D) < 1e-8f;
    if (!degenerateUV)
    {
        const float invD = 1.0f / D;
        dpdu = (duv12.y * dp02 - duv02.y * dp12) * invD;
        dpdv = (-duv12.x * dp02 + duv02.x * dp12) * invD;
    }
    if (degenerateUV || length(cross(dpdu, dpdv)) == 0.0f)
    {
        const float3 n = normalize(cross(p2 - p0, p1 - p0));
        Onb onb(n);
        dpdu = onb.tangent;
        dpdv = onb.bitangent;
    }
    si->dpdu = normalize(optixTransformNormalFromObjectToWorldSpace(dpdu));
    si->dpdv = normalize(optixTransformNormalFromObjectToWorldSpace(dpdv));
}

extern "C" __device__ void __anyhit__alpha_discard()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    if (!data->alpha_texture.data) return;

    const uint32_t hit_kind = optixGetHitKind();

    const int prim_id = optixGetPrimitiveIndex();

    float2 texcoord;
    if (optixIsTriangleHit())
    {
        const MeshData* mesh = reinterpret_cast<MeshData*>(data->shape_data);
        const Face face = mesh->faces[prim_id];
        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;
        const float2 texcoord0 = mesh->texcoords[face.texcoord_id.x];
        const float2 texcoord1 = mesh->texcoords[face.texcoord_id.y];
        const float2 texcoord2 = mesh->texcoords[face.texcoord_id.z];

        texcoord = (1.0f - u - v) * texcoord0 + u * texcoord1 + v * texcoord2;
    }
    else
    {
        texcoord = getFloat2FromAttribute<3>();
    }

    const float4 alpha = optixDirectCall<float4, const float2&, void*>(
        data->alpha_texture.prg_id, texcoord, data->alpha_texture.data
        );

    if (alpha.w == 0.0f) optixIgnoreIntersection();
}