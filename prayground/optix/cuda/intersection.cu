#include "device_util.cuh"
#include <prayground/shape/plane.h>
#include <prayground/shape/trianglemesh.h>
#include <prayground/shape/sphere.h>
#include <prayground/shape/cylinder.h>
#include <prayground/core/ray.h>
#include <prayground/core/onb.h>
#include <prayground/core/bsdf.h>
#include <prayground/optix/sbt.h>

using namespace prayground;

/// @todo Implement dndu/dndv

// Plane
extern "C" __device__ void __intersection__plane()
{
    const auto* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
    const int prim_id = optixGetPrimitiveIndex();
    const Plane::Data plane = reinterpret_cast<Plane::Data*>(data->shape_data)[prim_id];

    const Vec2f min = plane.min;
    const Vec2f max = plane.max;

    Ray ray = getLocalRay();

    const float t = -ray.o.y() / ray.d.y();

    const float x = ray.o.x() + t * ray.d.x();
    const float z = ray.o.z() + t * ray.d.z();

    if (min.x() < x && x < max.x() && min.y() < z && z < max.y() && ray.tmin < t && t < ray.tmax)
    {
        Shading shading;
        shading.n = Vec3f(0, 1, 0);
        shading.uv = Vec2f((x - min.x()) / (max.x() - min.x()), (z - min.y()) / (max.y() - min.y()));
        shading.dpdu = Vec3f(1, 0, 0);
        shading.dpdv = Vec3f(0, 0, 1);

        uint32_t a0, a1;
        packPointer(&shading, a0, a1);

        optixReportIntersection(t, 0, a0, a1);
    }
}

// Sphere
static __forceinline__ __device__ void calcSphereShading(
    const Sphere::Data& sphere, const Vec3f& p, Shading& shading) 
{
    shading.n = (p - sphere.center) / sphere.radius;

    const Vec3f P = shading.n;

    float phi = atan2(P.z(), P.x());
    if (phi < 0) phi += math::two_pi;
    float theta = asin(P.y());
    float u = 1.0f - (phi + math::pi) / (2.0f * math::pi);
    float v = 1.0f - (theta + math::pi / 2.0f) / math::pi;

    shading.uv = Vec2f(u, v);
    shading.dpdu = Vec3f(-math::two_pi * P.z(), 0, math::two_pi * P.x());
    shading.dpdv = math::pi * Vec3f(P.y() * cosf(phi), -sinf(theta), P.y() * sinf(phi));
}

extern "C" __device__ void __intersection__sphere() {
    const auto* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
    const int prim_id = optixGetPrimitiveIndex();
    const Sphere::Data sphere = reinterpret_cast<Sphere::Data*>(data->shape_data)[prim_id];

    const Vec3f center = sphere.center;
    const float radius = sphere.radius;

    Ray ray = getLocalRay();

    const Vec3f oc = ray.o - center;
    const float a = dot(ray.d, ray.d);
    const float half_b = dot(oc, ray.d);
    const float c = dot(oc, oc) - radius * radius;
    const float discriminant = half_b * half_b - a * c;

    if (discriminant > 0.0f) {
        float sqrtd = sqrtf(discriminant);
        float t = (-half_b - sqrtd) / a;
        bool check_second = true;

        Shading shading;
        uint32_t a0, a1;

        if (t > ray.tmin && t < ray.tmax) {
            calcSphereShading(sphere, ray.at(t), shading);
            packPointer(&shading, a0, a1);
            optixReportIntersection(t, 0, a0, a1);
            check_second = false;
        }

        if (check_second) {
            t = (-half_b + sqrtd) / a;
            if (t > ray.tmin && t < ray.tmax) 
            {
                calcSphereShading(sphere, ray.at(t), shading);
                packPointer(&shading, a0, a1);
                optixReportIntersection(t, 0, a0, a1);
            }
        }
    }
}

// Cylinder 
static __forceinline__ __device__ void calcCylinderShading(
    const Cylinder::Data& cylinder, const Vec3f& p, const bool hit_disk, Shading& shading)
{                        
    if (hit_disk)
    {
        const float r = sqrtf(p.x()*p.x() + p.z()*p.z()) / cylinder.radius;
        const float theta = atan2(p.z(), p.x());
        float u = 1.0f - (theta + math::pi/2.0f) / math::pi;
        shading.n = normalize(p - Vec3f(p.x(), 0, p.z()));
        shading.uv = Vec2f(u, r);

        const float r_hit = sqrtf(p.x()*p.x() + p.z()*p.z());
        shading.dpdu = Vec3f(-math::two_pi * p.y(), 0, math::two_pi * p.z());
        shading.dpdv = Vec3f(p.x(), 0, p.z()) * cylinder.radius / r_hit;
    } 
    else
    {
        float phi = atan2(p.z(), p.x());
        if (phi < 0.0f) phi += math::two_pi;
        const float u = phi / math::two_pi;
        const float v = (p.y() + height / 2.0f) / height;
        shading.n = normalize(p - Vec3f(0, p.y(), 0));
        shading.uv = Vec2f(u, v);
        shading.dpdu = Vec3f(-math::two_pi * p.z(), 0, math::two_pi * p.x());
        shading.dpdv = Vec3f(0, cylinder.height, 0);
    }
}

extern "C" __device__ void __intersection__cylinder()
{
    const auto* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
    const int prim_id = optixGetPrimitiveIndex();
    const Cylinder::Data cylinder = reinterpret_cast<Cylinder::Data*>(data->shape_data)[prim_id];

    const float radius = cylinder.radius;
    const float height = cylinder.height;

    Ray ray = getLocalRay();
    
    const float a = dot(ray.d, ray.d) - ray.d.y() * ray.d.y();
    const float half_b = (ray.o.x() * ray.d.x() + ray.o.z() * ray.d.z());
    const float c = dot(ray.o, ray.o) - ray.o.y() * ray.o.y() - radius*radius;
    const float discriminant = half_b*half_b - a*c;

    if (discriminant > 0.0f)
    {
        const float sqrtd = sqrtf(discriminant);
        const float side_t1 = (-half_b - sqrtd) / a;
        const float side_t2 = (-half_b + sqrtd) / a;

        const float side_tmin = fmin( side_t1, side_t2 );
        const float side_tmax = fmax( side_t1, side_t2 );

        if ( side_tmin > ray.tmax || side_tmax < ray.tmin )
            return;

        const float upper = height / 2.0f;
        const float lower = -height / 2.0f;
        const float y_tmin = fmin( (lower - ray.o.y()) / ray.d.y(), (upper - ray.o.y()) / ray.d.y() );
        const float y_tmax = fmax( (lower - ray.o.y()) / ray.d.y(), (upper - ray.o.y()) / ray.d.y() );

        float t1 = fmax(y_tmin, side_tmin);
        float t2 = fmin(y_tmax, side_tmax);
        if (t1 > t2 || (t2 < ray.tmin) || (t1 > ray.tmax))
            return;
        
        bool check_second = true;
        Shading shading;
        uint32_t a0, a1;
        if (ray.tmin < t1 && t1 < ray.tmax)
        {
            const Vec3f p = ray.at(t1);
            bool hit_disk = y_tmin > side_tmin;
            calcCylinderShading(cylinder, p, hit_disk, shading);
            packPointer(&shading, a0, a1);
            optixReportIntersection(t1, 0, a0, a1);
            check_second = false;
        }
        
        if (check_second)
        {
            if (ray.tmin < t2 && t2 < ray.tmax)
            {
                Vec3f p = ray.at(t2);
                bool hit_disk = y_tmax < side_tmax;
                calcCylinderShading(cylinder, p, hit_disk, shading);
                packPointer(&shading, a0, a1);
                optixReportIntersection(t2, 0, a0, a1);
            }
        }
    }
}

// Box
static __forceinline__ __device__ void calcBoxShading(
    const Box::Data& box, const Vec3f& p, const int axis, Shading& shading)
{
    const Vec3f min = box.min;
    const Vec3f max = box.max;
    const Vec3f center = (box.min + box.max) / 2.0f;

    Vec3f center_axis = p;
    center_axis[axis] = center[axis];
    shading.n = normalize(p - center_axis);

    int u_axis = (axis + 1) % 3;
    int v_axis = (axis + 2) % 3;

    // axisがYの時は (u: Z, v: X) -> (u: X, v: Z)へ順番を変える
    if (axis == 1) swap(u_axis, v_axis);

    shading.uv = Vec2f((p[u_axis] - min[u_axis]) / (max[u_axis] - min[u_axis]), (p[v_axis] - min[v_axis]) / (max[v_axis] - min[v_axis]));

    // x 
    switch( axis )
    {
        case 0: // X
            shading.dpdu = Vec3f(0, 0, 1);
            shading.dpdv = Vec3f(0, 1, 0);
            break;
        case 1: // Y
            shading.dpdu = Vec3f(1, 0, 0);
            shading.dpdv = Vec3f(0, 0, 1);
            break;
        case 2: // Z
            shading.dpdu = Vec3f(1, 0, 0);
            shading.dpdv = Vec3f(0, 1, 0);
            break;
    }
}

extern "C" __device__ void __intersection__box()
{
    const auto* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
    const int prim_id = optixGetPrimitiveIndex();
    Box::Data box = reinterpret_cast<Box::Data*>(data->shape_data)[prim_id];

    Ray ray = getLocalRay();

    const Vec3f min = box.min;
    const Vec3f max = box.max;

    float tmin = ray.tmin;
    float tmax = ray.tmax;

    int min_axis = -1, max_axis = -1;

    for (int i = 0; i < 3; i++)
    {
        float t0, t1;
        if (ray.d[i] == 0.0f)
        {
            t0 = fminf(min[i] - ray.o[i], max[i] - ray.o[i]);
            t1 = fmaxf(min[i] - ray.o[i], max[i] - ray.o[i]);
        }
        else 
        {
            t0 = fminf((min[i] - ray.o[i]) / ray.d[i], (max[i] - ray.o[i]) / ray.d[i]);
            t1 = fmaxf((min[i] - ray.o[i]) / ray.d[i], (max[i] - ray.o[i]) / ray.d[i]);
        }
        min_axis = t0 > tmin ? i : min_axis;
        max_axis = t1 < tmax ? i : max_axis;

        tmin = fmaxf(t0, tmin);
        tmax = fminf(t1, tmax);

        if (tmax < tmin) return;
    }

    Vec3f center = (min + max) / 2.0f;

    Shading shading;
    uint32_t a0, a1;
    bool check_second = true;

    if ((ray.tmin < tmin && tmin < ray.tmax) && (-1 < min_axis && min_axis < 3))
    {
        Vec3f p = ray.at(tmin);
        calcBoxShading(box, p, min_axis, shading);
        packPointer(&shading, a0, a1);
        check_second = false;
        optixReportIntersection(tmin, 0, a0, a1);
    }

    if ((ray.tmin < tmax && tmax < ray.tmax) && (-1 < max_axis && max_axis < 3) && check_second)
    {
        Vec3f p = ray.at(tmax);
        calcBoxShading(box, p, max_axis, shading);
        packPointer(&shading, a0, a1);
        optixReportIntersection(tmin, 0, a0, a1);
    }
}