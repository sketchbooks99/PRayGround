#pragma once 

#include <prayground/shape/box.h>
#include <prayground/core/ray.h>

using namespace prayground;

static INLINE DEVICE float2 getBoxUV(const float3& p, const float3& min, const float3& max, const int axis)
{
    float2 uv;
    int u_axis = (axis + 1) % 3;
    int v_axis = (axis + 2) % 3;

    // axisがYの時は (u: Z, v: X) -> (u: X, v: Z)へ順番を変える
    if (axis == 1) swap(u_axis, v_axis);

    uv.x = getByIndex(p, u_axis) - getByIndex(min, u_axis) / (getByIndex(max, u_axis) - getByIndex(min, u_axis));
    uv.y = getByIndex(p, v_axis) - getByIndex(min, v_axis) / (getByIndex(max, v_axis) - getByIndex(min, v_axis));

    return clamp(uv, 0.0f, 1.0f);
}

extern "C" __device__ void __intersection__box()
{
    const HitGroupData* data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
    const Box::Data* box_data = reinterpret_cast<Box::Data*>(data->shape_data);

    float3 min = box_data->min;
    float3 max = box_data->max;

    Ray ray = getLocalRay();

    float tmin = ray.tmin, tmax = ray.tmax;
    int min_axis = -1, max_axis = -1;

    for (int i = 0; i < 3; i++)
    {
        float t0 = fmin((getByIndex(min, i) - getByIndex(ray.o, i)) / getByIndex(ray.d, i),
                        (getByIndex(max, i) - getByIndex(ray.o, i)) / getByIndex(ray.d, i));
        float t1 = fmax((getByIndex(min, i) - getByIndex(ray.o, i)) / getByIndex(ray.d, i),
                        (getByIndex(max, i) - getByIndex(ray.o, i)) / getByIndex(ray.d, i));

        tmin = fmax(t0, tmin);
        tmax = fmin(t1, tmax);

        min_axis += (int)(t0 > tmin);
        max_axis += (int)(t1 < tmax);

        if (tmax < tmin)
            return;
    }

    float3 center = (min + max) / 2.0f;
    if (ray.tmin < tmin && tmin < ray.tmax)
    {
        float3 p = ray.at(tmin);
        float3 center_axis = p;
        setByIndex(center_axis, min_axis, getByIndex(center, min_axis));
        float3 normal = normalize(p - center_axis);
        float2 uv = getBoxUV(p, min, max, min_axis);
        optixReportIntersection(tmin, 0, float3_as_ints(normal), float2_as_ints(uv));
        return;
    }

    if (ray.tmin < tmax && tmax < ray.tmax)
    {
        float3 p = ray.at(tmax);
        float3 center_axis = p;
        setByIndex(center_axis, max_axis, getByIndex(center, max_axis));
        float3 normal = normalize(p - center_axis);
        float2 uv = getBoxUV(p, min, max, max_axis);
        optixReportIntersection(tmax, 0, float3_as_ints(normal), float2_as_ints(uv));
        return;
    }
}

extern "C" __device__ void __closesthit__box()
{
    HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());

    Ray ray = getWorldRay();

    float3 local_n = make_float3(
        int_as_float( optixGetAttribute_0() ),
        int_as_float( optixGetAttribute_1() ), 
        int_as_float( optixGetAttribute_2() )
    );
    float2 uv = make_float2(
        int_as_float( optixGetAttribute_3() ),
        int_as_float( optixGetAttribute_4() )
    );
    float3 world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);

    SurfaceInteraction* si = getSurfaceInteraction();

    si->p = ray.at(ray.tmax);
    si->n = world_n;
    si->t = ray.tmax;
    si->wi = ray.d;
    si->uv = uv;
    si->surface_info = data->surface_info;
}