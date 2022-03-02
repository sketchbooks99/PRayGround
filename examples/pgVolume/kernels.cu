#include <prayground/prayground.h>
#include "params.h"

extern "C" { __constant__ LaunchParams params; }

using SurfaceInteraction = SurfaceInteraction_<float3>;

static __forceinline__ __device__ SurfaceInteraction* getSurfaceInteraction()
{
    const unsigned int u0 = getPayload<0>();
    const unsigned int u1 = getPayload<1>();
    return reinterpret_cast<SurfaceInteraction*>(unpackPointer(u0, u1));
}

static __forceinline__ __device__ void trace(
    OptixTraversableHandle handle,
    const float3& ro, const float3& rd,
    float tmin, float tmax,
    unsigned int ray_type,
    SurfaceInteraction* si
)
{
    unsigned int u0, u1;
    packPointer(si, u0, u1);
    optixTrace(
        handle,
        ro, rd,
        tmin, tmax, 0.0f,
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        ray_type,
        1,
        ray_type,
        u0, u1
    );
}

/* Raygen function */
static __forceinline__ __device__ void getCameraRay(
    const Camera::Data& camera,
    const float& x, const float& y,
    float3& ro, float3& rd)
{
    rd = normalize(x * camera.U + y * camera.V + camera.W);
    ro = camera.origin;
}

extern "C" __device__ void __raygen__medium()
{
    const RaygenData* raygen = reinterpret_cast<RaygenData*>(optixGetSbtDataPointer());

    const int subframe_index = params.frame;
    const uint3 idx = optixGetLaunchIndex();
    unsigned int seed = tea<4>(idx.x * params.width + idx.y, subframe_index);

    float3 result = make_float3(0.0f);
    float3 normal = make_float3(0.0f);
    float p_depth = 0.0f;
    float3 albedo = make_float3(0.0f);

    int i = params.samples_per_launch;

    do
    {
        const float2 subpixel_jitter = make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f);

        const float2 d = 2.0f * make_float2(
            (static_cast<float>(idx.x) + subpixel_jitter.x) / static_cast<float>(params.width),
            (static_cast<float>(idx.y) + subpixel_jitter.y) / static_cast<float>(params.height)
        ) - 1.0f;

        float3 ro, rd;
        getCameraRay(raygen->camera, d.x, d.y, ro, rd);

        float3 throughput = make_float3(1.0f);

        SurfaceInteraction si;
        si.seed = seed;
        si.emission = make_float3(0.0f);
        si.albedo = make_float3(0.0f);
        si.trace_terminate = false;
        si.radiance_evaled = false;

        int depth = 0;
        for (;; ) {

            if (depth >= params.max_depth)
                break;

            trace(params.handle, ro, rd, 0.01f, 1e16f, 0, &si);

            if (si.trace_terminate) {
                result += si.emission * throughput;
                break;
            }

            // Get emission from area emitter
            if (si.surface_info.type == SurfaceType::AreaEmitter)
            {
                // Evaluating emission from emitter
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.bsdf_id,
                    &si,
                    si.surface_info.data
                    );
                result += si.emission * throughput;

                if (si.trace_terminate)
                    break;
            }
            // Specular sampling
            else if (+(si.surface_info.type & SurfaceType::Delta))
            {
                // Sampling scattered direction
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.sample_id,
                    &si,
                    si.surface_info.data
                    );

                // Evaluate bsdf
                float3 bsdf_val = optixContinuationCall<float3, SurfaceInteraction*, void*>(
                    si.surface_info.bsdf_id,
                    &si,
                    si.surface_info.data
                    );
                throughput *= bsdf_val;
            }
            // Rough surface sampling with applying MIS
            else if (+(si.surface_info.type & (SurfaceType::Rough | SurfaceType::Diffuse)))
            {
                // Importance sampling according to the BSDF
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.sample_id,
                    &si,
                    si.surface_info.data
                    );

                // Evaluate PDF depends on BSDF
                float bsdf_pdf = optixDirectCall<float, SurfaceInteraction*, void*>(
                    si.surface_info.pdf_id,
                    &si,
                    si.surface_info.data
                    );

                // Evaluate BSDF
                float3 bsdf_val = optixContinuationCall<float3, SurfaceInteraction*, void*>(
                    si.surface_info.bsdf_id,
                    &si,
                    si.surface_info.data
                    );

                throughput *= bsdf_val / bsdf_pdf;
            }

            ro = si.p;
            rd = si.wo;

            ++depth;
        }
    } while (--i);

    const uint3 launch_index = optixGetLaunchIndex();
    const unsigned int image_index = launch_index.y * params.width + launch_index.x;

    if (result.x != result.x) result.x = 0.0f;
    if (result.y != result.y) result.y = 0.0f;
    if (result.z != result.z) result.z = 0.0f;

    float3 accum_color = result / static_cast<float>(params.samples_per_launch);

    if (subframe_index > 0)
    {
        const float a = 1.0f / static_cast<float>(subframe_index + 1);
        const float3 accum_color_prev = make_float3(params.accum_buffer[image_index]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }
    params.accum_buffer[image_index] = make_float4(accum_color, 1.0f);
    uchar3 color = make_color(accum_color);
    params.result_buffer[image_index] = make_uchar4(color.x, color.y, color.z, 255);
}

/* Material functions */

/* Hitgroup functions */
extern "C" __device__ void __intersection__plane()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const Plane::Data* plane = reinterpret_cast<Plane::Data*>(data->shape_data);

    const float2 min = plane->min;
    const float2 max = plane->max;

    Ray ray = getLocalRay();

    const float t = -ray.o.y / ray.d.y;

    const float x = ray.o.x + t * ray.d.x;
    const float z = ray.o.z + t * ray.d.z;

    float2 uv = make_float2((x - min.x) / (max.x - min.x), (z - min.y) / (max.y - min.y));

    if (min.x < x && x < max.x && min.y < z && z < max.y && ray.tmin < t && t < ray.tmax)
        optixReportIntersection(t, 0, float2_as_ints(uv));
}

extern "C" __device__ void __closesthit__plane()
{
    HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());

    Ray ray = getWorldRay();

    float3 local_n = make_float3(0, 1, 0);
    const float3 world_n = normalize(optixTransformNormalFromObjectToWorldSpace(local_n));
    const float2 uv = getFloat2FromAttribute<0>();

    SurfaceInteraction* si = getSurfaceInteraction();

    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->uv = uv;
    si->surface_info = data->surface_info;
    si->shading.dpdu = optixTransformNormalFromObjectToWorldSpace(make_float3(1.0f, 0.0f, 0.0f));
    si->shading.dpdv = optixTransformNormalFromObjectToWorldSpace(make_float3(0.0f, 0.0f, 1.0f));
}

extern "C" __device__ void __intersection__grid()
{
    const HitgroupData* data = reinterpret_cast<const HitgroupData*>(optixGetSbtDataPointer());
    const GridMedium::Data* grid = reinterpret_cast<const GridMedium::Data*>(data->shape_data);
    const nanovdb::FloatGrid* density = reinterpret_cast<const nanovdb::FloatGrid*>(grid->density);
    assert(density);

    Ray ray = getLocalRay;

    auto bbox = density->indexBBox();
    float t0 = ray.tmin;
    float t1 = ray.tmax;
    auto iRay = nanovdb::Ray<float>(reinterpret_cast<const nanovdb::Vec3f&>(ray.o),
        reinterpret_cast<const nanovdb::Vec3f&>(ray.d), t0, t1);

    if (iRay.intersects(bbox, t0, t1))
    {
        optixSetPayload_2(__float_as_int(t1));
        optixReportIntersection(fmaxf(t0, ray.tmin), 0);
    }
}

extern "C" __device__ void __closesthit__grid()
{
    const HitgroupData* data = reinterpret_cast<const HitgroupData*>(optixGetSbtDataPointer());
    const GridMedium::Data* grid = reinterpret_cast<const GridMedium::Data*>(data->shape_data);
    const nanovdb::FloatGrid* density = reinterpret_cast<const nanovdb::FloatGrid*>(grid->density);
    const auto& tree = density->tree();
    auto acc = tree.getAccessor();

    Ray ray = getWorldRay();
    const float t0 = optixGetRayTmax();
    const float t1 = __int_as_float(optixGetPayload_2());
}

/* Texture functions */
extern "C" __device__ float3 __direct_callable__bitmap(const float2& uv, void* tex_data)
{
    const BitmapTexture::Data* image = reinterpret_cast<BitmapTexture::Data*>(tex_data);
    float4 c = tex2D<float4>(image->texture, uv.x, uv.y);
    return make_float3(c.x, c.y, c.z);
}

extern "C" __device__ float3 __direct_callable__constant(const float2& uv, void* tex_data)
{
    const ConstantTexture::Data* constant = reinterpret_cast<ConstantTexture::Data*>(tex_data);
    return constant->color;
}

extern "C" __device__ float3 __direct_callable__checker(const float2& uv, void* tex_data)
{
    const CheckerTexture::Data* checker = reinterpret_cast<CheckerTexture::Data*>(tex_data);
    const bool is_odd = sinf(uv.x * math::pi * checker->scale) * sinf(uv.y * math::pi * checker->scale);
    return is_odd ? checker->color1 : checker->color2;
}



