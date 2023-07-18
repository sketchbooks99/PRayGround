#include <prayground/prayground.h>
#include "params.h"

extern "C" { __constant__ LaunchParams params; } 

using SurfaceInteraction = SurfaceInteraction_<Vec3f>;

static INLINE DEVICE SurfaceInteraction* getSurfaceInteraction()
{
    const uint32_t u0 = getPayload<0>();
    const uint32_t u1 = getPayload<1>();
    return reinterpret_cast<SurfaceInteraction*>(unpackPointer(u0, u1));
}

static INLINE DEVICE Vec3f reinhardTonemap(const Vec3f& c, const float white)
{
    const float l = luminance(c);
    return c * (1.0f + l / (white * white)) / (1.0f + l);
}

static INLINE DEVICE void trace(
    OptixTraversableHandle handle, const Vec3f& ro, const Vec3f& rd, 
    const float tmin, const float tmax, SurfaceInteraction* si)
{
    uint32_t u0, u1;
    packPointer(si, u0, u1);
    optixTrace(
        handle, ro, rd, tmin, tmax, 0.0f, OptixVisibilityMask(1), 
        OPTIX_RAY_FLAG_NONE, 
        (uint32_t)RayType::Radiance, (uint32_t)RayType::Count, (uint32_t)RayType::Radiance,
        u0, u1);
}

static INLINE DEVICE bool traceShadow(
    OptixTraversableHandle handle, const Vec3f& ro, const Vec3f& rd, 
    const float tmin, const float tmax)
{
    uint32_t hit = 0u;
    optixTrace(
        handle, ro, rd, tmin, tmax, 0.0f, OptixVisibilityMask(1), 
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, 
        (uint32_t)RayType::Shadow, (uint32_t)RayType::Count, (uint32_t)RayType::Shadow,
        hit);
    return static_cast<bool>(hit);
}

static INLINE DEVICE Vec3f randomSampleOnTriangle(const Triangle& t, uint32_t seed)
{
    // Uniform sampling of barycentric coordinates on a triangle
    Vec2f uv = UniformSampler::get2D(seed);
    return t.v0 + (1.0f - uv.x() - uv.y()) + t.v1 * uv.x() + t.v2 * uv.y();
}

static INLINE DEVICE Reservoir reservoirSampling(int32_t num_strategies, SurfaceInteraction* si, uint32_t& seed)
{
    Reservoir r{0, 0, 0, 0};
    for (int32_t i = 0; i < num_strategies; i++)
    {
        // Select a light source
        const int32_t light_idx = rndInt(seed, 0, params.num_lights - 1);
        LightInfo light = params.lights[light_idx];

        // Sample a point on the light source
        const Vec3f p = randomSampleOnTriangle(light.triangle, seed);
        const Vec3f wi = normalize(p - si->p);
        const float d = length(p - si->p);

        // Check visibility
        const bool occluded = traceShadow(params.handle, si->p, wi, 0.001f, d - 0.001f);
        if (occluded)
            continue;
        
        // Evaluate BRDF
        Vec3f brdf = optixContinuationCall<Vec3f, SurfaceInteraction*, void*, const Vec3f&>(
            si->surface_info.callable_id.bsdf, si, si->surface_info.data, p);

        const float nDl = dot(si->shading.n, wi);
        Vec3f LN = light.triangle.n;
        LN = faceforward(LN, wi, LN);
        const float LnDl = dot(LN, wi);
        if (nDl > 0.0f && LnDl > 0.0f)
        {
            const float area = length(cross(light.triangle.v1 - light.triangle.v0, light.triangle.v2 - light.triangle.v0)) * 0.5f;
            const float weight = LnDl * nDl * area / (math::pi * d * d);
            r.update(light_idx, weight, seed);
            r.W = weight;
            r.p = p;
        }
    }
    return r;
}

extern "C" DEVICE void __raygen__restir()
{
    const auto* rg = reinterpret_cast<pgRaygenData<Camera>*>(optixGetSbtDataPointer());

    const int32_t frame = params.frame;
    const Vec3ui idx(optixGetLaunchIndex());
    uint32_t seed = tea<4>(idx.x() * params.width + idx.y(), frame);

    printf("idx: %d, %d\n", idx.x(), idx.y());

    Vec3f result(0.0f);

    int32_t samples_per_launch = params.samples_per_launch;

    const int32_t M = 32;

    int32_t depth = 0;
    
    for (auto i = 0; i < samples_per_launch; i++)
    {
        const Vec2f jitter = UniformSampler::get2D(seed);

        // Random sampling on a pixel
        const Vec2f d = (2.0f * (Vec2f(idx.x(), idx.y()) + jitter) / Vec2f(params.width, params.height)) - 1.0f;

        Vec3f ro, rd;
        getCameraRay(rg->camera, d.x(), d.y(), ro, rd);

        Vec3f throughput(1.0f);
        Vec3f radiance(0.0f);

        SurfaceInteraction si;
        si.seed = seed;
        si.emission = Vec3f(0.0f);
        si.albedo = Vec3f(0.0f);
        si.trace_terminate = false;

        int32_t depth = 0;
        // Recursive ray tracing with Weighted Reservoir Sampling
        for (;;)
        {
            trace(params.handle, ro, rd, 0.001f, 1e16f, &si);

            if (si.trace_terminate || depth >= params.max_depth)
            {
                result += si.emission * throughput;
                break;
            }

            if (si.surface_info.type == SurfaceType::AreaEmitter)
            {
                // Evaluate emittance from area light
                const Vec3f emittance = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
                    si.surface_info.callable_id.bsdf, &si, si.surface_info.data);
                result += emittance * throughput;
                if (si.trace_terminate)
                    break;
            }
            else
            {
                Reservoir r = reservoirSampling(M, &si, si.seed);
                
                const Vec3f wi = normalize(r.p - si.p);
                Vec3f brdf = optixContinuationCall<Vec3f, SurfaceInteraction*, void*, const Vec3f&>(
                    si.surface_info.callable_id.bsdf, &si, si.surface_info.data, wi);
                throughput *= brdf;
                
                if (r.W > 0.0f && !isinf(r.W) && !isnan(r.W))
                {
                    printf("r.W: %f\n", r.W);
                    result += r.W * brdf / (1.0f / (float)r.M);
                    break;
                }

                si.trace_terminate = false;
                Vec2f u = UniformSampler::get2D(si.seed);
                Vec3f s = cosineSampleHemisphere(u.x(), u.y());
                Onb onb(si.shading.n);
                onb.inverseTransform(s);
                si.wi = normalize(s);
            }
            ro = si.p;
            rd = si.wi;
            ++depth;
        }
    }

     const uint32_t image_idx = idx.y() * params.width + idx.x();

    // Nan | Inf check
    if (!result.isValid()) 
        result = Vec3f(0.0f);

    Vec3f accum = result / static_cast<float>(samples_per_launch);

    if (frame > 0)
    {
        const float a = 1.0f / static_cast<float>(frame + 1);
        const Vec3f accum_prev = Vec3f(params.accum_buffer[image_idx]);
        accum = lerp(accum_prev, accum, a);
    }

    params.accum_buffer[image_idx] = Vec4f(accum, 1.0f);
    Vec3u color = make_color(reinhardTonemap(accum, params.white));
    params.result_buffer[image_idx] = Vec4u(color, 255);
}

// Miss -------------------------------------------------------------
extern "C" __device__ void __miss__envmap()
{
    const auto* data = (pgMissData*)optixGetSbtDataPointer();
    const auto* env = (EnvironmentEmitter::Data*)data->env_data;
    auto* si = getSurfaceInteraction();

    Ray ray = getWorldRay();

    const float a = dot(ray.d, ray.d);
    const float half_b = dot(ray.o, ray.d);
    const float c = dot(ray.o, ray.o) - 1e8f * 1e8f;
    const float D = half_b * half_b - a * c;

    const float sqrtD = sqrtf(D);
    const float t = (-half_b - sqrtD) / a;

    Vec3f p = normalize(ray.at(t));

    const float phi = atan2(p.z(), p.x());
    const float theta = asin(p.y());
    const float u = 1.0f - (phi + math::pi) / (math::two_pi);
    const float v = 1.0f - (theta + math::pi / 2.0f) / math::pi;
    si->shading.uv = Vec2f(u, v);
    si->trace_terminate = true;
    si->surface_info.type = SurfaceType::Envmap;
    si->emission = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
        env->texture.prg_id, si, env->texture.data);
}

extern "C" __device__ void __miss__shadow()
{
    setPayload<0>(0);
}

// Diffuse surface
extern "C" DEVICE void DC_FUNC(sample_diffuse)(SurfaceInteraction* si, void* data)
{
    const auto* diffuse = reinterpret_cast<const Diffuse::Data*>(data);
    si->wi = pgImportanceSamplingDiffuse(diffuse, si->wo, si->shading, si->seed);
    si->trace_terminate = false;
}

extern "C" DEVICE Vec3f CC_FUNC(brdf_diffuse)(SurfaceInteraction* si, void* data, const Vec3f& wi)
{
    const auto* diffuse = reinterpret_cast<const Diffuse::Data*>(data);
    si->emission = Vec3f(0.0f);
    const Vec3f albedo = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
        diffuse->texture.prg_id, si, diffuse->texture.data);
    si->albedo = albedo;
    return albedo * pgGetDiffuseBRDF(si->shading.n, wi) * math::inv_pi; 
}

extern "C" DEVICE float DC_FUNC(pdf_diffuse)(SurfaceInteraction* si, void* data, const Vec3f& wi)
{
    const auto* diffuse = reinterpret_cast<const Diffuse::Data*>(data);
    return pgGetDiffusePDF(si->shading.n, wi) * math::inv_pi;
}

// Area light
extern "C" DEVICE Vec3f DC_FUNC(area_emitter)(SurfaceInteraction* si, void* data)
{
    const auto* area = reinterpret_cast<AreaEmitter::Data*>(data);
    si->trace_terminate = true;
    float is_emitted = dot(si->wo, si->shading.n) < 0.0f ? 1.0f : 0.0f;
    if (area->twosided)
    {
        is_emitted = 1.0f;
        si->shading.n = faceforward(si->shading.n, -si->wo, si->shading.n);
    }

    const Vec3f base = optixDirectCall<Vec3f, const Vec2f&, void*>(
        area->texture.prg_id, si->shading.uv, area->texture.data);

    si->albedo = base;
    si->emission = base * area->intensity * is_emitted;
    return si->emission;
}

// Shapes
extern "C" __device__ void __closesthit__mesh()
{
    pgHitgroupData* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
    const TriangleMesh::Data* mesh_data = reinterpret_cast<TriangleMesh::Data*>(data->shape_data);

    Ray ray = getWorldRay();

    Shading shading = pgGetMeshShading(mesh_data, optixGetTriangleBarycentrics(), optixGetPrimitiveIndex());

    // Transform shading from object to world space
    shading.n = normalize(optixTransformNormalFromObjectToWorldSpace(shading.n));
    shading.dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(shading.dpdu));
    shading.dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(shading.dpdv));

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->shading = shading;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->surface_info = data->surface_info;
}

extern "C" __device__ void __closesthit__shadow()
{
    setPayload<0>(1);
}

// Textures
extern "C" DEVICE Vec3f DC_FUNC(bitmap)(const Vec2f& uv, void* data)
{
    return pgGetBitmapTextureValue<Vec3f>(uv, data);
}

extern "C" DEVICE Vec3f DC_FUNC(constant)(const Vec2f& uv, void* data)
{
    return pgGetConstantTextureValue<Vec3f>(uv, data);
}

extern "C" DEVICE Vec3f DC_FUNC(checker)(const Vec2f& uv, void* data)
{
    return pgGetCheckerTextureValue<Vec3f>(uv, data);
}