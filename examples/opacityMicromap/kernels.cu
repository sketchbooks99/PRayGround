#include <prayground/prayground.h>
#include "params.h"

extern "C" { __constant__ LaunchParams params; }

using SurfaceInteraction = SurfaceInteraction_<Vec4f>;

struct LightInteraction {
    Vec3f p;
    Vec3f n;
    Vec2f uv;
    float area;
    float pdf;
};

void planeLightSampling(const AreaEmitterInfo& area_emitter, const Vec3f& p, LightInteraction& li, uint32_t& seed);

// raygen
extern "C" GLOBAL void __raygen__pinhole()
{
    const RaygenData* raygen = (RaygenData*)optixGetSbtDataPointer();

    const int frame = params.frame;

    const Vec3ui idx(optixGetLaunchIndex());
    uint32_t seed = tea<4>(idx.y() * params.width + idx.x(), frame);

    Vec3f result(0.0f);
    Vec3f normal(0.0f);

    int i = params.samples_per_launch;

    while (i > 0) {
        const Vec2f jitter = UniformSampler::get2D(seed) - 0.5f;
        const Vec2f d = 2.0f * Vec2f(
            (static_cast<float>(idx.x()) + jitter.x()) / params.width,
            (static_cast<float>(idx.y()) + jitter.y()) / params.height
        ) - 1.0f;

        Vec3f ro, rd;
        getCameraRay(raygen->camera, d.x(), d.y(), ro, rd);

        Vec3f throughput(1.0f);

        SurfaceInteraction si;
        si.seed = seed;
        si.emission = 0.0f;
        si.albedo = 0.0f;
        si.trace_terminate = false;

        int depth = 0;
        for (;;) {
            if (depth >= params.max_depth)
                break;

            uint32_t u0 = 0, u1 = 0, u2 = 0;
            packPointer(&si, u0, u1);
            optixTrace(
                params.handle, ro, rd,
                0.01f, 1e16f, 0.0f,
                OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE,
                0, 2, 0,
                u0, u1, u2
            );

            if (si.trace_terminate) {
                result += Vec3f(si.emission) * throughput;
                break;
            }

            // Get emission from area emitter
            if (si.surface_info.type == SurfaceType::AreaEmitter)
            {
                // Evaluating emission from emitter
                Vec3f emission = optixDirectCall<Vec4f, SurfaceInteraction*, void*>(
                    si.surface_info.callable_id.bsdf, &si, si.surface_info.data);

                result += emission * throughput;
                if (si.trace_terminate)
                    break;
            }
            if (+(si.surface_info.type & SurfaceType::Rough))
            {
                // Sample next direction
                optixDirectCall<void, SurfaceInteraction*, void*>(si.surface_info.callable_id.sample, &si, si.surface_info.data);
                if (params.num_lights > 0)
                {
                    AreaEmitterInfo light;
                    const int light_id = rndInt(si.seed, 0, params.num_lights - 1);
                    light = params.lights[light_id];

                    LightInteraction li;
                    // Sampling light point
                    planeLightSampling(light, si.p, li, seed);
                    Vec3f to_light = li.p - si.p;
                    const float dist_to_light = length(to_light);
                    
                    const Vec3f unit_wi = normalize(to_light);

                    const float t_shadow = dist_to_light - 1e-3f;
                    uint32_t hit = 0;
                    // Trace shadow ray
                    optixTrace(params.handle, si.p, unit_wi, 0.01f, t_shadow, 0.0f, OptixVisibilityMask(1), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, 1, 2, 1, hit);
                    if (!hit){
                        // Accumurate contribution from light 
                        const Vec3f bsdf = optixDirectCall<Vec4f, SurfaceInteraction*, void*, const Vec3f&>(si.surface_info.callable_id.bsdf, &si, si.surface_info.data, unit_wi);
                        float bsdf_pdf = optixDirectCall<float, SurfaceInteraction*, void*, const Vec3f&>(si.surface_info.callable_id.pdf, &si, si.surface_info.data, unit_wi);

                        const float cos_theta = dot(-unit_wi, li.n);
                        const float light_pdf = li.pdf;

                        SurfaceInteraction light_si;
                        light_si.shading.uv = li.uv;
                        light_si.shading.n = li.n;
                        light_si.wo = unit_wi;
                        light_si.surface_info = light.surface_info;

                        Vec3f emission = optixDirectCall<Vec4f, SurfaceInteraction*, void*>(light_si.surface_info.callable_id.bsdf, &light_si, light_si.surface_info.data);

                        result += emission * bsdf / li.pdf;
                    }

                    // Evaluate BSDF 
                    Vec4f bsdf = optixDirectCall<Vec4f, SurfaceInteraction*, void*, const Vec3f&>(si.surface_info.callable_id.bsdf, &si, si.surface_info.data, si.wi);
                    // Evaluate PDF
                    float bsdf_pdf = optixDirectCall<float, SurfaceInteraction*, void*, const Vec3f&>(si.surface_info.callable_id.pdf, &si, si.surface_info.data, si.wi);
                    throughput *= bsdf / bsdf_pdf;
                }
                else
                {
                    // Evaluate BSDF 
                    Vec4f bsdf = optixDirectCall<Vec4f, SurfaceInteraction*, void*, const Vec3f&>(si.surface_info.callable_id.bsdf, &si, si.surface_info.data, si.wi);
                    // Evaluate PDF
                    float bsdf_pdf = optixDirectCall<float, SurfaceInteraction*, void*, const Vec3f&>(si.surface_info.callable_id.pdf, &si, si.surface_info.data, si.wi);
                    throughput *= bsdf / bsdf_pdf;
                }
            }
            
            // Generate next path
            ro = si.p;
            rd = si.wi;

            ++depth;
        }

        i--;
    }

    const uint32_t image_idx = idx.y() * params.width + idx.x();

    if (!result.isValid()) result = Vec3f(0.0f);

    Vec3f accum_color = result / static_cast<float>(params.samples_per_launch);
    
    // Accumrate color with previous frame
    if (frame > 0) {
        const float a = 1.0f / static_cast<float>(frame + 1);
        const Vec3f accum_color_prev(params.accum_buffer[image_idx]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }
    params.accum_buffer[image_idx] = Vec4f(accum_color, 1.0f);
    Vec3u color = make_color(accum_color);
    params.result_buffer[image_idx] = Vec4u(color, 255);
}

// miss
extern "C" DEVICE void __miss__envmap()
{
    MissData* data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    auto* env = reinterpret_cast<EnvironmentEmitter::Data*>(data->env_data);
    // Get pointer of SurfaceInteraction from two payload values
    SurfaceInteraction* si = getPtrFromTwoPayloads<SurfaceInteraction, 0>();

    Ray ray = getWorldRay();

    Shading shading;
    float t;
    const Sphere::Data env_sphere{Vec3f(0.0f), 1e8f};
    pgIntersectionSphere(&env_sphere, ray, &shading, &t);

    si->shading.uv = shading.uv;
    si->trace_terminate = true;
    si->surface_info.type = SurfaceType::None;
    si->emission = optixDirectCall<Vec4f, const Vec2f&, void*>(
        env->texture.prg_id, si->shading.uv, env->texture.data);
}

extern "C" DEVICE void __miss__shadow()
{
    setPayload<0>(0);
}

// Hitgroups 
extern "C" DEVICE void __intersection__box()
{
    pgHitgroupData* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
    auto* box = reinterpret_cast<Box::Data*>(data->shape_data);
    Ray ray = getLocalRay();
    pgReportIntersectionBox(box, ray);
}

extern "C" DEVICE void __intersection__plane()
{
    pgHitgroupData* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
    auto* plane = reinterpret_cast<Plane::Data*>(data->shape_data);
    Ray ray = getLocalRay();
    pgReportIntersectionPlane(plane, ray);
}

extern "C" DEVICE void __closesthit__custom()
{
    pgHitgroupData* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());

    Ray ray = getWorldRay();

    // If you use `reportIntersection*` function for intersection test, 
    // you can fetch the shading on a surface from two attributes
    Shading* shading = getPtrFromTwoAttributes<Shading, 0>();

    // Transform shading frame to world space
    shading->n = optixTransformNormalFromObjectToWorldSpace(shading->n);
    shading->dpdu = optixTransformVectorFromObjectToWorldSpace(shading->dpdu);
    shading->dpdv = optixTransformVectorFromObjectToWorldSpace(shading->dpdv);

    auto* si = getPtrFromTwoPayloads<SurfaceInteraction, 0>();

    si->p = ray.at(ray.tmax);
    si->shading = *shading;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->surface_info = data->surface_info;
}

// Mesh
extern "C" DEVICE void __closesthit__mesh()
{
    HitgroupData* data = (HitgroupData*)optixGetSbtDataPointer();
    const TriangleMesh::Data* mesh = reinterpret_cast<TriangleMesh::Data*>(data->shape_data);

    Ray ray = getWorldRay();
    const Vec2f bc = optixGetTriangleBarycentrics();

    const uint32_t prim_idx = optixGetPrimitiveIndex();
    Shading shading = pgGetMeshShading(mesh, bc, prim_idx);

    shading.n = normalize(optixTransformNormalFromObjectToWorldSpace(shading.n));
    shading.dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(shading.dpdu));
    shading.dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(shading.dpdv));

    auto* si = getPtrFromTwoPayloads<SurfaceInteraction, 0>();
    si->p = ray.at(ray.tmax);
    si->shading = shading;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->surface_info = data->surface_info;
}

extern "C" DEVICE void __anyhit__opacity()
{
    HitgroupData* data = (HitgroupData*)optixGetSbtDataPointer();
    if (!data->opacity_texture.data) return;

    const TriangleMesh::Data* mesh = reinterpret_cast<TriangleMesh::Data*>(data->shape_data);

    // Notify execution of anyhit shader
    setPayload<2>(1u);

    const Vec2f bc = optixGetTriangleBarycentrics();
    const uint32_t prim_idx = optixGetPrimitiveIndex();

    auto* si = getPtrFromTwoPayloads<SurfaceInteraction, 0>();

    const Face face = mesh->faces[prim_idx];

    const Vec2f texcoord0 = mesh->texcoords[face.texcoord_id.x()];
    const Vec2f texcoord1 = mesh->texcoords[face.texcoord_id.y()];
    const Vec2f texcoord2 = mesh->texcoords[face.texcoord_id.z()];

    const Vec2f texcoord = barycentricInterop(texcoord0, texcoord1, texcoord2, bc);

    const Vec4f opacity = optixDirectCall<Vec4f, const Vec2f&, void*>(data->opacity_texture.prg_id, texcoord, data->opacity_texture.data);
    if (opacity.w() == 0) {
       optixIgnoreIntersection();
    }
}

extern "C" DEVICE void __closesthit__shadow()
{
    // Hit to surface
    setPayload<0>(1);
}

// Direct light sampling
void planeLightSampling(const AreaEmitterInfo& area_emitter, const Vec3f& p, LightInteraction& li, uint32_t& seed)
{
    const auto* plane = (Plane::Data*)area_emitter.shape;

    // Sample local point on the area emitter
    const float x = rnd(seed, plane->min.x(), plane->max.x());
    const float z = rnd(seed, plane->min.x(), plane->max.x());
    //printf("plane->min: %f %f, plane->max: %f %f\n", plane->min.x(), plane->min.y(), plane->max.x(), plane->max.y());
    Vec3f rnd_p(x, 0.0f, z);
    rnd_p = area_emitter.objToWorld.pointMul(rnd_p);
    li.p = rnd_p;
    li.n = normalize(area_emitter.objToWorld.vectorMul(Vec3f(0, 1, 0)));
    li.uv = Vec2f((x - plane->min.x()) / (plane->max.x() - plane->min.x()), (z - plane->min.y()) / (plane->max.y() - plane->min.y()));
    
    const Vec3f corner0 = area_emitter.objToWorld.pointMul(Vec3f(plane->min.x(), 0.0f, plane->min.y()));
    const Vec3f corner1 = area_emitter.objToWorld.pointMul(Vec3f(plane->max.x(), 0.0f, plane->min.y()));
    const Vec3f corner2 = area_emitter.objToWorld.pointMul(Vec3f(plane->min.x(), 0.0f, plane->max.y()));
    li.area = length(cross(corner1 - corner0, corner2 - corner0));

    const Vec3f wi = rnd_p - p;
    const float t = length(wi);
    const float cos_theta = fabs(dot(li.n, normalize(wi)));
    if (cos_theta < math::eps)
        li.pdf = 0.0f;
    li.pdf = (t * t) / (li.area * cos_theta);
}

// Textures
extern "C" DEVICE Vec4f __direct_callable__bitmap(const Vec2f& uv, void* tex_data)
{
    return pgGetBitmapTextureValue<Vec4f>(uv, tex_data);
}

extern "C" DEVICE Vec4f __direct_callable__constant(const Vec2f& uv, void* tex_data)
{
    return pgGetConstantTextureValue<Vec4f>(uv, tex_data);
}

extern "C" DEVICE Vec4f __direct_callable__checker(const Vec2f& uv, void* tex_data)
{
    return pgGetCheckerTextureValue<Vec4f>(uv, tex_data);
}

// Surfaces
extern "C" DEVICE void __direct_callable__sample_diffuse(SurfaceInteraction* si, void* data)
{
    const Diffuse::Data* diffuse = reinterpret_cast<Diffuse::Data*>(data);
    si->wi = pgImportanceSamplingDiffuse(diffuse, si->wo, si->shading, si->seed);
    si->trace_terminate = false;
}

extern "C" DEVICE Vec4f __direct_callable__bsdf_diffuse(SurfaceInteraction* si, void* data, const Vec3f& wi)
{
    const Diffuse::Data* diffuse = reinterpret_cast<Diffuse::Data*>(data);
    const Vec4f albedo = optixDirectCall<Vec4f, const Vec2f&, void*>(diffuse->texture.prg_id, si->shading.uv, diffuse->texture.data);
    si->albedo = albedo;
    si->emission = Vec4f(0.0f);
    return si->albedo * pgGetDiffuseBRDF(wi, si->shading.n) * math::inv_pi;
}

extern "C" DEVICE float __direct_callable__pdf_diffuse(SurfaceInteraction* si, void* data, const Vec3f& wi)
{
    return pgGetDiffusePDF(wi, si->shading.n) * math::inv_pi;
}

extern "C" DEVICE Vec4f __direct_callable__area_emitter(SurfaceInteraction * si, void* data)
{
    const auto* area = reinterpret_cast<AreaEmitter::Data*>(data);
    si->trace_terminate = true;
    float is_emitted = dot(si->wo, si->shading.n) < 0.0f ? 1.0f : 0.0f;
    if (area->twosided)
    {
        is_emitted = 1.0f;
        si->shading.n = faceforward(si->shading.n, -si->wo, si->shading.n);
    }

    const Vec4f base = optixDirectCall<Vec4f, const Vec2f&, void*>(area->texture.prg_id, si->shading.uv, area->texture.data);
    si->albedo = base;
    si->emission = base * area->intensity * is_emitted;
    return si->emission;
}