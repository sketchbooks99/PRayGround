#include <prayground/prayground.h>
#include "params.h"

extern "C" { __constant__ LaunchParams params; }

using SurfaceInteraction = SurfaceInteraction_<Vec3f>;

#define STORE_INTERACTION() \
    si->p = ray.at(ray.tmax); \
    si->shading.n = n; \
    si->shading.uv = uv; \
    si->shading.dpdu = dpdu; \
    si->shading.dpdv = dpdv; \
    si->t = ray.tmax; \
    si->wo = -ray.d; \
    si->surface_info = data->surface_info;

static __forceinline__ __device__ void trace(
    OptixTraversableHandle handle, const Vec3f& ro, const Vec3f& rd,
    float tmin, float tmax, uint32_t ray_type, SurfaceInteraction* si
)
{
    uint32_t u0, u1;
    packPointer(si, u0, u1);
    optixTrace(
        handle, ro, rd, tmin, tmax, 0.0f,
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE, ray_type, 2, ray_type,
        u0, u1
    );
}

extern "C" __global__ void __raygen__pinhole() {
    const pgRaygenData<Camera>* raygen = (pgRaygenData<Camera>*)optixGetSbtDataPointer();

    const int frame = params.frame;

    const Vec3ui idx(optixGetLaunchIndex());

    const int image_idx = idx.y() * params.width + idx.x();
    const int width = params.width;
    const int height = params.height;
    uint32_t seed = tea<4>(image_idx, frame);

    Vec3f result(0.0f);

    int i = params.samples_per_launch;

    while (i > 0) {
        const Vec2f jitter = UniformSampler::get2D(seed) - 0.5f;
        const Vec2f d = 2.0f * Vec2f(
            static_cast<float>(idx.x()) + jitter.x(),
            static_cast<float>(idx.y()) + jitter.y()
        ) / Vec2f(width, height) - 1.0f;

        Vec3f ro, rd;
        getCameraRay(raygen->camera, d.x(), d.y(), ro, rd);

        Vec3f throughput(1.0f);

        SurfaceInteraction si;
        si.seed = seed;
        si.emission = 0.0f;
        si.albedo = 0.0f;
        si.trace_terminate = false;
        SurfaceInfo surface_info;
        surface_info.type = SurfaceType::None;
        si.surface_info = &surface_info;

        int depth = 0;
        for (;;) {
            if (depth >= params.max_depth)
                break;

            trace(params.handle, ro, rd, 0.01f, 1e10f, /* ray_type = */ 0, &si);

            if (si.trace_terminate) {
                result += throughput * si.emission;
                break;
            }

            // Get emission from area emitter
            if (si.surface_info->type == SurfaceType::AreaEmitter) {
                // Evaluate emission from emitter
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info->callable_id.bsdf, &si, si.surface_info->data);

                result += throughput * si.emission;
                if (si.trace_terminate)
                    break;
            }
            // Specular sampling
            else if (+(si.surface_info->type & SurfaceType::Delta)) {
                // Sample scattered direction
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info->callable_id.sample, &si, si.surface_info->data);

                // Evaluate BSDF
                Vec3f bsdf = optixContinuationCall<Vec3f, SurfaceInteraction*, void*>(
                    si.surface_info->callable_id.bsdf, &si, si.surface_info->data);

                throughput *= bsdf;
            }
            // Diffuse sampling
            else if (+(si.surface_info->type & SurfaceType::Diffuse)) {
                // Sampling scattered direction
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info->callable_id.sample, &si, si.surface_info->data);

                // Evaluate BSDF
                Vec3f bsdf = optixContinuationCall<Vec3f, SurfaceInteraction*, void*>(
                    si.surface_info->callable_id.bsdf, &si, si.surface_info->data);

                // Evaluate PDF
                float pdf = optixDirectCall<float, SurfaceInteraction*, void*>(
                    si.surface_info->callable_id.pdf, &si, si.surface_info->data);

                if (pdf > 0.0f)
                    throughput *= bsdf / pdf;
                else
                    break;
            }

            ro = si.p;
            rd = si.wi;

            ++depth;
        } i--;
    }

    if (!result.isValid())
        result = Vec3f(0.0f);
    
    Vec3f accum_color = result / static_cast<float>(params.samples_per_launch);

    if (frame > 0) {
        const float a = 1.0f / static_cast<float>(frame + 1);
        const Vec3f prev_color(params.accum_buffer[image_idx]);
        accum_color = lerp(prev_color, accum_color, a);
    }

    params.accum_buffer[image_idx] = Vec4f(accum_color, 1.0f);
    Vec3u color = make_color(accum_color);
    params.result_buffer[image_idx] = Vec4u(color, 255);
}

// ----------------------------------------------------------------
// Miss programs
// ----------------------------------------------------------------
extern "C" __global__ void __miss__envmap() {
    pgMissData* data = (pgMissData*)optixGetSbtDataPointer();
    auto* env = reinterpret_cast<EnvironmentEmitter::Data*>(data->env_data);
    SurfaceInteraction* si = getPtrFromTwoPayloads<SurfaceInteraction, 0>();

    Ray ray = getWorldRay();

    Shading shading;
    float t;
    const Sphere::Data env_sphere{ Vec3f(0.0f), 1e8f };
    pgIntersectionSphere(&env_sphere, ray, &shading, &t);

    si->shading.uv = shading.uv;
    si->trace_terminate = true;
    si->emission = optixDirectCall<Vec3f, const Vec2f&, void*>(
        env->texture.prg_id, si->shading.uv, env->texture.data);
}

extern "C" __global__ void __miss__shadow() {
    setPayload<0>(1);
}

// ----------------------------------------------------------------
// Hitgroup programs
// ----------------------------------------------------------------
// Mesh
extern "C" __global__ void __closesthit__mesh() {
    pgHitgroupData* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
    const TriangleMesh::Data* mesh = reinterpret_cast<TriangleMesh::Data*>(data->shape_data);

    Ray ray = getWorldRay();

    Shading shading = pgGetMeshShading(mesh, optixGetTriangleBarycentrics(), optixGetPrimitiveIndex());

    // Transform shading from object to world space
    shading.n = normalize(optixTransformNormalFromObjectToWorldSpace(shading.n));
    shading.dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(shading.dpdu));
    shading.dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(shading.dpdv));

    SurfaceInteraction* si = getPtrFromTwoPayloads<SurfaceInteraction, 0>();
    si->p = ray.at(ray.tmax);
    si->shading = shading;
    si->t = ray.tmax;
    si->wo = -ray.d;
    si->surface_info = data->surface_info;
}

extern "C" __global__ void __closesthit__shadow() {
    setPayload<0>(0);
}

// Sphere
extern "C" __global__ void __intersection__sphere() {
    pgHitgroupData* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
    const Sphere::Data* sphere = reinterpret_cast<Sphere::Data*>(data->shape_data);

    Ray ray = getLocalRay();

    Shading shading;
    float t;
    if (pgIntersectionSphere(sphere, ray, &shading, &t)) {
        optixReportIntersection(t, 0, Vec3f_as_ints(shading.n), Vec2f_as_ints(shading.uv));
    }
}

extern "C" __global__ void __closesthit__sphere() {
    pgHitgroupData* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());

    Vec3f n = getVec3fFromAttribute<0>();
    Vec2f uv = getVec2fFromAttribute<3>();

    // Compute local differential geometry
    float phi = atan2(n.z(), n.x());
    if (phi < 0) phi += math::two_pi;
    const float theta = acosf(n.y());
    Vec3f dpdu = Vec3f(-math::two_pi * n.z(), 0, math::two_pi * n.x());
    Vec3f dpdv = math::pi * Vec3f(n.y() * cosf(phi), -sinf(theta), n.y() * sinf(phi));

    // Convert local shading frame to world space
    n = normalize(optixTransformNormalFromObjectToWorldSpace(n));
    dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(dpdu));
    dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(dpdv));

    SurfaceInteraction* si = getPtrFromTwoPayloads<SurfaceInteraction, 0>();
    Ray ray = getWorldRay();

    STORE_INTERACTION();
}

// Plane
extern "C" __global__ void __intersection__plane() {
    const pgHitgroupData* data = (pgHitgroupData*)optixGetSbtDataPointer();
    const Plane::Data* plane = reinterpret_cast<Plane::Data*>(data->shape_data);

    Ray ray = getLocalRay();

    Shading shading = {};
    float time = 0.0f;
    if (pgIntersectionPlane(plane, ray, &shading, &time)) {
        optixReportIntersection(time, 0, Vec3f_as_ints(shading.n), Vec2f_as_ints(shading.uv));
    }
}

extern "C" __global__ void __closesthit__plane() {
    pgHitgroupData* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());

    Vec3f n = getVec3fFromAttribute<0>();
    Vec2f uv = getVec2fFromAttribute<3>();
    Vec3f dpdu = Vec3f(1, 0, 0);
    Vec3f dpdv = Vec3f(0, 0, 1);

    // Convert local shading frame to world space
    n = normalize(optixTransformNormalFromObjectToWorldSpace(n));
    dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(dpdu));
    dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(dpdv));

    SurfaceInteraction* si = getPtrFromTwoPayloads<SurfaceInteraction, 0>();
    Ray ray = getWorldRay();

    STORE_INTERACTION();
}

extern "C" __global__ void __intersection__pcd() {
    const pgHitgroupData* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
    const PointCloud::Data* pcd = reinterpret_cast<PointCloud::Data*>(data->shape_data);

    uint32_t idx = optixGetPrimitiveIndex();
    const PointCloud::Data p = pcd[idx];
    Sphere::Data s = { p.point, p.radius };

    Ray ray = getLocalRay();

    Shading shading = {};
    float time;
    if (pgIntersectionSphere(&s, ray, &shading, &time))
    {
        optixReportIntersection(time, 0, Vec3f_as_ints(shading.n), Vec2f_as_ints(shading.uv));
    }
}

// Curves
extern "C" __global__ void __closesthit__curves()
{
    const pgHitgroupData* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
    const Curves::Data* curves = reinterpret_cast<Curves::Data*>(data->shape_data);

    // Get segment ID
    const uint32_t primitive_id = optixGetPrimitiveIndex();

    Ray ray = getWorldRay();
    Vec3f hit_point = optixTransformPointFromWorldToObjectSpace(ray.at(ray.tmax));

    Shading shading = pgGetCurvesShading(hit_point, primitive_id, optixGetPrimitiveType());
    // Transform shading frame to world space
    shading.n = normalize(optixTransformNormalFromObjectToWorldSpace(shading.n));
    shading.dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(shading.dpdu));
    shading.dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(shading.dpdv));

    SurfaceInteraction* si = getPtrFromTwoPayloads<SurfaceInteraction, 0>();
    si->p = optixTransformPointFromObjectToWorldSpace(hit_point);
    si->shading = shading;
    si->t = ray.tmax;
    si->wo = -ray.d;
    si->surface_info = data->surface_info;
}

extern "C" __device__ void __direct_callable__sample_diffuse(SurfaceInteraction* si, void* data)
{
    const Diffuse::Data* diffuse = reinterpret_cast<Diffuse::Data*>(data);
    si->wi = pgImportanceSamplingDiffuse(diffuse, si->wo, si->shading, si->seed);
    si->trace_terminate = false;
}

extern "C" __device__ Vec3f __continuation_callable__bsdf_diffuse(SurfaceInteraction* si, void* data)
{
    const Diffuse::Data* diffuse = reinterpret_cast<Diffuse::Data*>(data);
    const Vec3f albedo = optixDirectCall<Vec3f, const Vec2f&, void*>(diffuse->texture.prg_id, si->shading.uv, diffuse->texture.data);
    si->albedo = albedo;
    si->emission = Vec3f(0.0f);
    return albedo * pgGetDiffuseBRDF(si->wi, si->shading.n);
}

extern "C" __device__ float __direct_callable__pdf_diffuse(SurfaceInteraction* si, void* data)
{
    return pgGetDiffusePDF(si->wi, si->shading.n);
}

// Dielectric
extern "C" __device__ void __direct_callable__sample_glass(SurfaceInteraction* si, void* data)
{
    const Dielectric::Data* dielectric = reinterpret_cast<Dielectric::Data*>(data);
    si->wi = pgSamplingSmoothDielectric(dielectric, si->wo, si->shading, si->seed);
    si->trace_terminate = false;
}

extern "C" __device__ Vec3f __continuation_callable__bsdf_glass(SurfaceInteraction* si, void* data)
{
    const Dielectric::Data* dielectric = reinterpret_cast<Dielectric::Data*>(data);
    const Vec3f albedo = optixDirectCall<Vec3f, const Vec2f&, void*>(dielectric->texture.prg_id, si->shading.uv, dielectric->texture.data);
    si->albedo = albedo;
    si->emission = Vec3f(0.0f);
    return albedo;
}

extern "C" __device__ float __direct_callable__pdf_glass(SurfaceInteraction* si, void* data)
{
    return 1.0f;
}

// Disney
extern "C" __device__ void __direct_callable__sample_disney(SurfaceInteraction* si, void* data)
{
    const Disney::Data* disney = reinterpret_cast<Disney::Data*>(data);
    si->wi = pgImportanceSamplingDisney(disney, si->wo, si->shading, si->seed);
    si->trace_terminate = false;
}

extern "C" __device__ Vec3f __continuation_callable__bsdf_disney(SurfaceInteraction* si, void* data)
{
    const Disney::Data* disney = reinterpret_cast<Disney::Data*>(data);
    const Vec3f base = optixDirectCall<Vec3f, const Vec2f&, void*>(disney->albedo.prg_id, si->shading.uv, disney->albedo.data);
    return pgGetDisneyBRDF(disney, si->wo, si->wi, si->shading, base);
}

extern "C" __device__ float __direct_callable__pdf_disney(SurfaceInteraction* si, void* data)
{
    const Disney::Data* disney = reinterpret_cast<Disney::Data*>(data);
    return pgGetDisneyPDF(disney, si->wo, si->wi, si->shading);
}

// Area emitter
extern "C" __device__ Vec3f __direct_callable__area_emitter(SurfaceInteraction* si, void* data)
{
    const auto* area = reinterpret_cast<AreaEmitter::Data*>(data);
    si->trace_terminate = true;
    float is_emitted = dot(si->wo, si->shading.n) > 0.0f ? 1.0f : 0.0f;
    if (area->twosided)
    {
        is_emitted = 1.0f;
        si->shading.n = faceforward(si->shading.n, si->wo, si->shading.n);
    }

    const Vec3f base = optixDirectCall<Vec3f, const Vec2f&, void*>(area->texture.prg_id, si->shading.uv, area->texture.data);
    si->albedo = base;
    si->emission = base * area->intensity * is_emitted;
}

// Textures
extern "C" __device__ Vec3f __direct_callable__bitmap(const Vec2f& uv, void* tex_data) {
    return pgGetBitmapTextureValue<Vec3f>(uv, tex_data);
}

extern "C" __device__ Vec3f __direct_callable__constant(const Vec2f& uv, void* tex_data) {
    return pgGetConstantTextureValue<Vec3f>(uv, tex_data);
}

extern "C" __device__ Vec3f __direct_callable__checker(const Vec2f& uv, void* tex_data) {
    return pgGetCheckerTextureValue<Vec3f>(uv, tex_data);
}

extern "C" __device__ Vec3f __direct_callable__procedural_wooden(const Vec2f& uv, void* tex_data) {
    ProceduralWoodenTexture::Data* wooden = reinterpret_cast<ProceduralWoodenTexture::Data*>(tex_data);
    return wooden->light_wood_color;
}