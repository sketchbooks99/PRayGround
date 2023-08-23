#include <prayground/prayground.h>
#include "params.h"

extern "C" { __constant__ LaunchParams params; }

using SurfaceInteraction = SurfaceInteraction_<Vec4f>;

static INLINE DEVICE void trace(
    OptixTraversableHandle handle, const Vec3f& ro, const Vec3f& rd,
    float tmin, float tmax, uint32_t ray_type, SurfaceInteraction* si
)
{
    uint32_t u0, u1;
    packPointer(si, u0, u1);
    optixTrace(
        handle, ro, rd,
        tmin, tmax, 0,
        OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE,
        ray_type, 1, ray_type,
        u0, u1
    );
}

// raygen
extern "C" GLOBAL void __raygen__pinhole()
{
    const pgRaygenData<Camera>* raygen = (pgRaygenData<Camera>*)optixGetSbtDataPointer();

    const int frame = params.frame;

    const Vec3ui idx(optixGetLaunchIndex());
    uint32_t seed = tea<4>(idx.y() * params.width + idx.x(), seed);

    Vec3f result(0.0f);
    Vec3f normal(0.0f);

    int i = params.samples_per_launch;

    while (i > 0) {
        const Vec2f jitter = UniformSampler::get2D(seed) - 0.5f;
        const Vec2f d = 2.0f * Vec2f(
            (static_cast<float>(idx.x()) + jitter.x()) / params.width,
            (static_cast<float>(idx.y()) + jitter.y()) / params.height
        );

        Vec3f ro, rd;
        getCameraRay(raygen->camera, d.x(), d.y(), ro, rd);

        SurfaceInteraction si;
        si.seed = seed;
        si.emission = 0.0f;
        si.albedo = 0.0f;
        si.trace_terminate = false;

        int depth = 0;
        for (;;) {
            if (depth >= params.max_depth)
                break;

            trace(params.handle, ro, rd, 0.01f, 1e10f, 0, &si);

            if (si.trace_terminate) {
                result = si.albedo;
                break;
            }

            if (depth == 0)
                normal = si.shading.n;
            
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
extern "C" GLOBAL void __miss__envmap()
{
    pgMissData* data = reinterpret_cast<pgMissData*>(optixGetSbtDataPointer());
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
    if (env->texture.prg_id == CONSTANT_TEXTURE_PRG_ID)
        si->emission = pgGetConstantTextureValue<Vec4f>(si->shading.uv, env->texture.data);
    else if (env->texture.prg_id == CHECKER_TEXTURE_PRG_ID)
        si->emission = pgGetCheckerTextureValue<Vec4f>(si->shading.uv, env->texture.data);
    else if (env->texture.prg_id == BITMAP_TEXTURE_PRG_ID)
        si->emission = pgGetBitmapTextureValue<Vec4f>(si->shading.uv, env->texture.data);
}

// Mesh
extern "C" GLOBAL void __closesthit__mesh()
{
    pgHitgroupData* data = (pgHitgroupData*)optixGetSbtDataPointer();
    const TriangleMesh::Data* mesh = reinterpret_cast<TriangleMesh::Data*>(data->shape_data);

    Ray ray = getWorldRay();
    const Vec2f bc = optixGetTriangleBarycentrics();

    Shading shading = pgGetMeshShading(mesh, bc, optixGetPrimitiveIndex());

    shading.n = optixTransformNormalFromObjectToWorldSpace(shading.n);
    shading.dpdu = optixTransformVectorFromObjectToWorldSpace(shading.dpdu);
    shading.dpdv = optixTransformVectorFromObjectToWorldSpace(shading.dpdv);

    auto* si = getPtrFromTwoPayloads<SurfaceInteraction, 0>();
    si->p = ray.at(ray.tmax);
    si->shading = shading;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->albedo = Vec3f(1.0f);
    si->surface_info = data->surface_info;
}

extern "C" GLOBAL void __anyhit__opacity()
{

}
