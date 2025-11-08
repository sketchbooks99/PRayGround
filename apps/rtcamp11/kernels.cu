#include <prayground/prayground.h>
#include "params.h"

extern "C" { __constant__ LaunchParams params; }

#define MIS 1

using SurfaceInteraction = SurfaceInteraction_<Vec3f>;

// Smoothstep function for smooth interpolation
__device__ __forceinline__ float smoothstep(float edge0, float edge1, float x) {
    float t = fminf(fmaxf((x - edge0) / (edge1 - edge0), 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

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

// Shadow ray trace (returns visibility: 1 = visible, 0 = occluded)
static __forceinline__ __device__ uint32_t traceShadow(
    OptixTraversableHandle handle, const Vec3f& ro, const Vec3f& rd,
    float tmin, float tmax
)
{
    uint32_t visibility = 1;
    optixTrace(
        handle, ro, rd, tmin, tmax, 0.0f,
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        /* ray_type = */ 1, 2, /* ray_type = */ 1,
        visibility
    );
    return visibility;
}

static __forceinline__ __device__ Vec3f reinhardToneMap(const Vec3f& color, const float white)
{
    const float l = luminance(color);
    return (color * 1.0f) / (1.0f + l / white);
}

// MIS power heuristic (beta = 2)
static __forceinline__ __device__ float powerHeuristic(float pdf1, float pdf2)
{
    const float p1 = pdf1 * pdf1;
    const float p2 = pdf2 * pdf2;
    return p1 / (p1 + p2);
}

// Sample a point on sphere light (callable function)
extern "C" __device__ LightInteraction __direct_callable__sample_sphere_light(
    SurfaceInteraction* si, void* data)
{
    const auto* sphere = reinterpret_cast<Sphere::Data*>(data);
    
    const Vec3f hit_point = si->p;
    const Vec3f dir_to_center = sphere->center - hit_point;
    const float dist_to_center_sq = dot(dir_to_center, dir_to_center);
    
    // Sample uniformly on sphere surface
    const float z = 1.0f - 2.0f * UniformSampler::get1D(si->seed);
    const float r = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    const float phi = 2.0f * math::pi * UniformSampler::get1D(si->seed);
    const Vec3f local_point(r * cosf(phi), r * sinf(phi), z);
    
    // Transform to world space
    const Vec3f w = normalize(dir_to_center);
    const Vec3f u = normalize(cross(fabs(w.x()) > 0.1f ? Vec3f(0, 1, 0) : Vec3f(1, 0, 0), w));
    const Vec3f v = cross(w, u);
    
    const Vec3f light_normal = normalize(local_point.x() * u + local_point.y() * v + local_point.z() * w);
    
    LightInteraction light_interaction;
    light_interaction.p = sphere->center + light_normal * sphere->radius;
    light_interaction.n = light_normal;
    light_interaction.uv = Vec2f(0.0f);
    
    // Calculate PDF (1 / surface_area)
    const float surface_area = 4.0f * math::pi * sphere->radius * sphere->radius;
    const float pdf_area = 1.0f / surface_area;
    
    // Convert to solid angle PDF
    const Vec3f to_light = light_interaction.p - hit_point;
    const float dist_sq = dot(to_light, to_light);
    const float cos_theta = fabs(dot(light_interaction.n, -normalize(to_light)));
    
    // Convert area PDF to solid angle PDF: pdf_solid_angle = pdf_area * dist^2 / cos_theta
    light_interaction.pdf = pdf_area * dist_sq / fmaxf(cos_theta, 1e-8f);
    light_interaction.area = surface_area;
    return light_interaction;
}

// Calculate PDF for sphere light (callable function)
extern "C" __device__ float __direct_callable__pdf_sphere_light(
    const Vec3f& hit_point, const Vec3f& light_point, const Vec3f& light_normal, void* data)
{
    const auto* sphere = reinterpret_cast<Sphere::Data*>(data);
    
    const float surface_area = 4.0f * math::pi * sphere->radius * sphere->radius;
    const float pdf_area = 1.0f / surface_area;
    
    const Vec3f to_light = light_point - hit_point;
    const float dist_sq = dot(to_light, to_light);
    const float cos_theta = fabs(dot(light_normal, -normalize(to_light)));
    
    return pdf_area * dist_sq / fmaxf(cos_theta, 1e-8f);
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
        si.shading = {};
        SurfaceInfo surface_info;
        surface_info.type = SurfaceType::None;
        si.surface_info = &surface_info;

        bool env_evaluated = false;

        int depth = 0;
        for (;;) {
            if (depth >= params.max_depth)
                break;

            trace(params.handle, ro, rd, 0.01f, 1e10f, /* ray_type = */ 0, &si);

            if (si.trace_terminate) {
                if (!env_evaluated)
                    result += throughput * si.emission;
                break;
            }

            // Get emission from area emitter
            if (si.surface_info->type == SurfaceType::AreaEmitter) {
                // Evaluate emission from emitter
                Vec3f emission = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
                    si.surface_info->callable_id.bsdf, &si, si.surface_info->data);

                result += throughput * emission;
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

                env_evaluated = false;
            }
            // Diffuse sampling
            else if (+(si.surface_info->type & SurfaceType::Diffuse)) 
            {
#if MIS
                Vec3f L_dir(0.0f);  // Direct lighting contribution
                
                // Strategy selection: 50% area lights, 50% environment map (if available)
                const bool has_envmap = params.envmap_sampling_data != nullptr;
                const bool has_lights = params.n_lights > 0;
                const float light_sampling_prob = has_lights && has_envmap ? 0.5f : (has_lights ? 1.0f : 0.0f);
                const bool sample_area_light = UniformSampler::get1D(si.seed) < light_sampling_prob;
                
                // ===== Area light sampling for MIS =====
                if (has_lights && sample_area_light) {
                    // Pick a random light (uniform sampling)
                    const uint32_t light_idx = min(
                        static_cast<uint32_t>(UniformSampler::get1D(si.seed) * params.n_lights),
                        params.n_lights - 1
                    );
                    const AreaEmitterInfo& light_info = params.lights[light_idx];
                    
                    // Sample a point on the light
                    LightInteraction light_interaction = optixDirectCall<LightInteraction, SurfaceInteraction*, void*>(
                        light_info.sample_id, &si, light_info.shape_data);
                    
                    const Vec3f to_light = light_interaction.p - si.p;
                    const float dist_to_light_sq = dot(to_light, to_light);
                    const float dist_to_light = sqrtf(dist_to_light_sq);
                    const Vec3f wi_light = to_light / dist_to_light;
                    
                    // Check if light is on the correct side
                    const float cos_theta_light = dot(light_interaction.n, -wi_light);
                    const float cos_theta_surface = dot(si.shading.n, wi_light);
                    
                    if (cos_theta_light > 0.0f && cos_theta_surface > 0.0f) {
                        // Cast shadow ray
                        uint32_t visibility = traceShadow(
                            params.handle, si.p, wi_light, 0.01f, dist_to_light - 0.01f);
                        
                        if (visibility == 1) {  // Not occluded
                            // Evaluate BSDF at light direction
                            si.wi = wi_light;
                            Vec3f bsdf_light = optixContinuationCall<Vec3f, SurfaceInteraction*, void*>(
                                si.surface_info->callable_id.bsdf, &si, si.surface_info->data);
                            
                            // Evaluate light emission
                            SurfaceInteraction si_light;
                            si_light.shading.uv = light_interaction.uv;
                            si_light.shading.n = light_interaction.n;
                            si_light.wo = -wi_light;
                            si_light.surface_info = light_info.surface_info;
                            Vec3f emission = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
                                light_info.surface_info->callable_id.bsdf, &si_light, light_info.surface_info->data);
                            
                            const float pdf_light = light_interaction.pdf * static_cast<float>(params.n_lights);
                            
                            // Calculate BSDF PDF for light direction
                            float pdf_bsdf = optixDirectCall<float, SurfaceInteraction*, void*>(
                                si.surface_info->callable_id.pdf, &si, si.surface_info->data);
                            
                            // MIS weight using power heuristic
                            const float mis_weight = powerHeuristic(pdf_light, pdf_bsdf);
                            
                            // Account for sampling strategy probability
                            L_dir += (bsdf_light * emission * cos_theta_surface * mis_weight / pdf_light) / light_sampling_prob;
                        }
                    }
                }
                // ===== Environment map sampling for MIS =====
                else if (has_envmap && !sample_area_light) {
                    // Sample direction from environment map
                    SurfaceInteraction si_env = si;
                    si_env.p = si.p;
                    si_env.seed = si.seed;
                    optixDirectCall<void, SurfaceInteraction*, void*>(
                        params.envmap_sample_id, &si_env, params.envmap_sampling_data);
                    
                    // Note: sample_envmap writes the sampled direction to si_env.wo
                    // But we need it as incoming direction (wi)
                    const Vec3f wi_env = si_env.wo;
                    const float cos_theta_surface = dot(si.shading.n, wi_env);
                    
                    if (cos_theta_surface > 0.0f) {
                        // Cast shadow ray to infinity
                        uint32_t visibility = traceShadow(
                            params.handle, si.p, wi_env, 0.01f, 1e10f);
                        
                        if (visibility == 1) {  // Not occluded - hits environment
                            // Evaluate BSDF at environment direction
                            si.wi = wi_env;
                            Vec3f bsdf_env = optixContinuationCall<Vec3f, SurfaceInteraction*, void*>(
                                si.surface_info->callable_id.bsdf, &si, si.surface_info->data);
                            
                            // Get environment radiance by converting direction to UV
                            float theta = acosf(clamp(wi_env.y(), -1.0f, 1.0f));
                            float phi = atan2f(wi_env.z(), wi_env.x());
                            if (phi < 0) phi += 2.0f * math::pi;
                            Vec2f env_uv(phi / (2.0f * math::pi), theta / math::pi);

                            Vec3f env_radiance = optixDirectCall<Vec4f, const Vec2f&, void*>(
                                params.envmap_texture_id, env_uv, params.envmap_texture_data);

                            // Calculate PDF for environment map sampling
                            si_env.wo = wi_env;
                            float pdf_env = optixDirectCall<float, SurfaceInteraction*, void*>(
                                params.envmap_pdf_id, &si_env, params.envmap_sampling_data);
                            
                            // Calculate BSDF PDF for environment direction
                            float pdf_bsdf = optixDirectCall<float, SurfaceInteraction*, void*>(
                                si.surface_info->callable_id.pdf, &si, si.surface_info->data);
                            
                            // MIS weight using power heuristic
                            const float mis_weight = powerHeuristic(pdf_env, pdf_bsdf);
                            
                            if (pdf_env > 0.0f) {
                                // Account for sampling strategy probability
                                L_dir += (bsdf_env * env_radiance * cos_theta_surface * mis_weight / pdf_env) / (1.0f - light_sampling_prob);
                            }
                        }
                    }
                }
#else
                Vec3f L_dir(0.0f);
#endif
                // ===== BSDF sampling for indirect lighting =====
                // Sampling scattered direction
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info->callable_id.sample, &si, si.surface_info->data);

                // Evaluate BSDF
                Vec3f bsdf = optixContinuationCall<Vec3f, SurfaceInteraction*, void*>(
                    si.surface_info->callable_id.bsdf, &si, si.surface_info->data);

                // Evaluate PDF
                float pdf_bsdf = optixDirectCall<float, SurfaceInteraction*, void*>(
                    si.surface_info->callable_id.pdf, &si, si.surface_info->data);

                env_evaluated = true;

                if (pdf_bsdf > 0.0f) {
                    // Add direct lighting from light/envmap sampling
                    result += throughput * L_dir;
                    
                    // Update throughput for BSDF sampling (indirect lighting)
                    // Note: If next ray hits environment map (via __miss__envmap), 
                    // the contribution is: throughput * bsdf/pdf_bsdf * envmap_radiance
                    // This is correct for BSDF importance sampling of indirect lighting
                    throughput *= bsdf / pdf_bsdf;
                } else {
                    break;
                }
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
        // Proper cumulative average in linear space (before tone mapping)
        const Vec3f prev_color(params.accum_buffer[image_idx]);
        const float weight = 1.0f / static_cast<float>(frame + 1);
        accum_color = prev_color + (accum_color - prev_color) * weight;
    }

    // Store linear color for next frame's accumulation
    params.accum_buffer[image_idx] = Vec4f(accum_color, 1.0f);
    
    // Apply tone mapping only for display
    Vec3f display_color = reinhardToneMap(accum_color, params.white);
    Vec3u color = make_color(display_color);
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
    
    // Evaluate environment map radiance
    // Note: This handles two cases:
    // 1. Primary rays (depth=0): Direct view of environment background
    // 2. Indirect rays (depth>0): BSDF-sampled rays that miss geometry
    //    These are already weighted by BSDF PDF, so no additional MIS needed here
    Vec4f envmap_color = optixDirectCall<Vec4f, const Vec2f&, void*>(
        env->texture.prg_id, si->shading.uv, env->texture.data);
    
    // Convert Vec4f (RGBA) to Vec3f (RGB) for emission
    si->emission = Vec3f(envmap_color.x(), envmap_color.y(), envmap_color.z());
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

    Vec3f mesh_n = shading.n;
    Vec3f n(0.0f);
    if (data->surface_info->use_bumpmap) {
        n = optixDirectCall<Vec3f, Vec2f&, void*>(data->surface_info->bumpmap.prg_id, shading.uv, data->surface_info->bumpmap.data);
        Onb onb(mesh_n);
        onb.inverseTransform(n);
        shading.n = normalize(n);
    }

    // Transform shading from object to world space
    shading.n = normalize(optixTransformNormalFromObjectToWorldSpace(shading.n));
    shading.dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(shading.dpdu));
    shading.dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(shading.dpdv));

    SurfaceInteraction* si = getPtrFromTwoPayloads<SurfaceInteraction, 0>();
    //si->p = ray.at(ray.tmax) + shading.n * n.z() * 10.0f;
    si->p = ray.at(ray.tmax);
    si->shading = shading;
    si->t = ray.tmax;
    si->wo = -ray.d;
    si->surface_info = data->surface_info;
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
    if (pgIntersectionSphere(&s, ray, &shading, &time)) {
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

extern "C" __global__ void __closesthit__shadow() {
    setPayload<0>(0);
}

extern "C" __global__ void __anyhit__mesh_opacity() {
    pgHitgroupData* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
    const TriangleMesh::Data* mesh = reinterpret_cast<TriangleMesh::Data*>(data->shape_data);
    if (!data->surface_info->use_opacity_texture) return;

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

    const Vec4f opacity = optixDirectCall<Vec4f, const Vec2f&, void*>(
        data->surface_info->opacity_texture.prg_id, texcoord, data->surface_info->opacity_texture.data);
    
    if (opacity.w() == 0.0f) {
        optixIgnoreIntersection();
    }
}

extern "C" __global__ void __anyhit__custom_opacity() {

}

// ----------------------------------------------------------------
// BSDF callable programs
// ----------------------------------------------------------------
extern "C" __device__ void __direct_callable__sample_diffuse(SurfaceInteraction* si, void* data)
{
    const Diffuse::Data* diffuse = reinterpret_cast<Diffuse::Data*>(data);
    si->wi = pgImportanceSamplingDiffuse(diffuse, si->wo, si->shading, si->seed);
    si->trace_terminate = false;
}

extern "C" __device__ Vec3f __continuation_callable__bsdf_diffuse(SurfaceInteraction* si, void* data)
{
    const Diffuse::Data* diffuse = reinterpret_cast<Diffuse::Data*>(data);
    const Vec4f albedo = optixDirectCall<Vec4f, const Vec2f&, void*>(diffuse->texture.prg_id, si->shading.uv, diffuse->texture.data);
    si->albedo = albedo;
    si->emission = Vec3f(0.0f);
    return si->albedo * pgGetDiffuseBRDF(si->wi, si->shading.n);
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
    const Vec4f albedo = optixDirectCall<Vec4f, const Vec2f&, void*>(dielectric->texture.prg_id, si->shading.uv, dielectric->texture.data);
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
    const Vec4f base = optixDirectCall<Vec4f, const Vec2f&, void*>(disney->albedo.prg_id, si->shading.uv, disney->albedo.data);
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

    const Vec4f base = optixDirectCall<Vec4f, const Vec2f&, void*>(area->texture.prg_id, si->shading.uv, area->texture.data);
    si->albedo = base;
    si->emission = base * area->intensity * is_emitted;
    return si->emission;
}

// Textures
extern "C" __device__ Vec4f __direct_callable__bitmap(const Vec2f& uv, void* tex_data) {
    return pgGetBitmapTextureValue<Vec4f>(uv, tex_data);
}

extern "C" __device__ Vec4f __direct_callable__constant(const Vec2f& uv, void* tex_data) {
    return pgGetConstantTextureValue<Vec4f>(uv, tex_data);
}

extern "C" __device__ Vec4f __direct_callable__checker(const Vec2f& uv, void* tex_data) {
    return pgGetCheckerTextureValue<Vec4f>(uv, tex_data);
}

extern "C" __device__ Vec4f __direct_callable__procedural_wooden(const Vec2f& uv, void* tex_data) {
    ProceduralWoodenTexture::Data* wooden = reinterpret_cast<ProceduralWoodenTexture::Data*>(tex_data);
    return wooden->light_wood_color;
}

extern "C" __device__ Vec4f __direct_callable__terrain_heightmap(const Vec2f& uv, void* tex_data) {
    FloatBitmapTexture::Data* heightmap = reinterpret_cast<FloatBitmapTexture::Data*>(tex_data);
    const float height = tex2D<float>(heightmap->texture, uv.x(), uv.y());
    Vec4f color;
    if (height < 0.3f)      color = Vec4f(0.2, 0.6, 0.2, 1.0f);  // 草原
    else if (height < 0.6f) color = Vec4f(0.4, 0.3, 0.2, 1.0f);  // 森
    else if (height < 0.8f) color = Vec4f(0.5, 0.5, 0.5, 1.0f);  // 岩
    else                    color = Vec4f(1.0, 1.0, 1.0, 1.0f);  // 雪
    return color;
}

extern "C" __device__ Vec4f __direct_callable__star_night(const Vec2f& uv, void* tex_data) {
    StarNightTexture::Data* star_night = reinterpret_cast<StarNightTexture::Data*>(tex_data);
    auto n = star_night->noise_data;
    RandomNoise r_noise(n.seed);
    float star_thres = star_night->star_threshold;
    float star_intensity = star_night->star_intensity;
    Vec3f moon_dir = normalize(star_night->moon_dir);
    float moon_intensity = star_night->moon_intensity;
    Vec3i p = Vec3i(
        static_cast<int>(uv.x() * n.width),
        static_cast<int>(uv.y() * n.height),
        0
    );
    float value = r_noise.noise(p);

    float theta = uv.y() * math::pi;
    float phi = uv.x() * 2.0f * math::pi;
    Vec3f p3d(
        sinf(theta) * cosf(phi),
        cosf(theta),
        sinf(theta) * sinf(phi)
    );

    float moon_proximity = dot(moon_dir, p3d);
    float moon_radius = 0.03f;
    
    if (moon_proximity > cosf(moon_radius)) {
        float moon_factor = (moon_proximity - cosf(moon_radius)) / (1.0f - cosf(moon_radius));
        return Vec4f(1.0f, 1.0f, 0.9f, 1.0f) * (moon_intensity + moon_factor * moon_intensity);
    } 
    else if (value > star_thres) {
        float color_rnd = rnd(n.seed);
        Vec3f star_color = Vec3f(1.0f, 1.0f, 0.9f);
        float intensity = star_intensity * rnd(n.seed);
        if (color_rnd < 0.1f) {
            star_color = Vec3f(1.0f, 0.8f, 0.6f);  // Warm star
        }
        else if (color_rnd < 0.2f) {
            star_color = Vec3f(0.6f, 0.8f, 1.0f);  // Cool star
        }
        return star_color * intensity;
    }
    else {
        return star_night->base_color;
    }
}

// ========================================
// 環境マップの重点サンプリング
// ========================================

// Binary search for CDF inversion
__device__ uint32_t binary_search_cdf(const float* cdf, uint32_t size, float xi) {
    uint32_t left = 0;
    uint32_t right = size;
    while (left < right) {
        uint32_t mid = (left + right) / 2;
        if (cdf[mid] < xi) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return min(left, size - 1);
}

// Sample direction from environment map
extern "C" __device__ void __direct_callable__sample_envmap(
    SurfaceInteraction* si, void* data)
{
    EnvmapSamplingData* envmap = reinterpret_cast<EnvmapSamplingData*>(data);
    
    float xi1 = rnd(si->seed);
    float xi2 = rnd(si->seed);
    
    // 1. Sample row (v coordinate) from marginal CDF
    uint32_t v = binary_search_cdf(envmap->marginal_cdf, envmap->height, xi1);
    
    // 2. Sample column (u coordinate) from conditional CDF
    const float* row_cdf = envmap->conditional_cdf + v * envmap->width;
    uint32_t u = binary_search_cdf(row_cdf, envmap->width, xi2);
    
    // Get PDF values
    float pdf_v = v > 0 ? 
        (envmap->marginal_cdf[v] - envmap->marginal_cdf[v-1]) : 
        envmap->marginal_cdf[0];
    float pdf_u = u > 0 ? 
        (row_cdf[u] - row_cdf[u-1]) : 
        row_cdf[0];
    
    // Convert to continuous values (with jittering for better sampling)
    float du = rnd(si->seed);  // Random offset within pixel
    float dv = rnd(si->seed);
    
    Vec2f uv((u + du) / envmap->width, (v + dv) / envmap->height);
    
    // Convert UV coordinates to direction vector (spherical coordinates)
    float theta = uv.y() * math::pi;           // [0, pi]
    float phi = uv.x() * 2.0f * math::pi;      // [0, 2pi]
    
    Vec3f world_dir(
        sinf(theta) * cosf(phi),
        cosf(theta),
        sinf(theta) * sinf(phi)
    );
    
    // Store sampled direction in wi (incoming light direction)
    // Note: We use 'wo' field here for compatibility, but it represents incoming direction
    si->wo = world_dir;
    si->trace_terminate = false;
}

// Compute PDF for environment map sampling
extern "C" __device__ float __direct_callable__pdf_envmap(
    SurfaceInteraction* si, void* data)
{
    EnvmapSamplingData* envmap = reinterpret_cast<EnvmapSamplingData*>(data);
    
    Vec3f dir = si->wo;
    
    // Convert direction vector to UV coordinates
    float theta = acosf(clamp(dir.y(), -1.0f, 1.0f));
    float phi = atan2f(dir.z(), dir.x());
    if (phi < 0) phi += 2.0f * math::pi;
    
    Vec2f uv(phi / (2.0f * math::pi), theta / math::pi);
    
    // Convert UV coordinates to pixel index
    uint32_t u = min(static_cast<uint32_t>(uv.x() * envmap->width), envmap->width - 1);
    uint32_t v = min(static_cast<uint32_t>(uv.y() * envmap->height), envmap->height - 1);
    
    // Get pixel luminance via callable function
    Vec3f radiance = optixDirectCall<Vec4f, const Vec2f&, void*>(
        envmap->texture_id, uv, envmap->texture_data);
    
    float luminance = 0.299f * radiance.x() + 0.587f * radiance.y() + 0.114f * radiance.z();
    
    // PDF in solid angle measure
    // PDF(ω) = (L(ω) * sin(θ)) / (∫ L(ω') * sin(θ') dω')
    //        = (L(ω) * sin(θ)) / total_luminance
    // Then convert from solid angle to area measure on the environment sphere
    
    float sin_theta = sinf(theta);
    if (sin_theta < 1e-6f) sin_theta = 1e-6f;  // Avoid division by zero
    
    // PDF in solid angle
    float pdf_omega = (luminance * sin_theta) / envmap->total_luminance;
    
    // Convert to PDF per steradian by accounting for the Jacobian of the transformation
    // from (u,v) to solid angle: dω = sin(θ) du dv / (width * height)
    float jacobian = (2.0f * math::pi * math::pi) / (envmap->width * envmap->height * sin_theta);
    
    return pdf_omega / jacobian;
}





