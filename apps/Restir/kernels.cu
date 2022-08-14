#include <prayground/prayground.h>
#include "params.h"

extern "C" { __constant__ LaunchParams params; }

using SurfaceInteraction = SurfaceInteraction_<Vec3f>;

struct Reservoir {
    int y;          // The output sample (the index of light)
    float wsum;     // The sum of weights
    int M;          // The number of samples seen so far
    float W;        // Probalistic weight

    void update(int i, float weight, uint32_t& seed)
    {
        wsum += weight;
        M++;
        if (rnd(seed) < (weight / wsum))
            y = i;
    }
};

static __forceinline__ __device__ float calcWeight(const Vec3f& sample)
{
    /// @todo provisional value
    return 1.0f;
}

static __forceinline__ __device__ Reservoir reservoirSampling(int32_t num_strategies, Vec3f sample, int* samples, uint32_t& seed)
{
    Reservoir r;
    for (int i = 0; i < num_strategies; i++)
        r.update(samples[i], calcWeight(sample), seed);
    return r;
}

// p^(x) = \rho(x) * Le(x) * G(x), where \rho(x) = BRDF
static __forceinline__ __device__ float targetPDF(
    const Vec3f& brdf, SurfaceInteraction* si, const Vec3f& to_light, const LightInfo& light)
{
    float3 N = light.triangle.n;
    N = faceforward(N, normalize(-to_light), N);
    const float area = length(cross(light.triangle.v1 - light.triangle.v0, light.triangle.v2 - light.triangle.v0)) * 0.5f;
    const float cos_theta = fmaxf(dot(light.triangle.n, normalize(-to_light)), 0.0001f);
    const float d = length(to_light);
    const float G = (d * d) / (area * cos_theta);

    return length(brdf * light.emission) * G;
}

static __forceinline__ __device__ Vec3f randomSampleOnTriangle(uint32_t& seed, const Triangle& triangle)
{
    // Uniform sampling of barycentric coordinates on a triangle
    Vec2f uv = UniformSampler::get2D(seed);
    return triangle.v0 * (1.0f - uv.x() - uv.y()) + triangle.v1 * uv.x() + triangle.v2 * uv.y();
}

static __forceinline__ __device__ Reservoir reservoirImportanceSampling(
    SurfaceInteraction* si, int M, uint32_t& seed)
{
    Reservoir r{ 0, 0, 0, 0 };
    for (int i = 0; i < min(params.num_lights, M); i++)
    {
        // Sample a light
        int light_idx = rndInt(seed, 0, params.num_lights - 1);
        LightInfo light = params.lights[light_idx];

        const Vec3f light_p = randomSampleOnTriangle(seed, light.triangle);

        // Get brdf wrt the sampled light
        Vec3f brdf = optixContinuationCall<Vec3f, SurfaceInteraction*, void*, const Vec3f&>(
            si->surface_info.callable_id.bsdf, si, si->surface_info.data, light_p);
        float pdf = optixDirectCall<float, SurfaceInteraction*, void*, const Vec3f&>(
            si->surface_info.callable_id.pdf, si, si->surface_info.data, light_p);
        pdf = fmaxf(pdf, 0.001f);
        // Get target pdf 
        const float target_pdf = targetPDF(brdf, si, light_p - si->p, light);
        // update reservoir
        r.update(light_idx, target_pdf / pdf, seed);
    }

    LightInfo light = params.lights[r.y];
    const Vec3f light_p = randomSampleOnTriangle(seed, light.triangle);
    Vec3f brdf = optixContinuationCall<Vec3f, SurfaceInteraction*, void*, const Vec3f&>(
        si->surface_info.callable_id.bsdf, si, si->surface_info.data, light_p);

    // r.W = ( ( 1.0f / targetPDF(brdf, si, light_p - si->p, light) ) * ( 1.0f / r.M ) * r.wsum );
    //r.W = fmaxf(( 1.0f / targetPDF(brdf, si, light_p - si->p, light) ) * ( 1.0f / (float)r.M ) * r.wsum, 0.0f);
    r.W = fmaxf( (1.0f / targetPDF(brdf, si, light_p - si->p, light)) * (1.0f / (float)r.M) * r.wsum, 0.0f);
    if (r.W != r.W) 
        r.W = 0.0f;

    //printf("Reservoir: r.y: %d, r.wsum: %f, r.M: %d, r.W: %f\n", r.y, r.wsum, r.M, r.W);
    return r;
}

static __forceinline__ __device__ SurfaceInteraction* getSurfaceInteraction()
{
    const uint32_t u0 = getPayload<0>();
    const uint32_t u1 = getPayload<1>();
    return reinterpret_cast<SurfaceInteraction*>(unpackPointer(u0, u1));
}

static __forceinline__ __device__ void trace(
    OptixTraversableHandle handle, const Vec3f& ro, const Vec3f& rd, 
    float tmin, float tmax, SurfaceInteraction* si)
{
    uint32_t u0, u1;
    packPointer(si, u0, u1);
    optixTrace(
        handle, ro, rd, 
        tmin, tmax, 0, 
        OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE, 
        (uint32_t)RayType::Radiance, (uint32_t)RayType::NRay, (uint32_t)RayType::Radiance, 
        u0, u1);
}

static __forceinline__ __device__ bool traceShadow(
    OptixTraversableHandle handle, const Vec3f& ro, const Vec3f& rd,
    float tmin, float tmax, SurfaceInteraction* si)
{
    uint32_t hit = 0u;
    optixTrace(
        handle, ro, rd, 
        tmin, tmax, 0.0f,
        OptixVisibilityMask(1), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        (uint32_t)RayType::Shadow, (uint32_t)RayType::NRay, (uint32_t)RayType::Shadow,
        hit
    );
    return static_cast<bool>(hit);
}

// raygen 
extern "C" __device__ void __raygen__restir()
{
    const auto* raygen = reinterpret_cast<pgRaygenData<Camera>*>(optixGetSbtDataPointer());

    const int frame = params.frame;
    const Vec3ui idx(optixGetLaunchIndex());
    uint32_t seed = tea<4>(idx.x() * params.width + idx.y(), frame);

    Vec3f result(0.0f);

    int spl = params.samples_per_launch;

    const int M = 32;

    for (int i = 0; i < spl; i++) 
    {
        const Vec2f jitter = UniformSampler::get2D(seed) - 0.5f;

        const Vec2f d = 2.0f * Vec2f(
            (static_cast<float>(idx.x()) + jitter.x()) / params.width,
            (static_cast<float>(idx.y()) + jitter.y()) / params.height
        ) - 1.0f;

        Vec3f ro,rd;
        getCameraRay(raygen->camera, d.x(), d.y(), ro, rd);

        Vec3f throughput(1.0f);

        SurfaceInteraction si;
        si.seed = seed;
        si.emission = Vec3f(0.0f);
        si.albedo = Vec3f(0.0f);
        si.trace_terminate = false;

        int depth = 0;
        for (;;) 
        {
            if ( depth >= params.max_depth || si.trace_terminate)
                break;

            trace(params.handle, ro, rd, 0.01f, 1e16f, &si);

            if (si.trace_terminate)
            {
                result += si.emission * throughput;
                break;
            }

            if ( si.surface_info.type == SurfaceType::AreaEmitter )
            {
                // Evaluation of emittance from area emitter
                const Vec3f emission = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
                    si.surface_info.callable_id.bsdf, &si, si.surface_info.data);
                result += emission * throughput;
                if (si.trace_terminate)
                    break;
            }
            else 
            {
                
                Reservoir r = reservoirImportanceSampling(&si, M, seed);
                LightInfo light = params.lights[r.y];
                const Vec3f light_p = randomSampleOnTriangle(seed, light.triangle);
                const Vec3f to_light = light_p - si.p;

                const float nDl = dot(si.shading.n, normalize(to_light));

                Vec3f LN = light.triangle.n;
                LN = faceforward(LN, normalize(to_light), LN);
                const float LnDl = dot(LN, normalize(to_light));
                // rho
                Vec3f brdf = optixContinuationCall<Vec3f, SurfaceInteraction*, void*, const Vec3f&>(
                    si.surface_info.callable_id.bsdf, &si, si.surface_info.data, light_p);

                float weight = 0.0f;
                if (nDl > 0.0f && LnDl > 0.0f)
                {
                    // Visibility term
                    bool occluded = traceShadow(params.handle, si.p, normalize(to_light), 0.01f, length(to_light) - 0.01f, &si);
                    if (!occluded)
                    {
                        // G
                        const float area = length(cross(light.triangle.v1 - light.triangle.v0, light.triangle.v2 - light.triangle.v0)) * 0.5f;
                        //const float cos_theta = fmaxf(dot(light.triangle.n, normalize(to_light)), 0.0f);
                        const float d = length(to_light);
                        const float G = area * nDl / (d * d);
                        weight = nDl * LnDl * area / (d * d);
                    }
                }
                //result += light.emission * brdf * weight;
                //result += r.W * light.emission * brdf * weight;
                result += r.W * light.emission * brdf;

                //printf("r.W: %f, light.emission: %f %f %f, brdf: %f %f %f, weight: %f\n",
                    //r.W, light.emission.x(), light.emission.y(), light.emission.z(), brdf.x(), brdf.y(), brdf.z(), weight);

                // Uniform hemisphere sampling
                si.trace_terminate = true;
                Vec2f u = UniformSampler::get2D(seed);
                Vec3f wi = cosineSampleHemisphere(u[0], u[1]);
                Onb onb(si.shading.n);
                onb.inverseTransform(wi);
                si.wi = normalize(wi);
                si.seed = seed;

                //const Vec3f brdf = optixDirectCall<Vec3f, SurfaceInteraction*, void*, const Vec3f&>(
                //    si.surface_info.callable_id.bsdf, &si, si.surface_info.data, si.p + si.wi);

                //throughput *= brdf;
            }

            ro = si.p;
            rd = si.wi;

            ++depth;
        }
    }

    const uint32_t image_idx = idx.y() * params.width + idx.x();

    // Nan | Inf check
    if (result.x() != result.x()) result.x() = 0.0f;
    if (result.y() != result.y()) result.y() = 0.0f;
    if (result.z() != result.z()) result.z() = 0.0f;

    Vec3f accum = result / static_cast<float>(spl);

    if (frame > 0)
    {
        const float a = 1.0f / static_cast<float>(frame + 1);
        const Vec3f accum_prev = Vec3f(params.accum_buffer[image_idx]);
        accum = lerp(accum_prev, accum, a);
    }

    params.accum_buffer[image_idx] = Vec4f(accum, 1.0f);
    Vec3u color = make_color(accum);
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
    si->surface_info.type = SurfaceType::None;
    si->emission = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
        env->texture.prg_id, si, env->texture.data);
}

extern "C" __device__ void __miss__shadow()
{
    setPayload<0>(0);
}

// Hitgroups ---------------------------------------------------------
// Mesh
extern "C" __device__ void __closesthit__mesh()
{
    const auto* data = (pgHitgroupData*)optixGetSbtDataPointer();
    const auto* mesh = (TriangleMesh::Data*)data->shape_data;
    
    Ray ray = getWorldRay();

    const int prim_id = optixGetPrimitiveIndex();
    const Face face = mesh->faces[prim_id];
    const Vec2f uv = optixGetTriangleBarycentrics();

    const Vec3f p0 = mesh->vertices[face.vertex_id.x()];
    const Vec3f p1 = mesh->vertices[face.vertex_id.y()];
    const Vec3f p2 = mesh->vertices[face.vertex_id.z()];

    const Vec2f texcoord0 = mesh->texcoords[face.texcoord_id.x()];
    const Vec2f texcoord1 = mesh->texcoords[face.texcoord_id.y()];
    const Vec2f texcoord2 = mesh->texcoords[face.texcoord_id.z()];
    const Vec2f texcoord = (1.0f - uv.x() - uv.y()) * texcoord0 + uv.x() * texcoord1 + uv.y() * texcoord2;

    const Vec3f n0 = mesh->normals[face.normal_id.x()];
    const Vec3f n1 = mesh->normals[face.normal_id.y()];
    const Vec3f n2 = mesh->normals[face.normal_id.z()];

    Vec3f local_n = (1.0f - uv.x() - uv.y()) * n0 + uv.x() * n1 + uv.y() * n2;
    Vec3f world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);

    auto* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->shading.uv = texcoord;
    si->surface_info = data->surface_info;

    Vec3f dpdu, dpdv;
    const Vec2f duv02 = texcoord0 - texcoord2, duv12 = texcoord1 - texcoord2;
    const Vec3f dp02 = p0 - p2, dp12 = p1 - p2;
    const float D = duv02.x() * duv12.y() - duv02.y() * duv12.x();
    bool degenerateUV = fabs(D) < 1e-8f;
    if (!degenerateUV)
    {
        const float invD = 1.0f / D;
        dpdu = (duv12.y() * dp02 - duv02.y() * dp12) * invD;
        dpdv = (duv02.x() * dp12 - duv12.x() * dp02) * invD;
    }
    if (degenerateUV || length(cross(dpdu, dpdv)) == 0.0f)
    {
        Onb onb(si->shading.n);
        dpdu = onb.tangent;
        dpdv = onb.bitangent;  
    }
    si->shading.dpdu = normalize(optixTransformNormalFromObjectToWorldSpace(dpdu));
    si->shading.dpdv = normalize(optixTransformNormalFromObjectToWorldSpace(dpdv));
}

extern "C" __device__ void __closesthit__shadow()
{
    setPayload<0>(1);
}

// Surfaces -----------------------------------------------------------------------
// Diffuse
extern "C" __device__ void __direct_callable__sample_diffuse(SurfaceInteraction* si, void* mat_data)
{

}

extern "C" __device__ Vec3f __continuation_callable__brdf_diffuse(SurfaceInteraction * si, void* mat_data, const Vec3f & p)
{
    const auto* diffuse = (Diffuse::Data*)mat_data;
    si->emission = Vec3f(0.0f);

    const Vec3f wi = normalize(p - si->p);
    const Vec3f albedo = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
        diffuse->texture.prg_id, si, diffuse->texture.data);
    si->albedo = albedo;
    const float cos_theta = fmaxf(dot(si->shading.n, wi), 0.0f);
    return albedo * cos_theta * math::inv_pi;
}

extern "C" __device__ float __direct_callable__pdf_diffuse(SurfaceInteraction * si, void* mat_data, const Vec3f& p)
{
    const Vec3f wi = normalize(p - si->p);
    const float cos_theta = fmaxf(dot(si->shading.n, wi), 0.0f);
    return cos_theta * math::inv_pi;
}

// Disney
extern "C" __device__ void __direct_callable__sample_disney(SurfaceInteraction * si, void* mat_data)
{

}

extern "C" __device__ Vec3f __continuation_callable__brdf_disney(SurfaceInteraction* si, void* mat_data, const Vec3f& p)
{
    const auto* disney = (Disney::Data*)mat_data;
    si->emission = Vec3f(0.0f);
    
    const Vec3f V = -normalize(si->wo);
    const Vec3f L = normalize(p - si->p);
    const Vec3f N = si->shading.n;

    const float NdotV = fabs(dot(N, V));
    const float NdotL = fabs(dot(N, L));

    if (NdotV == 0.0f || NdotL == 0.0f)
        return Vec3f(0.0f);

    const Vec3f H = normalize(V + L);
    const float NdotH = dot(N, H);
    const float LdotH = dot(L, H);

    const Vec3f base = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
        disney->base.prg_id, si, disney->base.data);
    si->albedo = base;

    // Diffuse term
    const float Fd90 = 0.5f + 2.0f * disney->roughness * LdotH * LdotH;
    const float FVd90 = fresnelSchlickT(NdotV, Fd90);
    const float FLd90 = fresnelSchlickT(NdotL, Fd90);
    const Vec3f f_diffuse = (base * math::inv_pi) * FVd90 * FLd90;

    // Subsurface
    const float Fss90 = disney->roughness * LdotH * LdotH;
    const float FVss90 = fresnelSchlickT(NdotV, Fss90);
    const float FLss90 = fresnelSchlickT(NdotL, Fss90);
    const Vec3f f_subsurface = (base * math::inv_pi) * 1.25f * (FVss90 * FLss90 * ((1.0f / (NdotV * NdotL)) - 0.5f) + 0.5f);

    // Sheen
    const Vec3f rho_tint = base / luminance(base);
    const Vec3f rho_sheen = lerp(Vec3f(1.0f), rho_tint, disney->sheen_tint);
    const Vec3f f_sheen = disney->sheen * rho_sheen * powf(1.0f - LdotH, 5.0f);

    // Specular
    const Vec3f X = si->shading.dpdu;
    const Vec3f Y = si->shading.dpdv;
    const float alpha = fmaxf(0.001f, disney->roughness);
    const float aspect = sqrtf(1.0f - disney->anisotropic * 0.9f);
    const float ax = fmaxf(0.001f, pow2(alpha) / aspect);
    const float ay = fmaxf(0.001f, pow2(alpha) * aspect);
    const Vec3f rho_specular = lerp(Vec3f(1.0f), rho_tint, disney->specular_tint);
    const Vec3f Fs0 = lerp(0.08f * disney->specular * rho_specular, base, disney->metallic);
    const Vec3f FHs0 = fresnelSchlickR(LdotH, Fs0);
    const float Ds = GTR2_aniso(NdotH, dot(H, X), dot(H, Y), ax, ay);
    float Gs = smithG_GGX_aniso(NdotL, dot(L, X), dot(L, Y), ax, ay);
    Gs *= smithG_GGX_aniso(NdotV, dot(V, X), dot(V, Y), ax, ay);
    const Vec3f f_specular = FHs0 * Fs0 * Gs;

    // Clearcoat
    const float Fcc = fresnelSchlickR(LdotH, 0.04f);
    const float alpha_cc = lerp(0.1f, 0.001f, disney->clearcoat_gloss);
    const float Dcc = GTR1(NdotH, alpha_cc);
    const float Gcc = smithG_GGX(NdotV, 0.25f);
    const Vec3f f_clearcoat = Vec3f(0.25f * disney->clearcoat * (Fcc * Dcc * Gcc));

    const Vec3f out = (1.0f - disney->metallic) * (lerp(f_diffuse, f_subsurface, disney->subsurface) + f_sheen) + f_specular + f_clearcoat;
    return out * fmaxf(NdotL, 0.0f);
}

extern "C" __device__ float __direct_callable__pdf_disney(SurfaceInteraction* si, void* mat_data, const Vec3f& p)
{

}

extern "C" __device__ Vec3f __direct_callable__area_emitter(SurfaceInteraction* si, void* surface_data)
{
    const auto* area = reinterpret_cast<AreaEmitter::Data*>(surface_data);
    si->trace_terminate = true;
    float is_emitted = dot(si->wo, si->shading.n) < 0.0f ? 1.0f : 0.0f;
    if (area->twosided)
    {
        is_emitted = 1.0f;
        si->shading.n = faceforward(si->shading.n, -si->wi, si->shading.n);
    }

    const Vec3f base = optixDirectCall<Vec3f, SurfaceInteraction*, void*>(
        area->texture.prg_id, si, area->texture.data);
    si->albedo = base;
    
    si->emission = base * area->intensity * is_emitted;
    return si->emission;
}

// Textures ----------------------------------------------------------------------
extern "C" __device__ Vec3f __direct_callable__bitmap(SurfaceInteraction* si, void* tex_data)
{
    const auto* bitmap = (BitmapTexture::Data*)tex_data;
    float4 c = tex2D<float4>(bitmap->texture, si->shading.uv.x(), si->shading.uv.y());
    return Vec3f(c);
}

extern "C" __device__ Vec3f __direct_callable__constant(SurfaceInteraction* si, void* tex_data)
{
    const auto* constant = (ConstantTexture::Data*)tex_data;
    return constant->color;
}

extern "C" __device__ Vec3f __direct_callable__checker(SurfaceInteraction* si, void* tex_data)
{
    const auto* checker = (CheckerTexture::Data*)tex_data;
    const Vec2f uv = si->shading.uv;
    const bool is_odd = sinf(uv.x() * math::pi * checker->scale) * sinf(uv.y() * math::pi * checker->scale) < 0.0f;
    return lerp(checker->color1, checker->color2, (float)is_odd);
}