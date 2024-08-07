#include "util.cuh"

// Diffuse -----------------------------------------------------------------------------------------------
extern "C" __device__ void __direct_callable__sample_diffuse(SurfaceInteraction* si, void* mat_data) {
    const Diffuse::Data* diffuse = reinterpret_cast<Diffuse::Data*>(mat_data);

    if (diffuse->twosided)
        si->shading.n = faceforward(si->shading.n, -si->wo, si->shading.n);

    si->trace_terminate = false;
    uint32_t seed = si->seed;
    const Vec2f u = UniformSampler::get2D(seed);
    Vec3f wi = cosineSampleHemisphere(u[0], u[1]);
    Onb onb(si->shading.n);
    onb.inverseTransform(wi);
    si->wi = normalize(wi);
    si->seed = seed;
}

extern "C" __device__ Vec3f __continuation_callable__bsdf_diffuse(SurfaceInteraction* si, void* mat_data)
{
    const Diffuse::Data* diffuse = reinterpret_cast<Diffuse::Data*>(mat_data);
    const Vec4f albedo = optixDirectCall<Vec4f, const Vec2f&, void*>(diffuse->texture.prg_id, si->shading.uv, diffuse->texture.data);
    si->albedo = Vec3f(albedo);
    si->emission = Vec3f(0.0f);
    const float cosine = fmaxf(0.0f, dot(si->shading.n, si->wi));
    return si->albedo * cosine * math::inv_pi;
}

extern "C" __device__ float __direct_callable__pdf_diffuse(SurfaceInteraction* si, void* mat_data)
{
    const float cosine = fmaxf(0.0f, dot(si->shading.n, si->wi));
    return cosine * math::inv_pi;
}

// Dielectric --------------------------------------------------------------------------------------------
extern "C" __device__ void __direct_callable__sample_dielectric(SurfaceInteraction* si, void* mat_data) {
    const Dielectric::Data* dielectric = reinterpret_cast<Dielectric::Data*>(mat_data);

    float ni = 1.000292f; // air
    float nt = dielectric->ior;  // ior specified 
    float cosine = dot(si->wo, si->shading.n);
    bool into = cosine < 0;
    Vec3f outward_normal = into ? si->shading.n : -si->shading.n;

    if (!into) swap(ni, nt);

    cosine = fabs(cosine);
    float sine = sqrtf(1.0 - cosine*cosine);
    bool cannot_refract = (ni / nt) * sine > 1.0f;

    float reflect_prob = fresnel(cosine, ni, nt);
    unsigned int seed = si->seed;

    if (cannot_refract || reflect_prob > rnd(seed))
        si->wi = reflect(si->wo, outward_normal);
    else    
        si->wi = refract(si->wo, outward_normal, cosine, ni, nt);
    si->trace_terminate = false;
    si->seed = seed;
}

extern "C" __device__ Vec3f __continuation_callable__bsdf_dielectric(SurfaceInteraction* si, void* mat_data)
{
    const Dielectric::Data* dielectric = reinterpret_cast<Dielectric::Data*>(mat_data);
    si->emission = Vec3f(0.0f);
    Vec4f albedo = optixDirectCall<Vec4f, const Vec2f&, void*>(dielectric->texture.prg_id, si->shading.uv, dielectric->texture.data);
    si->albedo = Vec3f(albedo);
    return si->albedo;
}

extern "C" __device__ float __direct_callable__pdf_dielectric(SurfaceInteraction* si, void* mat_data)
{
    return 1.0f;
}

// Conductor --------------------------------------------------------------------------------------------
extern "C" __device__ void __direct_callable__sample_conductor(SurfaceInteraction* si, void* mat_data) {
    const Conductor::Data* conductor = reinterpret_cast<Conductor::Data*>(mat_data);
    if (conductor->twosided)
        si->shading.n = faceforward(si->shading.n, -si->wo, si->shading.n);

    si->wi = reflect(si->wo, si->shading.n);
    si->trace_terminate = false;
}

extern "C" __device__ Vec3f __continuation_callable__bsdf_conductor(SurfaceInteraction* si, void* mat_data)
{
    const Conductor::Data* conductor = reinterpret_cast<Conductor::Data*>(mat_data);
    si->emission = Vec3f(0.0f);
    Vec4f albedo = optixDirectCall<Vec4f, const Vec2f&, void*>(conductor->texture.prg_id, si->shading.uv, conductor->texture.data);
    si->albedo = Vec3f(albedo);
    return si->albedo;
}

extern "C" __device__ float __direct_callable__pdf_conductor(SurfaceInteraction* si, void* mat_data)
{
    return 1.0f;
}

// Disney BRDF ------------------------------------------------------------------------------------------
extern "C" __device__ void __direct_callable__sample_disney(SurfaceInteraction* si, void* mat_data)
{
    const Disney::Data* disney = reinterpret_cast<Disney::Data*>(mat_data);
    si->wi = pgImportanceSamplingDisney(disney, si->wo, si->shading, si->seed);
    si->trace_terminate = false;
}

extern "C" __device__ Vec3f __continuation_callable__bsdf_disney(SurfaceInteraction* si, void* mat_data)
{   
    const Disney::Data* disney = reinterpret_cast<Disney::Data*>(mat_data);
    const Vec4f base = optixDirectCall<Vec4f, const Vec2f&, void*>(disney->base.prg_id, si->shading.uv, disney->base.data);
    return pgGetDisneyBRDF(disney, si->wo, si->wi, si->shading, Vec3f(base));
}

extern "C" __device__ float __direct_callable__pdf_disney(SurfaceInteraction* si, void* mat_data)
{
    const Disney::Data* disney = reinterpret_cast<Disney::Data*>(mat_data);
    return pgGetDisneyPDF(disney, si->wo, si->wi, si->shading);
}

// Area emitter ------------------------------------------------------------------------------------------
extern "C" __device__ void __direct_callable__area_emitter(SurfaceInteraction* si, void* surface_data)
{
    const AreaEmitter::Data* area = reinterpret_cast<AreaEmitter::Data*>(surface_data);
    si->trace_terminate = true;
    float is_emitted = dot(si->wo, si->shading.n) < 0.0f ? 1.0f : 0.0f;
    if (area->twosided)
    {
        is_emitted = 1.0f;
        si->shading.n = faceforward(si->shading.n, -si->wo, si->shading.n);
    }

    const Vec4f base = optixDirectCall<Vec4f, const Vec2f&, void*>(
        area->texture.prg_id, si->shading.uv, area->texture.data);
    si->albedo = Vec3f(base);
    
    si->emission = si->albedo * area->intensity * is_emitted;
}
