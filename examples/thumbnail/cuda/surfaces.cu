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
    si->radiance_evaled = false;
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
    si->radiance_evaled = false;
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

    if (disney->twosided)
        si->shading.n = faceforward(si->shading.n, -si->wo, si->shading.n);

    unsigned int seed = si->seed;
    const Vec2f u = UniformSampler::get2D(seed);
    const float diffuse_ratio = 0.5f * (1.0f - disney->metallic);
    Onb onb(si->shading.n);

    if (rnd(seed) < diffuse_ratio)
    {
        Vec3f wi = cosineSampleHemisphere(u[0], u[1]);
        onb.inverseTransform(wi);
        si->wi = normalize(wi);
    }
    else
    {
        float gtr2_ratio = 1.0f / (1.0f + disney->clearcoat);
        Vec3f h;
        const float alpha = fmaxf(0.001f, disney->roughness);
        /// For debugging microfacet sampling according to this cite @ref: https://jcgt.org/published/0007/04/01/
        //const float aspect = sqrtf(1.0f - disney->anisotropic * 0.9f);
        //const float ax = fmaxf(0.001f, math::sqr(alpha) / aspect);
        //const float ay = fmaxf(0.001f, math::sqr(alpha) * aspect);
        if (rnd(seed) < gtr2_ratio)
             h = sampleGGX(u[0], u[1], alpha);
            //h = sampleGGXAniso(-si->wo, ax, ay, z1, z2);
        else
            h = sampleGTR1(u[0], u[1], alpha);
        onb.inverseTransform(h);
        si->wi = normalize(reflect(si->wo, h));
    }
    si->radiance_evaled = false;
    si->trace_terminate = false;
    si->seed = seed;
}

/**
 * @ref: https://rayspace.xyz/CG/contents/Disney_principled_BRDF/
 * 
 * @note 
 * ===== Prefix =====
 * F : fresnel 
 * f : brdf function
 * G : geometry function
 * D : normal distribution function
 */
extern "C" __device__ Vec3f __continuation_callable__bsdf_disney(SurfaceInteraction* si, void* mat_data)
{   
    const Disney::Data* disney = reinterpret_cast<Disney::Data*>(mat_data);
    si->emission = Vec3f(0.0f);

    const Vec3f V = -normalize(si->wo);
    const Vec3f L = normalize(si->wi);
    const Vec3f N = normalize(si->shading.n);

    const float NdotV = dot(N, V);
    const float NdotL = dot(N, L);

    if (NdotV <= 0.0f || NdotL <= 0.0f)
        return Vec3f(0.0f);

    const Vec3f H = normalize(V + L);
    const float NdotH = dot(N, H);
    const float LdotH /* = VdotH */ = dot(L, H);

    Vec4f tmp = optixDirectCall<Vec4f, const Vec2f&, void*>(
        disney->base.prg_id, si->shading.uv, disney->base.data);
    const Vec3f base_color = Vec3f(tmp);
    si->albedo = base_color;

    // Diffuse term (diffuse, subsurface, sheen) ======================
    // Diffuse
    const float Fd90 = 0.5f + 2.0f * disney->roughness * LdotH*LdotH;
    const float FVd90 = fresnelSchlickT(NdotV, Fd90);
    const float FLd90 = fresnelSchlickT(NdotL, Fd90);
    const Vec3f f_diffuse = (base_color / math::pi) * FVd90 * FLd90;

    // Subsurface
    const float Fss90 = disney->roughness * LdotH*LdotH;
    const float FVss90 = fresnelSchlickT(NdotV, Fss90);
    const float FLss90 = fresnelSchlickT(NdotL, Fss90); 
    const Vec3f f_subsurface = (base_color / math::pi) * 1.25f * (FVss90 * FLss90 * ((1.0f / (NdotV * NdotL)) - 0.5f) + 0.5f);

    // Sheen
    const Vec3f rho_tint = base_color / luminance(base_color);
    const Vec3f rho_sheen = lerp(Vec3f(1.0f), rho_tint, disney->sheen_tint);
    const Vec3f f_sheen = disney->sheen * rho_sheen * powf(1.0f - LdotH, 5.0f);

    // Specular term (specular, clearcoat) ============================
    // Spcular
    const Vec3f X = si->shading.dpdu;
    const Vec3f Y = si->shading.dpdv;
    const float alpha = fmaxf(0.001f, disney->roughness);
    const float aspect = sqrtf(1.0f - disney->anisotropic * 0.9f);
    const float ax = fmaxf(0.001f, pow2(alpha) / aspect);
    const float ay = fmaxf(0.001f, pow2(alpha) * aspect);
    const Vec3f rho_specular = lerp(Vec3f(1.0f), rho_tint, disney->specular_tint);
    const Vec3f Fs0 = lerp(0.08f * disney->specular * rho_specular, base_color, disney->metallic);
    const Vec3f FHs0 = fresnelSchlickR(LdotH, Fs0);
    const float Ds = GTR2_aniso(NdotH, dot(H, X), dot(H, Y), ax, ay);
    float Gs = smithG_GGX_aniso(NdotL, dot(L, X), dot(L, Y), ax, ay);
    Gs *= smithG_GGX_aniso(NdotV, dot(V, X), dot(V, Y), ax, ay);
    const Vec3f f_specular = FHs0 * Ds * Gs;

    // Clearcoat
    const float Fcc = fresnelSchlickR(LdotH, 0.04f);
    const float alpha_cc = lerp(0.1f, 0.001f, disney->clearcoat_gloss);
    const float Dcc = GTR1(NdotH, alpha_cc);
    const float Gcc = smithG_GGX(NdotV, 0.25f);
    const Vec3f f_clearcoat = Vec3f( 0.25f * disney->clearcoat * (Fcc * Dcc * Gcc) );

    const Vec3f out = ( 1.0f - disney->metallic ) * ( lerp( f_diffuse, f_subsurface, disney->subsurface ) + f_sheen ) + f_specular + f_clearcoat;
    return out * clamp(NdotL, 0.0f, 1.0f);
}

/**
 * @ref http://simon-kallweit.me/rendercompo2015/report/#adaptivesampling
 * 
 * @todo Investigate correct evaluation of PDF.
 */
extern "C" __device__ float __direct_callable__pdf_disney(SurfaceInteraction* si, void* mat_data)
{
    const Disney::Data* disney = reinterpret_cast<Disney::Data*>(mat_data);

    const Vec3f V = -si->wo;
    const Vec3f L = si->wi;
    const Vec3f N = si->shading.n;

    const float diffuse_ratio = 0.5f * (1.0f - disney->metallic);
    const float specular_ratio = 1.0f - diffuse_ratio;

    const float NdotL = dot(N, L);
    const float NdotV = dot(N, V);

    if (NdotL <= 0.0f || NdotV <= 0.0f)
        return 1.0f;

    const float alpha = fmaxf(0.001f, disney->roughness);
    const float alpha_cc = lerp(0.1f, 0.001f, disney->clearcoat_gloss);
    const Vec3f H = normalize(V + L);
    const float NdotH = abs(dot(H, N));

    const float pdf_Ds = GTR2(NdotH, alpha);
    const float pdf_Dcc = GTR1(NdotH, alpha_cc);
    const float ratio = 1.0f / (1.0f + disney->clearcoat);
    const float pdf_specular = (pdf_Dcc + ratio * (pdf_Ds - pdf_Dcc));
    const float pdf_diffuse = NdotL * math::inv_pi;

    return diffuse_ratio * pdf_diffuse + specular_ratio * pdf_specular;
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
