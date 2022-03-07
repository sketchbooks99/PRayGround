#include <prayground/prayground.h>
#include "params.h"
#include <prayground/ext/nanovdb/util/Ray.h>
#include <prayground/ext/nanovdb/util/HDDA.h>

extern "C" { __constant__ LaunchParams params; }

using SurfaceInteraction = SurfaceInteraction_<Vec3f>;

static __forceinline__ __device__ SurfaceInteraction* getSurfaceInteraction()
{
    const unsigned int u0 = getPayload<0>();
    const unsigned int u1 = getPayload<1>();
    return reinterpret_cast<SurfaceInteraction*>(unpackPointer(u0, u1));
}

static __forceinline__ __device__ void trace(
    OptixTraversableHandle handle,
    const Vec3f& ro, const Vec3f& rd,
    float tmin, float tmax,
    unsigned int ray_type,
    SurfaceInteraction* si
)
{
    unsigned int u0, u1;
    packPointer(si, u0, u1);
    optixTrace(
        handle,
        ro.toCUVec(), rd.toCUVec(),
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
    Vec3f& ro, Vec3f& rd)
{
    rd = normalize(x * camera.U + y * camera.V + camera.W);
    ro = camera.origin;
}

extern "C" __device__ void __raygen__medium()
{
    const RaygenData* raygen = reinterpret_cast<RaygenData*>(optixGetSbtDataPointer());

    const int frame = params.frame;
    const Vec3ui idx = optixGetLaunchIndex();
    unsigned int seed = tea<4>(idx.x() * params.width + idx.y(), frame);

    Vec3f result = Vec3f(0.0f);
    Vec3f albedo = Vec3f(0.0f);

    int i = params.samples_per_launch;

    do
    {
        const Vec2f subpixel_jitter = Vec2f(rnd(seed) - 0.5f, rnd(seed) - 0.5f);

        const Vec2f d = 2.0f * Vec2f(
            (static_cast<float>(idx.x()) + subpixel_jitter.x()) / static_cast<float>(params.width),
            (static_cast<float>(idx.y()) + subpixel_jitter.y()) / static_cast<float>(params.height)
        ) - 1.0f;

        Vec3f ro, rd;
        getCameraRay(raygen->camera, d.x(), d.y(), ro, rd);

        Vec3f throughput = Vec3f(1.0f);

        SurfaceInteraction si;
        si.seed = seed;
        si.emission = Vec3f(0.0f);
        si.albedo = Vec3f(0.0f);
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
                Vec3f bsdf_val = optixContinuationCall<Vec3f, SurfaceInteraction*, void*>(
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
                Vec3f bsdf_val = optixContinuationCall<Vec3f, SurfaceInteraction*, void*>(
                    si.surface_info.bsdf_id,
                    &si,
                    si.surface_info.data
                    );

                throughput *= bsdf_val / bsdf_pdf;
            }

            ro = si.p;
            rd = si.wi;

            ++depth;
        }
    } while (--i);

    const uint32_t image_index = idx.y() * params.width + idx.x();

    if (result.x() != result.x()) result.x() = 0.0f;
    if (result.y() != result.y()) result.y() = 0.0f;
    if (result.z() != result.z()) result.z() = 0.0f;

    Vec3f accum_color = result / static_cast<float>(params.samples_per_launch);

    if (frame > 0)
    {
        const float a = 1.0f / static_cast<float>(frame + 1);
        const Vec3f accum_color_prev = Vec3f(params.accum_buffer[image_index]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }
    params.accum_buffer[image_index] = Vec4f(accum_color, 1.0f);
    Vec3u color = make_color(accum_color, true);
    params.result_buffer[image_index] = Vec4u(color.x(), color.y(), color.z(), 255);
}

/* Miss function */
extern "C" __device__ void __miss__envmap()
{
    MissData* data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
    EnvironmentEmitter::Data* env = reinterpret_cast<EnvironmentEmitter::Data*>(data->env_data);
    SurfaceInteraction* si = getSurfaceInteraction();

    Ray ray = getWorldRay();
    const float lambda = __int_as_float(getPayload<2>());

    const float a = dot(ray.d, ray.d);
    const float half_b = dot(ray.o, ray.d);
    const float c = dot(ray.o, ray.o) - 1e8f*1e8f;
    const float D = half_b * half_b - a * c;

    float sqrtD = sqrtf(D);
    float t = (-half_b + sqrtD) / a;

    const Vec3f p = normalize(ray.at(t));

    const float phi = atan2(p.z(), p.x());
    const float theta = asin(p.y());
    const float u = 1.0f - (phi + math::pi) / (2.0f * math::pi);
    const float v = 1.0f - (theta + math::pi / 2.0f) / math::pi;
    si->uv = Vec2f(u, v);
    si->trace_terminate = true;
    si->surface_info.type = SurfaceType::None;
    si->emission = optixDirectCall<Vec3f, const Vec2f&, void*>(
        env->texture.prg_id, si->uv, env->texture.data
    );
    si->albedo = si->emission;
}

/* Material functions */
extern "C" __device__ Vec3f __direct_callable__sample_diffuse(SurfaceInteraction* si, void* mat_data, float& pdf) {
    const Diffuse::Data* diffuse = reinterpret_cast<Diffuse::Data*>(mat_data);

    if (diffuse->twosided)
        si->shading.n = faceforward(si->shading.n, -si->wo, si->shading.n);

    // Sampling scattering direction 
    si->trace_terminate = false;
    uint32_t seed = si->seed;
    const float z0 = rnd(seed);
    const float z1 = rnd(seed);
    Vec3f wi = cosineSampleHemisphere(z0, z1);
    Onb onb(si->shading.n);
    onb.inverseTransform(wi);
    si->wi = normalize(wi);
    si->seed = seed;

    // Get albedo from texture
    const Vec3f albedo = optixDirectCall<Vec3f, const Vec2f&, void*>(
        diffuse->texture.prg_id, si->uv, diffuse->texture.data);
    si->albedo = albedo;
    si->emission = Vec3f(0.0f);

    const float cosine = fmaxf(0.0f, dot(si->shading.n, si->wi));
    pdf = cosine / math::pi;
    return albedo * (cosine / math::pi);
}

extern "C" __device__ Vec3f __direct_callable__sample_medium(SurfaceInteraction* si, void* mat_data, float& pdf)
{
    const VDBGrid::Data* grid = reinterpret_cast<const VDBGrid::Data*>(mat_data);
    const float g = grid->g;
    const Vec3f sigma_s = grid->sigma_s;
    const float sigma_t = grid->sigma_t;
    si->trace_terminate = false;
    uint32_t seed = si->seed;
    Vec2f u = UniformSampler::get2D(seed);

    // Sampling scattering direction from Henyey--Greenstein phase function
    Vec3f wi = sampleHenyeyGreenstein(u, g);
    Onb onb(si->wo);
    onb.inverseTransform(wi);
    si->wi = wi;

    si->albedo = sigma_s;
    const float d = sigma_t * si->t;
    const float Tr = expf(-d);
    pdf = d;
    return sigma_s * Tr / pdf;
}

extern "C" __device__ void __direct_callable__area_emitter(SurfaceInteraction* si, void* surface_data)
{
    const AreaEmitter::Data* area = reinterpret_cast<AreaEmitter::Data*>(surface_data);
    si->trace_terminate = true;
    float is_emitted = 1.0f;
    if (!area->twosided)
        is_emitted = dot(si->wo, si->shading.n) < 0.0f ? 1.0f : 0.0f;
    
    const Vec3f base = optixDirectCall<Vec3f, const Vec2f&, void*>(
        area->texture.prg_id, si->uv, area->texture.data);
    si->albedo = base;
    si->emission = base * area->intensity * is_emitted;
}

/* Hitgroup functions */
extern "C" __device__ void __intersection__plane()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const Plane::Data* plane = reinterpret_cast<Plane::Data*>(data->shape_data);

    const Vec2f min = plane->min;
    const Vec2f max = plane->max;

    Ray ray = getLocalRay();

    const float t = -ray.o.y() / ray.d.y();

    const float x = ray.o.x() + t * ray.d.x();
    const float z = ray.o.z() + t * ray.d.z();

    Vec2f uv((x - min.x()) / (max.x() - min.x()), (z - min.y()) / (max.y() - min.y()));

    if (min.x() < x && x < max.x() && min.y() < z && z < max.y() && ray.tmin < t && t < ray.tmax)
        optixReportIntersection(t, 0, Vec2f_as_ints(uv));
}

extern "C" __device__ void __closesthit__plane()
{
    HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());

    Ray ray = getWorldRay();

    Vec3f local_n = Vec3f(0, 1, 0);
    const Vec3f world_n = normalize(optixTransformNormalFromObjectToWorldSpace(local_n.toCUVec()));
    const Vec2f uv = getVec2fFromAttribute<0>();

    SurfaceInteraction* si = getSurfaceInteraction();

    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->uv = uv;
    si->surface_info = data->surface_info;
    si->shading.dpdu = optixTransformNormalFromObjectToWorldSpace({1.0f, 0.0f, 0.0f});
    si->shading.dpdv = optixTransformNormalFromObjectToWorldSpace({0.0f, 0.0f, 1.0f});
}

static inline __device__ void confine(const nanovdb::BBox<nanovdb::Coord>& bbox, nanovdb::Vec3f& ivec)
{
    auto imin = nanovdb::Vec3f( bbox.min() );
    auto imax = nanovdb::Vec3f( bbox.max() ) + nanovdb::Vec3f( 1.0f );

    if (ivec[0] < imin[0]) ivec[0] = imin[0];
    if (ivec[1] < imin[1]) ivec[1] = imin[1];
    if (ivec[2] < imin[2]) ivec[2] = imin[2];
    if (ivec[0] >= imax[0]) ivec[0] = imax[0] - fmaxf(1.0f, fabsf(ivec[0])) * math::eps;
    if (ivec[1] >= imax[1]) ivec[1] = imax[1] - fmaxf(1.0f, fabsf(ivec[1])) * math::eps;
    if (ivec[2] >= imax[2]) ivec[2] = imax[2] - fmaxf(1.0f, fabsf(ivec[2])) * math::eps;
}

static inline __device__ void confine(const nanovdb::BBox<nanovdb::Coord> &bbox, nanovdb::Vec3f& istart, nanovdb::Vec3f& iend)
{
    confine(bbox, istart);
    confine(bbox, iend);
}

template <typename AccT>
static inline __device__ float transmittanceHDDA(
    const nanovdb::Vec3f& start, 
    const nanovdb::Vec3f& end, 
    AccT& acc)
{
    float transmittance = 1.0f;
    auto dir = end - start;
    auto len = dir.length();
    nanovdb::Ray<float> ray(start, dir / len, 0.0f, len);
    nanovdb::Coord ijk = nanovdb::RoundDown<nanovdb::Coord>(ray.start());

    nanovdb::HDDA<nanovdb::Ray<float>> hdda(ray, acc.getDim(ijk, ray));

    float t = 0.0f;
    float density = acc.getValue(ijk);
    while (hdda.step())
    {
        float dt = hdda.time() - t;
        transmittance *= expf(-density * dt);
        t = hdda.time();
        ijk = hdda.voxel();

        density = acc.getValue(ijk);
        hdda.update(ray, acc.getDim(ijk, ray));
    }

    return transmittance;
}

template <typename AccT>
static inline __device__ float rayMarching(
    const nanovdb::Vec3f& ro, const nanovdb::Vec3f& rd,
    const float tmin, const float tmax, uint32_t& seed, VDBGrid::Data* grid)
{
    const nanovdb::FloatGrid* density = reinterpret_cast<const nanovdb::FloatGrid*>(grid->density);
    const auto& tree = density->tree();
    auto acc = tree.getAccessor();

    nanovdb::Ray<float> ray(ro, rd, tmin, tmax);
    nanovdb::Coord ijk = nanovdb::RoundDown<nanovdb::Coord>(ray.start());

    nanovdb::HDDA<nanovdb::Ray<float>> hdda(ray, acc.getDim(ijk, ray));

    // Sampling optical depth
    float tau_s = -logf(1.0f - rnd(seed));
    // Initialize free-path
    float t = 0.0f;
    float tau = 0.0f;
    float density = acc.getValue(ijk);
    while (tau < tau_s && hdda.step())
    {
        float dt = hdda.time() - t;
        float sigma_e = density * grid->sigma_t;
        tau += sigma_e * dt;
        t = hdda.time();
        ijk = hdda.voxel();

        density = acc.getValue(ijk);
        hdda.update(ray, acc.getDim());
    }
}

extern "C" __device__ void __intersection__grid()
{
    const HitgroupData* data = reinterpret_cast<const HitgroupData*>(optixGetSbtDataPointer());
    const VDBGrid::Data* grid = reinterpret_cast<const VDBGrid::Data*>(data->shape_data);
    const nanovdb::FloatGrid* density = reinterpret_cast<const nanovdb::FloatGrid*>(grid->density);
    assert(density);

    const auto& tree = density->tree();
    auto acc = tree.getAccessor();

    Ray ray = getLocalRay();

    auto bbox = density->indexBBox();
    float t0 = ray.tmin;
    float t1 = ray.tmax;
    auto iRay = nanovdb::Ray<float>(reinterpret_cast<const nanovdb::Vec3f&>(ray.o),
        reinterpret_cast<const nanovdb::Vec3f&>(ray.d), t0, t1);

    if (iRay.intersects(bbox, t0, t1))
    {
        optixSetPayload_2(__float_as_uint(t1));
        optixReportIntersection(fmaxf(t0, ray.tmin), 0);
    }
}

extern "C" __device__ void __closesthit__grid()
{
    const HitgroupData* data = reinterpret_cast<const HitgroupData*>(optixGetSbtDataPointer());
    const VDBGrid::Data* medium = reinterpret_cast<const VDBGrid::Data*>(data->shape_data);
    const nanovdb::FloatGrid* grid = reinterpret_cast<const nanovdb::FloatGrid*>(medium->density);
    const auto& tree = grid->tree();
    auto acc = tree.getAccessor();

    const float sigma_t = medium->sigma_t;

    Ray ray = getWorldRay();
    const float t0 = optixGetRayTmax();
    const float t1 = __int_as_float(getPayload<2>());

    const auto nanoray = nanovdb::Ray<float>(reinterpret_cast<const nanovdb::Vec3f&>(ray.o), 
        reinterpret_cast<const nanovdb::Vec3f&>(ray.d), t0, t1);
    //auto start = grid->worldToIndexF(nanoray(t0));
    //auto end = grid->worldToIndexF(nanoray(t1));

    //auto bbox = grid->indexBBox();
    //confine(bbox, start, end);

    SurfaceInteraction* si = getSurfaceInteraction();

    nanovdb::Coord ijk = nanovdb::RoundDown<nanovdb::Coord>(nanoray.start());
    nanovdb::HDDA<nanovdb::Ray<float>> hdda(nanoray, acc.getDim(ijk, nanoray));

    float t = t0;
    while (hdda.step())
    {
        if (t > t1) break;
        const float density = acc.getValue(ijk);
        if (density > UniformSampler::get1D(si->seed))
        {
            si->t = t;
            si->p = ray.at(t);
            si->shading.n = Vec3f(0,1,0);   // arbitrary
            si->wo = ray.d;
            si->surface_info = data->surface_info;
            si->uv = Vec2f(0.5f);           // arbitrary
            return;
        }

        t = hdda.time();
        hdda.update(nanoray, acc.getDim(ijk, nanoray));
    }

    // No scattering is accepted
    si->t = t1;
    si->p = ray.o;
    si->shading.n = Vec3f(0, 1, 0); 
    si->wo = ray.d;
    si->surface_info.type = SurfaceType::None;
    si->uv = Vec2f(0.5f);
}

/* Texture functions */
extern "C" __device__ Vec3f __direct_callable__bitmap(const Vec2f& uv, void* tex_data)
{
    const BitmapTexture::Data* image = reinterpret_cast<BitmapTexture::Data*>(tex_data);
    Vec4f c = tex2D<float4>(image->texture, uv.x(), uv.y());
    return Vec3f(c.x(), c.y(), c.z());
}

extern "C" __device__ Vec3f __direct_callable__constant(const Vec2f& uv, void* tex_data)
{
    const ConstantTexture::Data* constant = reinterpret_cast<ConstantTexture::Data*>(tex_data);
    return constant->color;
}

extern "C" __device__ Vec3f __direct_callable__checker(const Vec2f& uv, void* tex_data)
{
    const CheckerTexture::Data* checker = reinterpret_cast<CheckerTexture::Data*>(tex_data);
    const bool is_odd = sinf(uv.x() * math::pi * checker->scale) * sinf(uv.y() * math::pi * checker->scale) < 0;
    return is_odd ? checker->color1 : checker->color2;
}



