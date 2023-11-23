#include <prayground/prayground.h>
#include "params.h"

#define MY_DEBUG 0

extern "C" { __constant__ LaunchParams params; }

using SurfaceInteraction = SurfaceInteraction_<Vec3f>;

static __forceinline__ __device__ SurfaceInteraction* getSurfaceInteraction()
{
	const uint32_t u0 = getPayload<0>();
	const uint32_t u1 = getPayload<1>();
	return reinterpret_cast<SurfaceInteraction*>(unpackPointer(u0, u1));
}

static __forceinline__ __device__ void trace(
	OptixTraversableHandle handle, const Vec3f& ro, const Vec3f& rd,
	float tmin, float tmax, uint32_t ray_type, SurfaceInteraction* si)
{
	uint32_t u0, u1;
	packPointer(si, u0, u1);
	optixTrace(
		handle, ro, rd,
		tmin, tmax, 0,
		OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE,
		ray_type, 2, ray_type,
		u0, u1
	);
}

// raygen
extern "C" __device__ void __raygen__pinhole()
{
	const pgRaygenData<Camera>* raygen = (pgRaygenData<Camera>*)optixGetSbtDataPointer();

	const int frame = params.frame;

	const Vec3ui idx(optixGetLaunchIndex());
	uint32_t seed = tea<4>(idx.y() * params.width + idx.x(), frame);

	Vec3f result(0.0f);
	Vec3f normal(0.0f);

	int i = params.samples_per_launch;

	while (i > 0)
	{
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
		for (;;)
		{
            if ( depth >= params.max_depth )
				break;

            trace(params.handle, ro, rd, 0.01f, 1e16f, /* ray_type = */ 0, &si);

            if (si.trace_terminate) {
                result += si.emission * throughput;
                break;
            }

            if (depth == 0)
                normal = si.shading.n;

            // Get emission from area emitter
            if ( si.surface_info.type == SurfaceType::AreaEmitter )
            {
                // Evaluating emission from emitter
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.callable_id.bsdf, &si, si.surface_info.data);

                result += si.emission * throughput;
                if (si.trace_terminate)
                    break;
            }
            // Specular sampling
            else if (+(si.surface_info.type & SurfaceType::Delta))
            {
                // Sampling scattered direction
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.callable_id.sample, &si, si.surface_info.data);
                
                // Evaluate bsdf
                Vec3f bsdf_val = optixContinuationCall<Vec3f, SurfaceInteraction*, void*>(
                    si.surface_info.callable_id.bsdf, &si, si.surface_info.data);
                throughput *= bsdf_val;
            }
            // Rough surface sampling with applying MIS
            else if ( +(si.surface_info.type & (SurfaceType::Rough | SurfaceType::Diffuse)) )
            {
                // Importance sampling according to the BSDF
                optixDirectCall<void, SurfaceInteraction*, void*>(
                    si.surface_info.callable_id.sample, &si, si.surface_info.data);

                // Evaluate PDF of area emitter
                float pdf = optixDirectCall<float, SurfaceInteraction*, void*>(
                    si.surface_info.callable_id.pdf, &si, si.surface_info.data);

                // Evaluate BSDF
                Vec3f bsdf = optixContinuationCall<Vec3f, SurfaceInteraction*, void*>(
                    si.surface_info.callable_id.bsdf, &si, si.surface_info.data);

                if (pdf == 0.0f) break;

                throughput *= bsdf / pdf;
            }
            
            ro = si.p;
            rd = si.wi;

            ++depth;
		}
        i--;
	}

    const uint32_t image_idx = idx.y() * params.width + idx.x();

    if (!result.isValid()) result = Vec3f(0.0f);

    Vec3f accum_color = result / static_cast<float>(params.samples_per_launch);

    if (frame > 0)
    {
        const float a = 1.0f / static_cast<float>(frame + 1);
        const Vec3f accum_color_prev(params.accum_buffer[image_idx]);
        accum_color = lerp(accum_color_prev, accum_color, a);
    }
    params.accum_buffer[image_idx] = Vec4f(accum_color, 1.0f);
    Vec3u color = make_color(accum_color);
    params.result_buffer[image_idx] = Vec4u(color, 255);
}

// Miss
extern "C" __device__ void __miss__envmap()
{
    pgMissData* data = reinterpret_cast<pgMissData*>(optixGetSbtDataPointer());
    auto* env = reinterpret_cast<EnvironmentEmitter::Data*>(data->env_data);
    SurfaceInteraction* si = getSurfaceInteraction();

    Ray ray = getWorldRay();

    const float a = dot(ray.d, ray.d);
    const float half_b = dot(ray.o, ray.d);
    const float c = dot(ray.o, ray.o) - 1e8f * 1e8f;
    const float discriminant = half_b * half_b - a * c;

    float sqrtd = sqrtf(discriminant);
    float t = (-half_b + sqrtd) / a;

    Vec3f p = normalize(ray.at(t));

    float phi = atan2(p.z(), p.x());
    float theta = asin(p.y());
    float u = 1.0f - (phi + math::pi) / (2.0f * math::pi);
    float v = 1.0f - (theta + math::pi / 2.0f) * math::inv_pi;
    si->shading.uv = Vec2f(u, v);
    si->trace_terminate = true;
    si->surface_info.type = SurfaceType::None;
    si->emission = optixDirectCall<Vec3f, const Vec2f&, void*>(
        env->texture.prg_id, si->shading.uv, env->texture.data);
}

extern "C" __device__ void __miss__shadow()
{
    setPayload<0>(1);
}

// Hitgroups 
extern "C" __device__ void __intersection__box()
{
    pgHitgroupData* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
    auto* box = reinterpret_cast<Box::Data*>(data->shape_data);
    Ray ray = getLocalRay();
    pgReportIntersectionBox(box, ray);
}

extern "C" __device__ void __intersection__sphere()
{
    pgHitgroupData* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
    auto* sphere = reinterpret_cast<Sphere::Data*>(data->shape_data);
    Ray ray = getLocalRay();
    pgReportIntersectionSphere(sphere, ray);
}

extern "C" __device__ void __intersection__plane()
{
    pgHitgroupData* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
    auto* plane = reinterpret_cast<Plane::Data*>(data->shape_data);
    Ray ray = getLocalRay();
    pgReportIntersectionPlane(plane, ray);
}

extern "C" __device__ void __closesthit__custom()
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

    auto* si = getSurfaceInteraction();

    si->p = ray.at(ray.tmax);
    si->shading = *shading;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->surface_info = data->surface_info;
}

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

extern "C" __device__ void __closesthit__curves()
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

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = optixTransformPointFromObjectToWorldSpace(hit_point);
    si->shading = shading;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->surface_info = data->surface_info;
}

extern "C" __device__ void __closesthit__shadow()
{
    setPayload<0>(0);
}

// Surfaces
// Diffuse
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
    const Vec3f base = optixDirectCall<Vec3f, const Vec2f&, void*>(disney->base.prg_id, si->shading.uv, disney->base.data);
    return pgGetDisneyBRDF(disney, si->wo, si->wi, si->shading, base);
}

extern "C" __device__ float __direct_callable__pdf_disney(SurfaceInteraction * si, void* data)
{
    const Disney::Data* disney = reinterpret_cast<Disney::Data*>(data);
    return pgGetDisneyPDF(disney, si->wo, si->wi, si->shading);
}

// Area emitter
extern "C" __device__ Vec3f __direct_callable__area_emitter(SurfaceInteraction* si, void* data)
{
    const auto* area = reinterpret_cast<AreaEmitter::Data*>(data);
    si->trace_terminate = true;
    float is_emitted = dot(si->wo, si->shading.n) < 0.0f ? 1.0f : 0.0f;
    if (area->twosided)
    {
        is_emitted = 1.0f;
        si->shading.n = faceforward(si->shading.n, -si->wo, si->shading.n);
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