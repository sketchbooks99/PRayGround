#include <optix.h>

#include "../core/pathtracer.h"
#include "random.h"

#include <sutil/vec_math.h>
#include "../core/helpers.h"

/** URGENTTODO:
 * Ray must not be launched from shading program. */

extern "C" {
__constant__ Params params;
}

// -------------------------------------------------------------------------------
static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
{
	// Uniformly sample disk.
	const float r = sqrtf(u1);
	const float phi = 2.0f * M_PIf * u2;
	p.x = r * cosf(phi);
	p.y = r * sinf(phi);

	// Project up to hemisphere
	p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}

// -------------------------------------------------------------------------------
static __forceinline__ __device__ float schlick(float cosine, float ior)
{
	float r0 = (1.0f - ior) / (1.0f + ior);
	r0 = r0 * r0;
	return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
}

// -------------------------------------------------------------------------------
// ref: http://www.sic.shibaura-it.ac.jp/~yaoki/cg/CG2008-10.pdf
static __forceinline__ __device__ float3 refract(float3 r_dir, float3 normal, float etai_over_etat)
{
	float k_f = 1.0f / (sqrtf(pow(etai_over_etat, 2.0f) * powf(length(r_dir), 2.0f) - powf(length(normal + r_dir), 2.0f)));
	return k_f * (normal + r_dir) - normal;
}

 // -------------------------------------------------------------------------------
 static __forceinline__ __device__ float3 traceRadiance(
	OptixTraversableHandle handle,
	float3 ray_origin,
	float3 ray_direction,
	unsigned int depth,
	unsigned int seed
)
{
	RadiancePRD prd;
	prd.depth = depth;
	prd.seed = seed;
	optixTrace(
		handle,
		ray_origin,
		ray_direction,
		0.01f,
		1e16f,
		0.0f,
		OptixVisibilityMask(1),
		OPTIX_RAY_FLAG_NONE,
		RAY_TYPE_RADIANCE,
		RAY_TYPE_COUNT,
		RAY_TYPE_RADIANCE,
		float3_as_args(prd.result),
		reinterpret_cast<unsigned int&>(prd.depth),
		reinterpret_cast<unsigned int&>(prd.seed)
	);

	return prd.result;
}

// -------------------------------------------------------------------------------
static __forceinline__ __device__ bool traceOcclusion(
	OptixTraversableHandle handle,
	float3 ray_origin,
	float3 ray_direction,
	float tmin, float tmax)
{
	unsigned int occluded = 0u;
	optixTrace(
		handle,
		ray_origin,
		ray_direction,
		0.01f, // tmin
		1e16f,  // tmax
		0.0f,	// rayTime
		OptixVisibilityMask(1),
		OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
		RAY_TYPE_OCCLUSION,
		RAY_TYPE_COUNT,
		RAY_TYPE_OCCLUSION,
		occluded);
	return occluded;
}

// Emission -------------------------------------------------------------------------
extern "C" __global__ void __closesthit__radiance__emission()
{
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
	/** TODO: 
	 * Material data should be independently allocated to GPU through sbt, 
	 * and gotten by cuda as like following declaration. 
	 * Emission* emission = (Emission*)optixGetSbtDataPointer(); */
	Emission* emission_data = (Emission*)rt_data->material_ptr;

    const int prim_idx = optixGetPrimitiveIndex();
    const int3 index = rt_data->mesh.indices[prim_idx];
    const float3 ray_dir = optixGetWorldRayDirection();
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

	const float3 n0 = normalize(rt_data->mesh.normals[index.x]);
	const float3 n1 = normalize(rt_data->mesh.normals[index.y]);
	const float3 n2 = normalize(rt_data->mesh.normals[index.z]);

    float3 normal = normalize((1.0f - u - v) * n0 + u * n1 + v * n2);
    normal = faceforward(normal, -ray_dir, normal);
    const float3 hit_point = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;

    RadiancePRD prd = getPRD();

    // float3 emission = emission_data->color;
	float3 emission = make_float3(15.0f);
    float3 result = make_float3(0.0f);
    if(prd.depth < params.max_depth)
    {
        result = emission;
    }
    prd.result = result;
    setPRD(prd);
}

extern "C" __global__ void __closesthit__radiance__diffuse()
{
    HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
	Diffuse* diffuse_data = (Diffuse*)rt_data->material_ptr;

    const int prim_idx = optixGetPrimitiveIndex();
    const int3 index = rt_data->mesh.indices[prim_idx];
    const float3 ray_dir = optixGetWorldRayDirection();
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

	const float3 n0 = normalize(rt_data->mesh.normals[index.x]);
	const float3 n1 = normalize(rt_data->mesh.normals[index.y]);
	const float3 n2 = normalize(rt_data->mesh.normals[index.z]);

	/// MEMO: Failed to allocate material_ptr correctly */
    const float3 diffuse_color = diffuse_data->mat_color;
	// const float3 diffuse_color = make_float3(1.0f, 1.0f, 1.0f);

    float3 normal = normalize((1.0f - u - v) * n0 + u * n1 + v * n2);
    normal = faceforward(normal, -ray_dir, normal);
    const float3 hit_point = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;

    RadiancePRD prd = getPRD();
    float3 result = make_float3(0.0f);
    unsigned int seed = prd.seed;

	float3 w_in = make_float3(0.0f);
	const float z1 = rnd(seed);
	const float z2 = rnd(seed);

    cosine_sample_hemisphere(z1, z2, w_in);
    Onb onb(normal);
	onb.inverse_transform(w_in);
	
	/** TODO: Next Event Estimation
	* Check if surface is occuluded by other surfaces 
	* in a way from surface to the light.
	* 
	* Ray must be launched at once for shadow ray.
	*/

    if(prd.depth < params.max_depth)
    {
        float dist = optixGetRayTmax();
        float weight = 1.0f; // Attenuation weight
        float3 radiance = traceRadiance(     // Recursively get radiance from next hitpoint
            params.handle, 
            hit_point, 
            w_in, 
            prd.depth + 1,
            seed
        );
        result = radiance * weight;
	}
	if(diffuse_data->is_normal)
		prd.result = result * normal;
	else
		prd.result = result * diffuse_color;
    setPRD(prd);
}

extern "C" __global__ void __closesthit__radiance__dielectric()
{
    // Get binded data by application.
	HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
	Dielectric* dielectric_data = (Dielectric*)rt_data->material_ptr;

	const int prim_idx = optixGetPrimitiveIndex();
	const int3 index = rt_data->mesh.indices[prim_idx];
	const float3 ray_dir = optixGetWorldRayDirection();
	const float u = optixGetTriangleBarycentrics().x;
	const float v = optixGetTriangleBarycentrics().y;

	const float3 n0 = normalize(rt_data->mesh.normals[index.x]);
	const float3 n1 = normalize(rt_data->mesh.normals[index.y]);
	const float3 n2 = normalize(rt_data->mesh.normals[index.z]);
    
    const float3 mat_color = dielectric_data->mat_color;
    const float ior = dielectric_data->ior;

	float3 normal = normalize((1.0f - u - v) * n0 + u * n1 + v * n2);
	normal = faceforward(normal, -ray_dir, normal);
	const float3 P = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;
	float3 result = make_float3(0.0f);

	HitType hit_type = (HitType)optixGetHitKind();
	float3 front_hit_point = P, back_hit_point = P;
	if (hit_type & HIT_OUTSIDE_FROM_OUTSIDE || hit_type & HIT_INSIDE_FROM_INSIDE)
	{
		front_hit_point += 0.01f * normal;
		back_hit_point -= 0.01f * normal;
	}
	else
	{
		front_hit_point -= 0.01f * normal;
		back_hit_point += 0.01f * normal;
	}

	float etai_over_etat = optixIsTriangleFrontFaceHit() ? (1.0f / ior) : ior;
	//float etai_over_etat = ior;

	RadiancePRD prd = getPRD();

	float3 w_in = normalize(ray_dir);
	float cos_theta = fminf(dot(-w_in, normal), 1.0f);
	float reflect_prob = schlick(cos_theta, etai_over_etat);

	//prd->emitted = rt_data->emission_color;
	float3 w_out = make_float3(0.0f);
	unsigned int seed = prd.seed;
	if (prd.depth < params.max_depth)
	{
		// refraction
		{
			w_out = refract(w_in, normal, etai_over_etat);

			// I can simply change roughness of glass
			//w_out += make_float3(rnd(seed), rnd(seed), rnd(seed)) * 1.0f - make_float3(0.5f);
			float3 radiance = traceRadiance(params.handle,
				back_hit_point,
				w_out,
				prd.depth + 1,
				seed);
			result += (1.0f - reflect_prob) * radiance;
		}

		// reflection
		{
			w_out = reflect(w_in, normal);
			float3 radiance = traceRadiance(params.handle,
				front_hit_point,
				w_out,
				prd.depth + 1,
				seed);
			result += reflect_prob * radiance;
		}
	}
	//prd.result = result * mat_color;
	prd.result = result * mat_color;
	//prd.result = make_float3(1.0f, 0.0f, 0.0f);
	setPRD(prd);
}

extern "C" __global__ void __closesthit__radiance__metal()
{
    // Get binded data by application.
	HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();
	Metal* metal_data = (Metal*)rt_data->material_ptr;

	const int prim_idx = optixGetPrimitiveIndex();
	const int3 index = rt_data->mesh.indices[prim_idx];
	const float3 ray_dir = optixGetWorldRayDirection();
	const float u = optixGetTriangleBarycentrics().x;
	const float v = optixGetTriangleBarycentrics().y;
    const int vert_idx_offset = prim_idx * 3;
    
    const float3 mat_color = metal_data->mat_color;
    const float reflection = metal_data->reflection;

	const float3 n0 = normalize(rt_data->mesh.normals[index.x]);
	const float3 n1 = normalize(rt_data->mesh.normals[index.y]);
	const float3 n2 = normalize(rt_data->mesh.normals[index.z]);
    
    float3 N = normalize((1.0f - u - v) * n0 + u * n1 + v * n2);
    const float3 hit_point = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;
    float3 result = make_float3(0.0f);

	RadiancePRD prd = getPRD();

    unsigned int seed = prd.seed;
    float3 w_in = normalize(ray_dir);
    if(prd.depth < params.max_depth)
    {
        float rnd_offset = (1.0f - reflection) - 0.5f * (1.0f - reflection);
        float3 w_out = reflect(w_in, N);
        w_out += make_float3(rnd(seed), rnd(seed), rnd(seed)) * rnd_offset; 
        float3 radiance = traceRadiance(params.handle,
            hit_point,
            w_out,
            prd.depth + 1,
            seed
        );
        result = radiance * reflection * mat_color;
    }
    prd.result = result;
    setPRD(prd);
}