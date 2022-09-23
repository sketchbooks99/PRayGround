#include "util.cuh"
#include <prayground/shape/cuda/shapes.cuh>

extern "C" __device__ void __closesthit__shadow()
{
    setPayload<0>(1);
}

// Get surface interaction on a plane light
extern "C" __device__ LightInteraction __direct_callable__rnd_sample_plane(const AreaEmitterInfo& area_info, SurfaceInteraction* si)
{
    LightInteraction li;
    const Plane::Data* plane = reinterpret_cast<Plane::Data*>(area_info.shape_data);

    const Vec3f corner0(plane->min.x(), 0.0f, plane->min.y());
    const Vec3f corner1(plane->max.x(), 0.0f, plane->min.y());
    const Vec3f corner2(plane->min.x(), 0.0f, plane->max.y());

    const Vec2f uv = UniformSampler::get2D(si->seed);
    const Vec3f rnd_p = Vec3f(
        lerp(plane->min.x(), plane->max.x(), uv[0]),
        0.0f, 
        lerp(plane->min.y(), plane->max.y(), uv[1])
    );
    li.p = area_info.objToWorld.pointMul(rnd_p);
    li.wi = normalize(li.p - si->p);
    li.n = normalize(area_info.objToWorld.normalMul(Vec3f(0, 1, 0)));
    li.uv = uv;

    const Vec3f local_o = area_info.worldToObj.pointMul(si->p);
    const float d2 = lengthSquared(rnd_p - local_o);
    const float area = length(cross(corner1 - corner0, corner2 - corner0));
    li.pdf = d2 / area;

    return li;
}

extern "C" __device__ LightInteraction __direct_callable__rnd_sample_sphere(const AreaEmitterInfo& area_info, SurfaceInteraction* si)
{
    LightInteraction li;
    const Sphere::Data* sphere = reinterpret_cast<Sphere::Data*>(area_info.shape_data);

    // Move surface point to object space
    const Vec3f local_o = area_info.worldToObj.pointMul(si->p);
    Vec3f to_light = sphere->center - local_o;
    const float d2 = lengthSquared(to_light);

    // Sample ray towards a sphere
    Onb onb(normalize(to_light));
    Vec3f rnd_vec = randomSampleToSphere(si->seed, sphere->radius, d2);
    onb.inverseTransform(rnd_vec);

    Ray ray(local_o, rnd_vec, 0.001f, 1e10f);
    Shading shading;
    float time;
    // Get surface information (point, normal, texcoord) on the sphere, and calculate PDF
    if (!intersectionSphere(sphere, ray, &shading, &time))
        li.pdf = 0.0f;
    else
    {
        const float cos_theta_max = sqrtf(1.0f - pow2(sphere->radius) / lengthSquared(sphere->center - local_o));
        const float solid_angle = 2.0f * math::pi * (1.0f - cos_theta_max);
        li.pdf = 1.0f / solid_angle;
    }
    
    // Move to world space
    li.n = normalize(area_info.objToWorld.normalMul(shading.n));
    li.p = area_info.objToWorld.pointMul(ray.at(time));
    li.wi = normalize(area_info.objToWorld.vectorMul(rnd_vec));

    return li;
}

// Custom shape -------------------------------------------------------------------------------
extern "C" DEVICE void __closesthit__custom()
{
    const pgHitgroupData* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
    Ray ray = getWorldRay();

    // Get shading infomation from intersection test implmented in prayground library 
    // ref: prayground/shape/cuda/shapes.cuh
    Shading* shading = getPtrFromTwoAttributes<Shading, 0>();

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->shading = *shading;
    si->shading.n = normalize(optixTransformNormalFromObjectToWorldSpace(si->shading.n));
    si->shading.dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(si->shading.dpdu));
    si->shading.dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(si->shading.dpdv));
    si->t = ray.tmax;
    si->wo = ray.d;
    si->surface_info = data->surface_info;
}


// Triangle mesh -------------------------------------------------------------------------------
extern "C" __device__ void __closesthit__mesh()
{
    pgHitgroupData* data = reinterpret_cast<pgHitgroupData*>(optixGetSbtDataPointer());
    const TriangleMesh::Data* mesh_data = reinterpret_cast<TriangleMesh::Data*>(data->shape_data);

    Ray ray = getWorldRay();
    
    const int prim_id = optixGetPrimitiveIndex();
    const Face face = mesh_data->faces[prim_id];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    const Vec3f p0 = mesh_data->vertices[face.vertex_id.x()];
    const Vec3f p1 = mesh_data->vertices[face.vertex_id.y()];
    const Vec3f p2 = mesh_data->vertices[face.vertex_id.z()];

    const Vec2f texcoord0 = mesh_data->texcoords[face.texcoord_id.x()];
    const Vec2f texcoord1 = mesh_data->texcoords[face.texcoord_id.y()];
    const Vec2f texcoord2 = mesh_data->texcoords[face.texcoord_id.z()];
    const Vec2f texcoords = (1-u-v)*texcoord0 + u*texcoord1 + v*texcoord2;

    const Vec3f n0 = mesh_data->normals[face.normal_id.x()];
	const Vec3f n1 = mesh_data->normals[face.normal_id.y()];
	const Vec3f n2 = mesh_data->normals[face.normal_id.z()];

    // Linear interpolation of normal by barycentric coordinates.
    Vec3f local_n = (1.0f-u-v)*n0 + u*n1 + v*n2;
    Vec3f world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->shading.uv = texcoords;
    si->surface_info = data->surface_info;

    // Calculate partial derivative on texture coordinates
    Vec3f dpdu, dpdv;
    const Vec2f duv02 = texcoord0 - texcoord2, duv12 = texcoord1 - texcoord2;
    const Vec3f dp02 = p0 - p2, dp12 = p1 - p2;
    const float D = duv02.x() * duv12.y() - duv02.y() * duv12.x();
    bool degenerateUV = abs(D) < 1e-8f;
    if (!degenerateUV)
    {
        const float invD = 1.0f / D;
        dpdu = (duv12.y() * dp02 - duv02.y() * dp12) * invD;
        dpdv = (-duv12.x() * dp02 + duv02.x() * dp12) * invD;
    }
    if (degenerateUV || length(cross(dpdu, dpdv)) == 0.0f)
    {
        const Vec3f n = normalize(cross(p2 - p0, p1 - p0));
        Onb onb(n);
        dpdu = onb.tangent;
        dpdv = onb.bitangent;
    }
    si->shading.dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(dpdu));
    si->shading.dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(dpdv));
}