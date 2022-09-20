#include "util.cuh"

extern "C" __device__ void __closesthit__shadow()
{
    setPayload<0>(1);
}

// Plane -------------------------------------------------------------------------------
static __forceinline__ __device__ bool hitPlane(
    const Plane::Data* plane, const Vec3f& o, const Vec3f& v, const float tmin, const float tmax, SurfaceInteraction& si)
{
    const Vec2f min = plane->min;
    const Vec2f max = plane->max;
    
    const float t = -o.y() / v.y();
    const float x = o.x() + t * v.x();
    const float z = o.z() + t * v.z();

    if (min.x() < x && x < max.x() && min.y() < z && z < max.y() && tmin < t && t < tmax)
    {
        si.shading.uv = Vec2f((x - min.x()) / (max.x() - min.x()), (z - min.y()) / max.y() - min.y());
        si.shading.n = Vec3f(0, 1, 0);
        si.t = t;
        si.p = o + t*v;
        return true;
    }
    return false;
}

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

    Vec2f uv = Vec2f((x - min.x()) / (max.x() - min.x()), (z - min.y()) / (max.y() - min.y()));

    if (min.x() < x && x < max.x() && min.y() < z && z < max.y() && ray.tmin < t && t < ray.tmax)
        optixReportIntersection(t, 0, Vec2f_as_ints(uv));
}

extern "C" __device__ void __closesthit__plane()
{
    HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());

    Ray ray = getWorldRay();

    Vec3f local_n = Vec3f(0, 1, 0);
    Vec3f world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);
    Vec2f uv = getVec2fFromAttribute<0>();

    SurfaceInteraction* si = getSurfaceInteraction();

    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->shading.uv = uv;
    si->surface_info = data->surface_info;

    si->shading.dpdu = optixTransformNormalFromObjectToWorldSpace(Vec3f(1.0f, 0.0f, 0.0f));
    si->shading.dpdv = optixTransformNormalFromObjectToWorldSpace(Vec3f(0.0f, 0.0f, 1.0f));
}

extern "C" __device__ float __continuation_callable__pdf_plane(const AreaEmitterInfo& area_info, const Vec3f& origin, const Vec3f& direction)
{
    const Plane::Data* plane = reinterpret_cast<Plane::Data*>(area_info.shape_data);

    SurfaceInteraction si;
    const Vec3f local_o = area_info.worldToObj.pointMul(origin);
    const Vec3f local_d = area_info.worldToObj.vectorMul(direction);

    if (!hitPlane(plane, local_o, local_d, 0.01f, 1e16f, si))
        return 0.0f;

    const Vec3f corner0 = area_info.objToWorld.pointMul(Vec3f(plane->min.x(), 0.0f, plane->min.y()));
    const Vec3f corner1 = area_info.objToWorld.pointMul(Vec3f(plane->max.x(), 0.0f, plane->min.y()));
    const Vec3f corner2 = area_info.objToWorld.pointMul(Vec3f(plane->min.x(), 0.0f, plane->max.y()));
    si.shading.n = normalize(area_info.objToWorld.vectorMul(si.shading.n));
    const float area = length(cross(corner1 - corner0, corner2 - corner0));
    const float distance_squared = si.t * si.t;
    const float cosine = fabs(dot(si.shading.n, direction));
    if (cosine < math::eps)
        return 0.0f;
    return distance_squared / (cosine * area);
}

// グローバル空間における si.p -> 光源上の点 のベクトルを返す
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

// Sphere -------------------------------------------------------------------------------
static __forceinline__ __device__ Vec2f getSphereUV(const Vec3f& p) {
    float phi = atan2(p.z(), p.x());
    float theta = asin(p.y());
    float u = 1.0f - (phi + math::pi) / (2.0f * math::pi);
    float v = 1.0f - (theta + math::pi / 2.0f) / math::pi;
    return Vec2f(u, v);
}

static __forceinline__ __device__ bool hitSphere(
    const Sphere::Data* sphere, const Vec3f& o, const Vec3f& v, const float tmin, const float tmax, LightInteraction& li)
{
    const Vec3f center = sphere->center;
    const float radius = sphere->radius;

    const Vec3f oc = o - center;
    const float a = dot(v, v);
    const float half_b = dot(oc, v);
    const float c = dot(oc, oc) - radius * radius;
    const float discriminant = half_b * half_b - a * c;

    if (discriminant <= 0.0f) return false;

    const float sqrtd = sqrtf(discriminant);

    float t = (-half_b - sqrtd) / a;
    if (t < tmin || tmax < t)
    {
        t = (-half_b + sqrtd) / a;
        if (t < tmin || tmax < t)
            return false;
    }

    li.p = o + t * v;
    li.n = li.p / radius;
    li.uv = getSphereUV(li.n);
    return true;
}

extern "C" __device__ void __intersection__sphere() {
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const Sphere::Data* sphere = reinterpret_cast<Sphere::Data*>(data->shape_data);

    const Vec3f center = sphere->center;
    const float radius = sphere->radius;

    Ray ray = getLocalRay();

    const Vec3f oc = ray.o - center;
    const float a = dot(ray.d, ray.d);
    const float half_b = dot(oc, ray.d);
    const float c = dot(oc, oc) - radius * radius;
    const float discriminant = half_b * half_b - a * c;

    if (discriminant > 0.0f) {
        float sqrtd = sqrtf(discriminant);
        float t1 = (-half_b - sqrtd) / a;
        bool check_second = true;
        if (t1 > ray.tmin && t1 < ray.tmax) {
            Vec3f normal = normalize((ray.at(t1) - center) / radius);
            check_second = false;
            optixReportIntersection(t1, 0, Vec3f_as_ints(normal));
        }

        if (check_second) {
            float t2 = (-half_b + sqrtd) / a;
            if (t2 > ray.tmin && t2 < ray.tmax) {
                Vec3f normal = normalize((ray.at(t2) - center) / radius);
                optixReportIntersection(t2, 0, Vec3f_as_ints(normal));
            }
        }
    }
}

extern "C" __device__ void __closesthit__sphere() {
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const Sphere::Data* sphere = reinterpret_cast<Sphere::Data*>(data->shape_data);

    Ray ray = getWorldRay();

    Vec3f local_n = getVec3fFromAttribute<0>();
    Vec3f world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->shading.uv = getSphereUV(local_n);
    si->surface_info = data->surface_info;

    // Calculate partial derivative on texture coordinates
    float phi = atan2(local_n.z(), local_n.x());
    if (phi < 0) phi += 2.0f * math::pi;
    const float theta = acos(local_n.y());
    const Vec3f dpdu = Vec3f(-math::two_pi * local_n.z(), 0, math::two_pi * local_n.x());
    const Vec3f dpdv = math::pi * Vec3f(local_n.y() * cos(phi), -sin(theta), local_n.y() * sin(phi));
    si->shading.dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(dpdu));
    si->shading.dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(dpdv));
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

    // Get surface information (point, normal, texcoord) on the sphere, and calculate PDF
    if (!hitSphere(sphere, local_o, rnd_vec, 0.001f, 1e10f, li))
        li.pdf = 0.0f;
    else
    {
        const float cos_theta_max = sqrtf(1.0f - pow2(sphere->radius) / lengthSquared(sphere->center - local_o));
        const float solid_angle = 2.0f * math::pi * (1.0f - cos_theta_max);
        li.pdf = 1.0f / solid_angle;
    }
    
    // Move to world space
    li.n = normalize(area_info.objToWorld.normalMul(li.n));
    li.p = area_info.objToWorld.pointMul(li.p);
    li.wi = normalize(area_info.objToWorld.vectorMul(rnd_vec));

    return li;
}

// Cylinder -------------------------------------------------------------------------------
static INLINE DEVICE Vec2f getCylinderUV(
    const Vec3f& p, const float radius, const float height, const bool hit_disk)
{
    if (hit_disk)
    {
        const float r = sqrtf(p.x()*p.x() + p.z()*p.z()) / radius;
        const float theta = atan2(p.z(), p.x());
        float u = 1.0f - (theta + math::pi/2.0f) / math::pi;
        return Vec2f(u, r);
    } 
    else
    {
        float phi = atan2(p.z(), p.x());
        if (phi < 0.0f) phi += math::two_pi;
        const float u = phi / math::two_pi;
        const float v = (p.y() + height / 2.0f) / height;
        return Vec2f(u, v);
    }
}

extern "C" __device__ void __intersection__cylinder()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const Cylinder::Data* cylinder = reinterpret_cast<Cylinder::Data*>(data->shape_data);

    const float radius = cylinder->radius;
    const float height = cylinder->height;

    Ray ray = getLocalRay();
    SurfaceInteraction* si = getSurfaceInteraction();
    
    const float a = dot(ray.d, ray.d) - ray.d.y() * ray.d.y();
    const float half_b = (ray.o.x() * ray.d.x() + ray.o.z() * ray.d.z());
    const float c = dot(ray.o, ray.o) - ray.o.y() * ray.o.y() - radius*radius;
    const float discriminant = half_b*half_b - a*c;

    if (discriminant > 0.0f)
    {
        const float sqrtd = sqrtf(discriminant);
        const float side_t1 = (-half_b - sqrtd) / a;
        const float side_t2 = (-half_b + sqrtd) / a;

        const float side_tmin = fmin( side_t1, side_t2 );
        const float side_tmax = fmax( side_t1, side_t2 );

        if ( side_tmin > ray.tmax || side_tmax < ray.tmin )
            return;

        const float upper = height / 2.0f;
        const float lower = -height / 2.0f;
        const float y_tmin = fmin( (lower - ray.o.y()) / ray.d.y(), (upper - ray.o.y()) / ray.d.y() );
        const float y_tmax = fmax( (lower - ray.o.y()) / ray.d.y(), (upper - ray.o.y()) / ray.d.y() );

        float t1 = fmax(y_tmin, side_tmin);
        float t2 = fmin(y_tmax, side_tmax);
        if (t1 > t2 || (t2 < ray.tmin) || (t1 > ray.tmax))
            return;
        
        bool check_second = true;
        if (ray.tmin < t1 && t1 < ray.tmax)
        {
            Vec3f P = ray.at(t1);
            bool hit_disk = y_tmin > side_tmin;
            Vec3f normal = hit_disk 
                          ? normalize(P - Vec3f(P.x(), 0.0f, P.z()))   // Hit at disk
                          : normalize(P - Vec3f(0.0f, P.y(), 0.0f));   // Hit at side
            Vec2f uv = getCylinderUV(P, radius, height, hit_disk);
            if (hit_disk)
            {
                const float rHit = sqrtf(P.x()*P.x() + P.z()*P.z());
                si->shading.dpdu = Vec3f(-math::two_pi * P.y(), 0.0f, math::two_pi * P.z());
                si->shading.dpdv = Vec3f(P.x(), 0.0f, P.z()) * radius / rHit;
            }
            else 
            {
                si->shading.dpdu = Vec3f(-math::two_pi * P.z(), 0.0f, math::two_pi * P.x());
                si->shading.dpdv = Vec3f(0.0f, height, 0.0f);
            }
            optixReportIntersection(t1, 0, Vec3f_as_ints(normal), Vec2f_as_ints(uv));
            check_second = false;
        }
        
        if (check_second)
        {
            if (ray.tmin < t2 && t2 < ray.tmax)
            {
                Vec3f P = ray.at(t2);
                bool hit_disk = y_tmax < side_tmax;
                Vec3f normal = hit_disk
                            ? normalize(P - Vec3f(P.x(), 0.0f, P.z()))   // Hit at disk
                            : normalize(P - Vec3f(0.0f, P.y(), 0.0f));   // Hit at side
                Vec2f uv = getCylinderUV(P, radius, height, hit_disk);
                if (hit_disk)
                {
                    const float rHit = sqrtf(P.x()*P.x() + P.z()*P.z());
                    si->shading.dpdu = Vec3f(-math::two_pi * P.y(), 0.0f, math::two_pi * P.z());
                    si->shading.dpdv = Vec3f(P.x(), 0.0f, P.z()) * radius / rHit;
                }
                else 
                {
                    si->shading.dpdu = Vec3f(-math::two_pi * P.z(), 0.0f, math::two_pi * P.x());
                    si->shading.dpdv = Vec3f(0.0f, height, 0.0f);
                }
                optixReportIntersection(t2, 0, Vec3f_as_ints(normal), Vec2f_as_ints(uv));
            }
        }
    }
}

extern "C" __device__ void __closesthit__cylinder()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const Cylinder::Data* cylinder = reinterpret_cast<Cylinder::Data*>(data->shape_data);

    Ray ray = getWorldRay();

    Vec3f local_n = getVec3fFromAttribute<0>();
    Vec2f uv = getVec2fFromAttribute<3>();

    Vec3f world_n = optixTransformNormalFromObjectToWorldSpace(local_n);

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->shading.n = normalize(world_n);
    si->t = ray.tmax;
    si->wo = ray.d;
    si->shading.uv = uv;
    si->surface_info = data->surface_info;

    // dpdu and dpdv are calculated in intersection shader
    si->shading.dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(si->shading.dpdu));
    si->shading.dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(si->shading.dpdv));
}


// Triangle mesh -------------------------------------------------------------------------------
extern "C" __device__ void __closesthit__mesh()
{
    HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
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