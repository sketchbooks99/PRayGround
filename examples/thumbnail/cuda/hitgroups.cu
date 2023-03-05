#include "util.cuh"

extern "C" __device__ void __closesthit__shadow()
{
    setPayload<0>(1u);
}

// Box -------------------------------------------------------------------------------
static INLINE DEVICE Vec2f getBoxUV(const Vec3f& p, const Vec3f& min, const Vec3f& max, const int axis)
{
    int u_axis = (axis + 1) % 3;
    int v_axis = (axis + 2) % 3;

    // axisがYの時は (u: Z, v: X) -> (u: X, v: Z)へ順番を変える
    if (axis == 1) swap(u_axis, v_axis);

    Vec2f uv((p[u_axis] - min[u_axis]) / (max[u_axis] - min[u_axis]), (p[v_axis] - min[v_axis]) / (max[v_axis] - min[v_axis]));

    return clamp(uv, 0.0f, 1.0f);
}

// Return hit axis
static INLINE DEVICE int hitBox(
    const Box::Data* box, const Vec3f& o, const Vec3f& v,
    const float tmin, const float tmax, SurfaceInteraction& si)
{
    Vec3f min = box->min;
    Vec3f max = box->max;

    float _tmin = tmin, _tmax = tmax;
    int min_axis = -1, max_axis = -1;

    for (int i = 0; i < 3; i++)
    {
        float t0, t1;
        if (getByIndex(v, i) == 0.0f)
        {
            t0 = fminf(min[i] - o[i], max[i] - o[i]);
            t1 = fmaxf(min[i] - o[i], max[i] - o[i]);
        }
        else
        {
            t0 = fminf((min[i] - o[i]) / v[i], (max[i] - o[i]) / v[i]);
            t1 = fmaxf((min[i] - o[i]) / v[i], (max[i] - o[i]) / v[i]);
        }
        min_axis = t0 > _tmin ? i : min_axis;
        max_axis = t1 < _tmax ? i : max_axis;

        _tmin = fmaxf(t0, _tmin);
        _tmax = fminf(t1, _tmax);

        if (_tmax < _tmin)
            return -1;
    }

    Vec3f center = (min + max) / 2.0f;
    if ((tmin < _tmin && _tmin < tmax) && (-1 < min_axis && min_axis < 3))
    {
        Vec3f p = o + _tmin * v;
        Vec3f center_axis = p;
        center_axis[min_axis] = center[min_axis];
        Vec3f normal = normalize(p - center_axis);
        Vec2f uv = getBoxUV(p, min, max, min_axis);
        si.p = p;
        si.shading.n = normal;
        si.shading.uv = uv;
        si.t = _tmin;
        return min_axis;
    }

    if ((tmin < _tmax && _tmax < tmax) && (-1 < max_axis && max_axis < 3))
    {
        Vec3f p = o + _tmax * v;
        Vec3f center_axis = p;
        center_axis[max_axis] = center[max_axis];
        Vec3f normal = normalize(p - center_axis);
        Vec2f uv = getBoxUV(p, min, max, max_axis);
        si.p = p;
        si.shading.n = normal;
        si.shading.uv = uv;
        si.t = _tmax;
        return max_axis;
    }
    return -1;
}

extern "C" __device__ void __intersection__box()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const int prim_id = optixGetPrimitiveIndex();
    Box::Data box = reinterpret_cast<Box::Data*>(data->shape_data)[prim_id];

    Ray ray = getLocalRay();

    SurfaceInteraction si;
    int hit_axis = hitBox(&box, ray.o, ray.d, ray.tmin, ray.tmax, si);
    if (hit_axis >= 0) {
        optixReportIntersection(si.t, 0, Vec3f_as_ints(si.shading.n), Vec2f_as_ints(si.shading.uv), hit_axis);
    }
}

extern "C" __device__ void __closesthit__box()
{
    HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());

    Ray ray = getWorldRay();

    Vec3f local_n = getVec3fFromAttribute<0>();
    Vec2f uv = getVec2fFromAttribute<3>();
    uint32_t hit_axis = getAttribute<5>();

    Vec3f world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);

    SurfaceInteraction* si = getSurfaceInteraction();

    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->shading.uv = uv;
    si->surface_info = data->surface_info;

    Vec3f dpdu, dpdv;
    // x
    if (hit_axis == 0)
    {
        dpdu = Vec3f(0.0f, 0.0f, 1.0f);
        dpdv = Vec3f(0.0f, 1.0f, 0.0f);
    }
    else if (hit_axis == 1)
    {
        dpdu = Vec3f(1.0f, 0.0f, 0.0f);
        dpdv = Vec3f(0.0f, 0.0f, 1.0f);
    }
    else if (hit_axis == 2)
    {
        dpdu = Vec3f(1.0f, 0.0f, 0.0f);
        dpdv = Vec3f(0.0f, 1.0f, 0.0f);
    }
    si->shading.dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(dpdu));
    si->shading.dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(dpdv));
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
    const float c = dot(ray.o, ray.o) - ray.o.y() * ray.o.y() - radius * radius;
    const float discriminant = half_b * half_b - a*c;

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
        si.shading.uv = Vec2f((x - min.x()) / (max.x() - min.x()), (z - min.y()) / (max.y() - min.y()));
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

    Vec3f n = Vec3f(0, 1, 0);

    if (min.x() < x && x < max.x() && min.y() < z && z < max.y() && ray.tmin < t && t < ray.tmax)
        optixReportIntersection(t, 0, Vec3f_as_ints(n), Vec2f_as_ints(uv));
}

extern "C" __device__ void __closesthit__plane()
{
    HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    
    Ray ray = getWorldRay();

    Vec3f local_n = getVec3fFromAttribute<0>();
    Vec3f world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);
    Vec2f uv = getVec2fFromAttribute<3>();

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

extern "C" __device__ float __continuation_callable__pdf_plane(
    const AreaEmitterInfo& area_info, const Vec3f& origin, const Vec3f& direction)
{
    const Plane::Data* plane_data = reinterpret_cast<Plane::Data*>(area_info.shape_data);

    SurfaceInteraction si;
    const Vec3f local_o = area_info.worldToObj.pointMul(origin);
    const Vec3f local_d = area_info.worldToObj.vectorMul(direction);

    if (!hitPlane(plane_data, local_o, local_d, 0.01f, 1e16f, si))
        return 0.0f;

    const Vec3f corner0 = area_info.objToWorld.pointMul(Vec3f(plane_data->min.x(), 0.0f, plane_data->min.y()));
    const Vec3f corner1 = area_info.objToWorld.pointMul(Vec3f(plane_data->max.x(), 0.0f, plane_data->min.y()));
    const Vec3f corner2 = area_info.objToWorld.pointMul(Vec3f(plane_data->min.x(), 0.0f, plane_data->max.y()));
    si.shading.n = normalize(area_info.objToWorld.vectorMul(si.shading.n));
    const float area = length(cross(corner1 - corner0, corner2 - corner0));
    const float distance_squared = si.t * si.t;
    const float cosine = fabs(dot(si.shading.n, direction));
    if (cosine < math::eps)
        return 0.0f;
    return distance_squared / (cosine * area);
}

// グローバル空間における si.p -> 光源上の点 のベクトルを返す
extern "C" __device__ Vec3f __direct_callable__rnd_sample_plane(const AreaEmitterInfo& area_info, SurfaceInteraction * si)
{
    const Plane::Data* plane_data = reinterpret_cast<Plane::Data*>(area_info.shape_data);
    // サーフェスの原点をローカル空間に移す
    const Vec3f local_p = area_info.worldToObj.pointMul(si->p);
    unsigned int seed = si->seed;
    // 平面光源上のランダムな点を取得
    const Vec3f rnd_p(rnd(seed, plane_data->min.x(), plane_data->max.x()), 0.0f, rnd(seed, plane_data->min.y(), plane_data->max.y()));
    Vec3f to_light = rnd_p - local_p;
    to_light = area_info.objToWorld.vectorMul(to_light);
    si->seed = seed;
    return to_light;
}

// Sphere -------------------------------------------------------------------------------
static __forceinline__ __device__ bool hitSphere(
    const Sphere::Data* sphere, const Vec3f& o, const Vec3f& v, const float tmin, const float tmax, SurfaceInteraction& si)
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

    si.t = t;
    si.p = o + t * v;
    si.shading.n = si.p / radius;
    si.shading.uv = pgGetSphereUV(si.shading.n);
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
            const Vec2f uv = pgGetSphereUV(normal);
            check_second = false;
            optixReportIntersection(t1, 0, Vec3f_as_ints(normal), Vec2f_as_ints(uv));
        }

        if (check_second) {
            float t2 = (-half_b + sqrtd) / a;
            if (t2 > ray.tmin && t2 < ray.tmax) {
                Vec3f normal = normalize((ray.at(t2) - center) / radius);
                const Vec2f uv = pgGetSphereUV(normal);
                optixReportIntersection(t2, 0, Vec3f_as_ints(normal), Vec2f_as_ints(uv));
            }
        }
    }
}

extern "C" __device__ void __closesthit__sphere() {
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const Sphere::Data* sphere = reinterpret_cast<Sphere::Data*>(data->shape_data);

    Ray ray = getWorldRay();

    Vec3f local_n = getVec3fFromAttribute<0>();
    Vec2f uv = getVec2fFromAttribute<3>();
    Vec3f world_n = optixTransformNormalFromObjectToWorldSpace(local_n);
    world_n = normalize(world_n);

    SurfaceInteraction* si = getSurfaceInteraction();
    si->p = ray.at(ray.tmax);
    si->shading.n = world_n;
    si->t = ray.tmax;
    si->wo = ray.d;
    si->shading.uv = uv;
    si->surface_info = data->surface_info;

    // Calculate partial derivative on texture coordinates
    float phi = atan2(local_n.z(), local_n.x());
    if (phi < 0) phi += math::two_pi;
    const float theta = acos(local_n.y());
    const Vec3f dpdu = Vec3f(-math::two_pi * local_n.z(), 0, math::two_pi * local_n.x());
    const Vec3f dpdv = math::pi * Vec3f(local_n.y() * cos(phi), -sin(theta), local_n.y() * sin(phi));
    si->shading.dpdu = normalize(optixTransformVectorFromObjectToWorldSpace(dpdu));
    si->shading.dpdv = normalize(optixTransformVectorFromObjectToWorldSpace(dpdv));
}

extern "C" __device__ float __continuation_callable__pdf_sphere(const AreaEmitterInfo& area_info, const Vec3f & origin, const Vec3f & direction)
{
    const Sphere::Data* sphere = reinterpret_cast<Sphere::Data*>(area_info.shape_data);
    SurfaceInteraction si;
    const Vec3f local_o = area_info.worldToObj.pointMul(origin);
    const Vec3f local_d = area_info.worldToObj.vectorMul(direction);
    
    if (!hitSphere(sphere, local_o, local_d, 0.01f, 1e16f, si))
        return 0.0f;

    const Vec3f center = sphere->center;
    const float radius = sphere->radius;
    const float cos_theta_max = sqrtf(1.0f - radius * radius / pow2(length(center - local_o)));
    const float solid_angle = 2.0f * math::pi * (1.0f - cos_theta_max);
    return 1.0f / solid_angle;
}

extern "C" __device__ Vec3f __direct_callable__rnd_sample_sphere(const AreaEmitterInfo& area_info, SurfaceInteraction* si)
{
    const Sphere::Data* sphere = reinterpret_cast<Sphere::Data*>(area_info.shape_data);
    const Vec3f center = sphere->center;
    const Vec3f local_o = area_info.worldToObj.pointMul(si->p);
    const Vec3f oc = center - local_o;
    float distance_squared = dot(oc, oc);
    Onb onb(normalize(oc));
    uint32_t seed = si->seed;
    Vec3f to_light = randomSampleToSphere(seed, sphere->radius, distance_squared);
    onb.inverseTransform(to_light);
    si->seed = seed;
    return normalize(area_info.objToWorld.vectorMul(to_light));
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
    si->shading.dpdu = normalize(optixTransformNormalFromObjectToWorldSpace(dpdu));
    si->shading.dpdv = normalize(optixTransformNormalFromObjectToWorldSpace(dpdv));
}

extern "C" __device__ void __anyhit__alpha_discard()
{
    const HitgroupData* data = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    if (!data->alpha_texture.data) return;

    const uint32_t hit_kind = optixGetHitKind();

    const int prim_id = optixGetPrimitiveIndex();

    Vec2f texcoord;
    if (optixIsTriangleHit())
    {
        const TriangleMesh::Data* mesh = reinterpret_cast<TriangleMesh::Data*>(data->shape_data);
        const Face face = mesh->faces[prim_id];
        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;
        const Vec2f texcoord0 = mesh->texcoords[face.texcoord_id.x()];
        const Vec2f texcoord1 = mesh->texcoords[face.texcoord_id.y()];
        const Vec2f texcoord2 = mesh->texcoords[face.texcoord_id.z()];

        texcoord = (1.0f - u - v) * texcoord0 + u * texcoord1 + v * texcoord2;
    }
    else
    {
        // Attributes for texture coordinates must be stored at Attribute_3 ~ Attribute_4
        texcoord = getVec2fFromAttribute<3>();
    }

    const Vec4f alpha = optixDirectCall<Vec4f, const Vec2f&, void*>(
        data->alpha_texture.prg_id, texcoord, data->alpha_texture.data);

    if (alpha.w() == 0.0f) optixIgnoreIntersection();
}
