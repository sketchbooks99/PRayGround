#pragma once 

#include <prayground/optix/cuda/device_util.cuh>
#include <prayground/shape/box.h>
#include <prayground/shape/cylinder.h>
#include <prayground/shape/plane.h>
#include <prayground/shape/sphere.h>
#include <prayground/shape/trianglemesh.h>
#include <prayground/core/ray.h>
#include <prayground/core/interaction.h>
#include <prayground/optix/sbt.h>

#ifdef __CUDACC__

namespace prayground {

    class LinearInterpolator;
    class QuadricInterpolator;
    class CubicInterpolator;

    // ----------------------------------------------------------------------------------------
    // Cylinder
    // ----------------------------------------------------------------------------------------
    INLINE DEVICE Vec2f pgGetCylinderUV(
        const Vec3f& p, const Cylinder::Data& cylinder, const bool hit_disk)
    {
        if (hit_disk)
        {
            const float r = sqrtf(p.x() * p.x() + p.z() * p.z()) / cylinder.radius;
            const float theta = atan2(p.z(), p.x());
            float u = 1.0f - (theta + math::pi / 2.0f) / math::pi;
            return Vec2f(u, r);
        }
        else
        {
            float phi = atan2(p.z(), p.x());
            if (phi < 0.0f) phi += math::two_pi;
            const float u = phi / math::two_pi;
            const float v = (p.y() + cylinder.height / 2.0f) / cylinder.height;
            return Vec2f(u, v);
        }
    }

    INLINE DEVICE bool pgIntersectionCylinder(
        const Cylinder::Data* cylinder, const Ray& ray, Shading* shading, float* time
    )
    {
        const float radius = cylinder->radius;
        const float height = cylinder->height;

        // Get discriminant with infinite cylinder along with Y axis
        const float a = pow2(ray.d.x()) + pow2(ray.d.z());
        const float half_b = ray.o.x() * ray.d.x() + ray.o.z() * ray.d.z();
        const float c = dot(ray.o, ray.o) - ray.o.y() * ray.o.y() - radius * radius;
        const float discriminant = half_b * half_b - a * c;

        if (discriminant <= 0.0f)
            return false;

        const float sqrtd = sqrtf(discriminant);

        const float side_tmin = (-half_b - sqrtd) / a;
        const float side_tmax = (-half_b + sqrtd) / a;

        if (side_tmin > ray.tmax || side_tmax < ray.tmin)
            return false;

        const float upper = height / 2.0f;
        const float lower = -height / 2.0f;
        const float y_tmin = fmin((lower - ray.o.y()) / ray.d.y(), (upper - ray.o.y()) / ray.d.y());
        const float y_tmax = fmax((lower - ray.o.y()) / ray.d.y(), (upper - ray.o.y()) / ray.d.y());

        float tmin = fmax(y_tmin, side_tmin);
        float tmax = fmin(y_tmax, side_tmax);
        if (tmin > tmax || (tmax < ray.tmin) || (tmin > ray.tmax))
            return false;

        bool hit_min = true;
        float t = tmin;

        // Check near intersection
        if (t < ray.tmin || ray.tmax < t)
        {
            hit_min = false;
            t = tmax;
            // Check far intersection
            if (t < ray.tmin || ray.tmax < t)
                return false;
        }

        Vec3f p = ray.at(t);
        bool hit_disk = hit_min ? y_tmin > side_tmin : y_tmax < side_tmax;
        Vec3f n = hit_disk
            ? normalize(p - Vec3f(p.x(), 0.0f, p.z()))   // Hit at disk
            : normalize(p - Vec3f(0.0f, p.y(), 0.0f));   // Hit at side
        Vec2f uv = pgGetCylinderUV(p, *cylinder, hit_disk);

        shading->n = n;
        shading->uv = uv;
        *time = t;

        if (hit_disk)
        {
            const float r_hit = sqrtf(p.x() * p.x() + p.z() * p.z());
            shading->dpdu = Vec3f(-math::two_pi * p.y(), 0.0f, math::two_pi * p.z());
            shading->dpdv = Vec3f(p.x(), 0.0f, p.z()) * radius / r_hit;
        }
        else
        {
            shading->dpdu = Vec3f(-math::two_pi * p.z(), 0.0f, math::two_pi * p.x());
            shading->dpdv = Vec3f(0.0f, height, 0.0f);
        }
        return true;
    }

    INLINE DEVICE void pgReportIntersectionCylinder(const Cylinder::Data* cylinder, const Ray& ray)
    {
        Shading shading;
        float time;
        if (pgIntersectionCylinder(cylinder, ray, &shading, &time))
        {
            // Pack shading pointer to two attributes
            // If you'd like to unpack the pointer, use
            // Shading* shading = getPtrFromTwoAttributes<Shading, 0>();
            uint32_t a0, a1;
            packPointer(&shading, a0, a1);
            optixReportIntersection(time, 0, a0, a1);
        }
    }

    // ----------------------------------------------------------------------------------------
    // Box
    // ----------------------------------------------------------------------------------------
    INLINE DEVICE Vec2f pgGetBoxUV(const Vec3f& p, const Box::Data& box, const int axis)
    {
        int u_axis = (axis + 1) % 3;
        int v_axis = (axis + 2) % 3;

        // Swap axis (u=Z, v=X) -> (u=X, v=Z) if axis == Y
        if (axis == 1) swap(u_axis, v_axis);

        Vec2f uv(
            (p[u_axis] - box.min[u_axis]) / (box.max[u_axis] - box.min[u_axis]),
            (p[v_axis] - box.min[v_axis]) / (box.max[v_axis] - box.min[v_axis])
        );

        return clamp(uv, 0.0f, 1.0f);
    }

    /* Return a hitting axis of box face, X=0, Y=1, Z=2 */
    INLINE DEVICE bool pgIntersectionBox(
        const Box::Data* box, const Ray& ray, Shading* shading, float* time
    )
    {
        const Vec3f min = box->min;
        const Vec3f max = box->max;

        float tmin = ray.tmin, tmax = ray.tmax;
        int min_axis = -1, max_axis = -1;

        // Intersection test
        for (int i = 0; i < 3; i++)
        {
            float t0, t1;
            if (ray.d[i] == 0.0f)
            {
                t0 = fminf(min[i] - ray.o[i], max[i] - ray.o[i]);
                t1 = fmaxf(min[i] - ray.o[i], max[i] - ray.o[i]);
            }
            else
            {
                t0 = fminf((min[i] - ray.o[i]) / ray.d[i], (max[i] - ray.o[i]) / ray.d[i]);
                t1 = fmaxf((min[i] - ray.o[i]) / ray.d[i], (max[i] - ray.o[i]) / ray.d[i]);
            }

            // Update hitting axis 
            min_axis = t0 > tmin ? i : min_axis;
            max_axis = t1 < tmax ? i : max_axis;

            // Update distance from the ray origin to an intersection surface
            tmin = fmaxf(t0, tmin);
            tmax = fminf(t1, tmax);

            // No intersection
            if (tmax < tmin)
                return false;
        }

        Vec3f center = (min + max) / 2.0f;
        int axis = min_axis;
        float t = tmin;
        // Check the near intersection
        if ((t < ray.tmin || ray.tmax < t) || (axis < 0 || 2 < axis))
        {
            axis = max_axis;
            t = tmax;
            // Check the far intersection
            if ((t < ray.tmin || ray.tmax < t) || (axis < 0 || 2 < axis))
            {
                return false;
            }
        }

        Vec3f p = ray.at(t);
        Vec3f center_axis = p;
        center_axis[axis] = center[axis];
        Vec3f n = normalize(p - center_axis);
        Vec2f uv = pgGetBoxUV(p, *box, axis);

        // Store the shading information
        shading->n = n;
        shading->uv = uv;

        // x
        if (axis == 0)
        {
            shading->dpdu = Vec3f(0.0f, 0.0f, 1.0f);
            shading->dpdv = Vec3f(0.0f, 1.0f, 0.0f);
        }
        // y
        else if (axis == 1)
        {
            shading->dpdu = Vec3f(1.0f, 0.0f, 0.0f);
            shading->dpdv = Vec3f(0.0f, 0.0f, 1.0f);
        }
        // z
        else if (axis == 2)
        {
            shading->dpdu = Vec3f(1.0f, 0.0f, 0.0f);
            shading->dpdv = Vec3f(0.0f, 1.0f, 0.0f);
        }

        *time = t;

        return true;
    }

    INLINE DEVICE void pgReportIntersectionBox(const Box::Data* box, const Ray& ray)
    {
        Shading shading;
        float time;
        if (pgIntersectionBox(box, ray, &shading, &time))
        {
            // Pack shading pointer to two attributes
            // If you'd like to unpack the pointer, use
            // Shading* shading = getPtrFromTwoAttributes<Shading, 0>();
            uint32_t a0, a1;
            packPointer(&shading, a0, a1);
            optixReportIntersection(time, 0, a0, a1);
        }
    }

    // ----------------------------------------------------------------------------------------
    // Plane
    // ----------------------------------------------------------------------------------------
    INLINE DEVICE Vec2f pgGetPlaneUV(const Vec2f p, const Plane::Data& plane)
    {
        const float u = (p.x() - plane.min.x()) / (plane.max.x() - plane.min.x());
        const float v = (p.y() - plane.min.y()) / (plane.max.y() - plane.min.y());
        return Vec2f(u, v);
    }

    INLINE DEVICE bool pgIntersectionPlane(const Plane::Data* plane, const Ray& ray, Shading* shading, float* time)
    {
        const float t = -ray.o.y() / ray.d.y();
        const float x = ray.o.x() + t * ray.d.x();
        const float z = ray.o.z() + t * ray.d.z();

        if (plane->min.x() < x && x < plane->max.x() &&
            plane->min.y() < z && z < plane->max.y() &&
            ray.tmin < t && t < ray.tmax)
        {
            shading->uv = pgGetPlaneUV(Vec2f(x, z), *plane);
            shading->n = Vec3f(0, 1, 0);
            shading->dpdu = Vec3f(1, 0, 0);
            shading->dpdv = Vec3f(0, 0, 1);
            *time = t;
            return true;
        }
        return false;
    }

    INLINE DEVICE void pgReportIntersectionPlane(const Plane::Data* plane, const Ray& ray)
    {
        Shading shading;
        float time;
        if (pgIntersectionPlane(plane, ray, &shading, &time))
        {
            // Pack shading pointer to two attributes
            // If you'd like to unpack the pointer, use
            // Shading* shading = getPtrFromTwoAttributes<Shading, 0>();
            uint32_t a0, a1;
            packPointer(&shading, a0, a1);
            optixReportIntersection(time, 0, a0, a1);
        }
    }

    // ----------------------------------------------------------------------------------------
    // Sphere
    // ----------------------------------------------------------------------------------------
    INLINE DEVICE Vec2f pgGetSphereUV(const Vec3f& p) {
        float phi = atan2(p.z(), p.x());
        if (phi < 0) phi += 2.0f * math::pi;
        float theta = acos(p.y());
        float u = phi / (2.0f * math::pi);
        float v = theta * math::inv_pi;
        return Vec2f(u, v);
    }

    INLINE DEVICE bool pgIntersectionSphere(
        const Sphere::Data* sphere, const Ray& ray, Shading* shading, float* time)
    {
        const Vec3f oc = ray.o - sphere->center;
        const float a = dot(ray.d, ray.d);
        const float half_b = dot(oc, ray.d);
        const float c = dot(oc, oc) - pow2(sphere->radius);
        const float discriminant = half_b * half_b - a * c;

        if (discriminant <= 0.0f)
            return false;

        const float sqrtd = sqrtf(discriminant);

        float t = (-half_b - sqrtd) / a;
        if (t < ray.tmin || ray.tmax < t)
        {
            t = (-half_b + sqrtd) / a;
            if (t < ray.tmin || ray.tmax < t)
                return false;
        }

        const Vec3f p = ray.at(t);
        shading->n = p / sphere->radius;
        shading->uv = pgGetSphereUV(shading->n);

        float phi = atan2(shading->n.z(), shading->n.x());
        if (phi < 0) phi += math::two_pi;
        const float theta = acosf(shading->n.y());
        shading->dpdu = Vec3f(-math::two_pi * shading->n.z(), 0, math::two_pi * shading->n.x());
        shading->dpdv = math::pi * Vec3f(shading->n.y() * cosf(phi), -sinf(theta), shading->n.y() * sinf(phi));

        *time = t;

        return true;
    }

    INLINE DEVICE void pgReportIntersectionSphere(const Sphere::Data* sphere, const Ray& ray)
    {
        Shading shading;
        float time;
        if (pgIntersectionSphere(sphere, ray, &shading, &time))
        {
            // Pack shading pointer to two attributes
            // If you'd like to unpack the pointer, use
            // Shading* shading = getPtrFromTwoAttributes<Shading, 0>();
            uint32_t a0, a1;
            packPointer(&shading, a0, a1);
            optixReportIntersection(time, 0, a0, a1);
        }
    }

    // ----------------------------------------------------------------------------------------
    // Mesh
    // ----------------------------------------------------------------------------------------

    /**
     * @brief Calculate shading frame on triangle 
     * @param mesh : Triangle mesh data
     * @param bc : Barycentric coordinate on triangle
     * @param primitive_index : The primitive index of intersecting surface
     * @return Shading frame on triangle
    */
    INLINE DEVICE Shading pgGetMeshShading(const TriangleMesh::Data* mesh, const Vec2f& bc, const uint32_t primitive_index)
    {
        Shading shading = {};

        const Face face = mesh->faces[primitive_index];

        const Vec3f p0 = mesh->vertices[face.vertex_id.x()];
        const Vec3f p1 = mesh->vertices[face.vertex_id.y()];
        const Vec3f p2 = mesh->vertices[face.vertex_id.z()];

        const Vec2f texcoord0 = mesh->texcoords[face.texcoord_id.x()];
        const Vec2f texcoord1 = mesh->texcoords[face.texcoord_id.y()];
        const Vec2f texcoord2 = mesh->texcoords[face.texcoord_id.z()];
        //shading.uv = (1 - bc.x() - bc.y()) * texcoord0 + bc.x() * texcoord1 + bc.y() * texcoord2;
        shading.uv = barycentricInterop(texcoord0, texcoord1, texcoord2, bc);

        const Vec3f n0 = mesh->normals[face.normal_id.x()];
        const Vec3f n1 = mesh->normals[face.normal_id.y()];
        const Vec3f n2 = mesh->normals[face.normal_id.z()];
        shading.n = (1.0f - bc.x() - bc.y()) * n0 + bc.x() * n1 + bc.y() * n2;

        const Vec2f duv02 = texcoord0 - texcoord2, duv12 = texcoord1 - texcoord2;
        const Vec3f dp02 = p0 - p2, dp12 = p1 - p2;
        const float D = duv02.x() * duv12.y() - duv02.y() * duv12.x();
        bool degenerateUV = abs(D) < 1e-8f;
        if (!degenerateUV)
        {
            const float invD = 1.0f / D;
            shading.dpdu = (duv12.y() * dp02 - duv02.y() * dp12) * invD;
            shading.dpdv = (-duv12.x() * dp02 + duv02.x() * dp12) * invD;
        }
        if (degenerateUV || length(cross(shading.dpdu, shading.dpdv)) == 0.0f)
        {
            const Vec3f n = normalize(cross(p2 - p0, p1 - p0));
            Onb onb(n);
            shading.dpdu = onb.tangent;
            shading.dpdv = onb.bitangent;
        }

        return shading;
    }

    // ----------------------------------------------------------------------------------------
    // Curves
    // ----------------------------------------------------------------------------------------

    // Interpolator for getting curves information from optixGetCurveParameter();
    // Elements in each interpolator contains 3D position, and 1D radius(width) of curve parameter
    // LinearInterpolator
    struct LinearInterpolator {
        // Constructor
        INLINE DEVICE LinearInterpolator() {}

        INLINE DEVICE void initialize(const Vec4f* q)
        {
            e[0] = q[0];
            e[1] = q[1] - q[0];
        }

        INLINE DEVICE Vec4f evaluate(float u) const
        {
            return e[0] + u * e[1];
        }

        INLINE DEVICE Vec3f position(float u) const
        {
            return Vec3f(evaluate(u));
        }

        INLINE DEVICE float radius(float u) const
        {
            return evaluate(u).w();
        }

        INLINE DEVICE Vec4f derivative(float u) const
        {
            return e[1];
        }

        INLINE DEVICE Vec3f dPosition(float u) const
        {
            return Vec3f(derivative(u));
        }

        INLINE DEVICE float dRadius(float u) const
        {
            return derivative(u).w();
        }

        Vec4f e[2];
    };

    // QuadricInterpolator
    struct QuadraticInterpolator {
        INLINE DEVICE QuadraticInterpolator() {}

        INLINE DEVICE void initializeFromBSpline(const Vec4f* q)
        {
            e[0] = (q[0] - 2.0f * q[1] + q[2]) / 2.0f;
            e[1] = (-2.0f * q[0] + 2.0f * q[1]) / 2.0f;
            e[2] = (q[0] + q[1]) / 2.0f;
        }

        INLINE DEVICE void exportToBSpline(Vec4f ret[3]) const
        {
            ret[0] = e[0] - e[1] / 2.0f;
            ret[1] = e[0] + e[1] / 2.0f;
            ret[2] = e[0] + 1.5f * e[1] + 2.0f * e[2];
        }

        INLINE DEVICE Vec4f evaluate(float u) const
        {
            return (e[0] * u + e[1]) * u + e[2];
        }

        INLINE DEVICE Vec3f position(float u) const
        {
            return Vec3f(evaluate(u));
        }

        INLINE DEVICE float radius(float u) const
        {
            return evaluate(u).w();
        }

        INLINE DEVICE Vec4f derivative(float u) const
        {
            return 2.0f * e[0] * u + e[1];
        }

        INLINE DEVICE Vec3f dPosition(float u) const
        {
            return Vec3f(derivative(u));
        }

        INLINE DEVICE float dRadius(float u) const
        {
            return derivative(u).w();
        }

        INLINE DEVICE Vec4f doubleDerivative(float u) const
        {
            return 2.0f * e[0];
        }

        INLINE DEVICE Vec3f ddPosition(float u) const
        {
            return Vec3f(doubleDerivative(u));
        }

        Vec4f e[3];
    };

    // CubicInterpolator
    struct CubicInterpolator {
        INLINE DEVICE CubicInterpolator() {}
        
        INLINE DEVICE void initializeFromBSpline(const Vec4f* q)
        {
            e[0] = (-1.0f * q[0] + 3.0f * q[1] - 3.0f * q[2] + q[3]) / 6.0f;
            e[1] = ( 3.0f * q[0] - 6.0f * q[1] + 3.0f * q[2]       ) / 6.0f;
            e[2] = (-3.0f * q[0]               + 3.0f * q[2]       ) / 6.0f;
            e[3] = (        q[0] + 4.0f * q[1] +        q[2]       ) / 6.0f;
        }

        INLINE DEVICE void exportToBSpline(Vec4f ret[4]) const
        {
            ret[0] = (        2.0f * e[1] -        e[2] + e[3]) / 3.0f;
            ret[1] = (              -e[1]               + e[3]) / 3.0f;
            ret[2] = (        2.0f * e[1] +        e[2] + e[3]) / 3.0f;
            ret[3] = (e[0] + 11.0f * e[1] + 2.0f * e[2] + e[3]) / 3.0f;
        }

        INLINE DEVICE void initializeFromCatmullRom(const Vec4f* q)
        {
            e[0] = (      -q[0] + 3.0f * q[1] - 3.0f * q[2] + q[3]) / 2.0f;
            e[1] = (2.0f * q[0] - 5.0f * q[1] + 4.0f * q[2] - q[3]) / 2.0f;
            e[2] = (      -q[0]                      + q[2]       ) / 2.0f;
            e[3] =                       q[1];
        }

        INLINE DEVICE void exportToCatmullRom(Vec4f ret[4]) const
        {
            ret[0] = (6.0f * e[0] - 5.0f * e[1] + 2.0f * e[2] + e[3]) / 6.0f;
            ret[1] = e[0];
            ret[2] = (6.0f * e[0] + e[1] + 2.0f * e[2] + e[3]) / 6.0f;
            ret[3] = e[0];
        }

        INLINE DEVICE Vec4f evaluate(float u) const
        {
            return (((e[0] * u) + e[1]) * u + e[2]) * u + e[3];
        }

        INLINE DEVICE Vec3f position(float u) const
        {
            return Vec3f(evaluate(u));
        }

        INLINE DEVICE float radius(float u) const
        {
            return evaluate(u).w();
        }

        INLINE DEVICE Vec4f derivative(float u) const
        {
            if (u == 0.0f)
                u = 0.000001f;
            if (u == 1.0f)
                u = 0.999999f;
            return ((3.0f * e[0] * u) + 2.0f * e[1]) * u + e[2];
        }

        INLINE DEVICE Vec3f dPosition(float u) const
        {
            return Vec3f(derivative(u));
        }

        INLINE DEVICE float dRadius(float u) const
        {
            return derivative(u).w();
        }

        INLINE DEVICE Vec4f doubleDerivative(float u) const
        {
            return 6.0f * e[0] * u + 2.0f * e[1];
        }

        INLINE DEVICE Vec3f ddPosition(float u) const
        {
            return Vec3f(doubleDerivative(u));
        }

        Vec4f e[4];
    };

    template <typename Interpolator, Curves::Type CurveType>
    INLINE DEVICE Shading pgGetCurvesShading(Vec3f& hit_point, const float u, const Interpolator& interpolator)
    {   
        Shading shading;
        if (u == 0.0f)
        {
            if constexpr (CurveType == Curves::Type::Linear)
                shading.n = hit_point - Vec3f(interpolator.e[0]);
            else
                shading.n = -interpolator.dPosition(0.0f);
        }
        else if (u >= 1.0f)
        {
            if constexpr (CurveType == Curves::Type::Linear)
            {
                Vec3f p = Vec3f(interpolator.e[1]) + Vec3f(interpolator.e[0]);
                shading.n = hit_point - p;
            }
            else
            {
                shading.n = interpolator.dPosition(1.0f);
            }
        }
        else
        {
            Vec4f p = interpolator.evaluate(u);
            Vec3f position = Vec3f(p);
            float radius = p.w();
            Vec4f d = interpolator.derivative(u);
            Vec3f d_position = Vec3f(d);
            float d_radius = d.w();
            float d_length2 = dot(d_position, d_position);

            Vec3f o1 = hit_point - position;

            o1 -= (dot(o1, d_position) / d_length2) * d_position;
            o1 *= radius / length(o1);
            hit_point = position + o1;

            if constexpr (CurveType != Curves::Type::Linear)
                d_length2 -= dot(interpolator.ddPosition(u), o1);
            shading.n = d_length2 * o1 - (d_radius * radius) * d_position;
        }

        shading.n = normalize(shading.n);
        shading.dpdv = normalize(interpolator.derivative(u));
        shading.dpdu = cross(shading.dpdv, shading.n);

        return shading;
    }

    INLINE DEVICE Shading pgGetCurvesShading(Vec3f& hit_point, const uint32_t primitive_idx, OptixPrimitiveType primitive_type)
    {
        const OptixTraversableHandle gas = optixGetGASTraversableHandle();
        const uint32_t gas_sbt_idx = optixGetSbtGASIndex();
        const float u = optixGetCurveParameter();

        switch (primitive_type)
        {
        case OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR:
        {
            LinearInterpolator interpolator;
            float4 p[2];
            optixGetLinearCurveVertexData(gas, primitive_idx, gas_sbt_idx, 0.0f, p);
            interpolator.initialize((Vec4f*)p);
            return pgGetCurvesShading<LinearInterpolator, Curves::Type::Linear>(hit_point, u, interpolator);
        }
        case OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE:
        {
            QuadraticInterpolator interpolator;
            float4 p[3];
            optixGetQuadraticBSplineVertexData(gas, primitive_idx, gas_sbt_idx, 0.0f, p);
            interpolator.initializeFromBSpline((Vec4f*)p);
            return pgGetCurvesShading<QuadraticInterpolator, Curves::Type::QuadraticBSpline>(hit_point, u, interpolator);
        }
        case OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE:
        {
            CubicInterpolator interpolator;
            float4 p[4];
            optixGetCubicBSplineVertexData(gas, primitive_idx, gas_sbt_idx, 0.0f, p);
            interpolator.initializeFromBSpline((Vec4f*)p);
            return pgGetCurvesShading<CubicInterpolator, Curves::Type::CubicBSpline>(hit_point, u, interpolator);
        }
#if OPTIX_VERSION >= 70400
        case OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM:
        {
            CubicInterpolator interpolator;
            float4 p[4];
            optixGetCatmullRomVertexData(gas, primitive_idx, gas_sbt_idx, 0.0f, p);
            interpolator.initializeFromCatmullRom((Vec4f*)p);
            return pgGetCurvesShading<CubicInterpolator, Curves::Type::CatmullRom>(hit_point, u, interpolator);
        }
#endif
        default:
            return Shading{};
        }
    }

} // namespace prayground

#endif // __CUDACC__
