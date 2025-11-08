#pragma once

#include <prayground/math/vec.h>

#ifndef __CUDACC__
#include <vector>
#include <memory>
#endif

namespace prayground {

    // ----------------------------------------------------------------------------------------
    // Curves
    // ----------------------------------------------------------------------------------------

    // Interpolator for getting curves information from optixGetCurveParameter();
    // Elements in each interpolator contains 3D position, and 1D radius(width) of curve parameter
    // LinearInterpolator
    struct LinearInterpolator {
        // Constructor
        INLINE HOSTDEVICE LinearInterpolator() {}

        INLINE HOSTDEVICE void initialize(const Vec4f* q)
        {
            e[0] = q[0];
            e[1] = q[1] - q[0];
        }

        INLINE HOSTDEVICE Vec4f evaluate(float u) const
        {
            return e[0] + u * e[1];
        }

        INLINE HOSTDEVICE Vec3f position(float u) const
        {
            return Vec3f(evaluate(u));
        }

        INLINE HOSTDEVICE float radius(float u) const
        {
            return evaluate(u).w();
        }

        INLINE HOSTDEVICE Vec4f derivative(float u) const
        {
            return e[1];
        }

        INLINE HOSTDEVICE Vec3f dPosition(float u) const
        {
            return Vec3f(derivative(u));
        }

        INLINE HOSTDEVICE float dRadius(float u) const
        {
            return derivative(u).w();
        }

        Vec4f e[2];
    };

    // QuadricInterpolator
    struct QuadraticInterpolator {
        INLINE HOSTDEVICE QuadraticInterpolator() {}

        INLINE HOSTDEVICE void initializeFromBSpline(const Vec4f* q)
        {
            e[0] = (q[0] - 2.0f * q[1] + q[2]) / 2.0f;
            e[1] = (-2.0f * q[0] + 2.0f * q[1]) / 2.0f;
            e[2] = (q[0] + q[1]) / 2.0f;
        }

        INLINE HOSTDEVICE void exportToBSpline(Vec4f ret[3]) const
        {
            ret[0] = e[0] - e[1] / 2.0f;
            ret[1] = e[0] + e[1] / 2.0f;
            ret[2] = e[0] + 1.5f * e[1] + 2.0f * e[2];
        }

        INLINE HOSTDEVICE Vec4f evaluate(float u) const
        {
            return (e[0] * u + e[1]) * u + e[2];
        }

        INLINE HOSTDEVICE Vec3f position(float u) const
        {
            return Vec3f(evaluate(u));
        }

        INLINE HOSTDEVICE float radius(float u) const
        {
            return evaluate(u).w();
        }

        INLINE HOSTDEVICE Vec4f derivative(float u) const
        {
            return 2.0f * e[0] * u + e[1];
        }

        INLINE HOSTDEVICE Vec3f dPosition(float u) const
        {
            return Vec3f(derivative(u));
        }

        INLINE HOSTDEVICE float dRadius(float u) const
        {
            return derivative(u).w();
        }

        INLINE HOSTDEVICE Vec4f doubleDerivative(float u) const
        {
            return 2.0f * e[0];
        }

        INLINE HOSTDEVICE Vec3f ddPosition(float u) const
        {
            return Vec3f(doubleDerivative(u));
        }

        Vec4f e[3];
    };

    // CubicInterpolator
    struct CubicInterpolator {
        INLINE HOSTDEVICE CubicInterpolator() {}

        INLINE HOSTDEVICE void initializeFromBSpline(const Vec4f* q)
        {
            e[0] = (-1.0f * q[0] + 3.0f * q[1] - 3.0f * q[2] + q[3]) / 6.0f;
            e[1] = (3.0f * q[0] - 6.0f * q[1] + 3.0f * q[2]) / 6.0f;
            e[2] = (-3.0f * q[0] + 3.0f * q[2]) / 6.0f;
            e[3] = (q[0] + 4.0f * q[1] + q[2]) / 6.0f;
        }

        INLINE HOSTDEVICE void exportToBSpline(Vec4f ret[4]) const
        {
            ret[0] = (2.0f * e[1] - e[2] + e[3]) / 3.0f;
            ret[1] = (-e[1] + e[3]) / 3.0f;
            ret[2] = (2.0f * e[1] + e[2] + e[3]) / 3.0f;
            ret[3] = (e[0] + 11.0f * e[1] + 2.0f * e[2] + e[3]) / 3.0f;
        }

        INLINE HOSTDEVICE void initializeFromCatmullRom(const Vec4f* q)
        {
            e[0] = (-q[0] + 3.0f * q[1] - 3.0f * q[2] + q[3]) / 2.0f;
            e[1] = (2.0f * q[0] - 5.0f * q[1] + 4.0f * q[2] - q[3]) / 2.0f;
            e[2] = (-q[0] + q[2]) / 2.0f;
            e[3] = q[1];
        }

        INLINE HOSTDEVICE void exportToCatmullRom(Vec4f ret[4]) const
        {
            ret[0] = (6.0f * e[0] - 5.0f * e[1] + 2.0f * e[2] + e[3]) / 6.0f;
            ret[1] = e[0];
            ret[2] = (6.0f * e[0] + e[1] + 2.0f * e[2] + e[3]) / 6.0f;
            ret[3] = e[0];
        }

        INLINE HOSTDEVICE Vec4f evaluate(float u) const
        {
            return (((e[0] * u) + e[1]) * u + e[2]) * u + e[3];
        }

        INLINE HOSTDEVICE Vec3f position(float u) const
        {
            return Vec3f(evaluate(u));
        }

        INLINE HOSTDEVICE float radius(float u) const
        {
            return evaluate(u).w();
        }

        INLINE HOSTDEVICE Vec4f derivative(float u) const
        {
            if (u == 0.0f)
                u = 0.000001f;
            if (u == 1.0f)
                u = 0.999999f;
            return ((3.0f * e[0] * u) + 2.0f * e[1]) * u + e[2];
        }

        INLINE HOSTDEVICE Vec3f dPosition(float u) const
        {
            return Vec3f(derivative(u));
        }

        INLINE HOSTDEVICE float dRadius(float u) const
        {
            return derivative(u).w();
        }

        INLINE HOSTDEVICE Vec4f doubleDerivative(float u) const
        {
            return 6.0f * e[0] * u + 2.0f * e[1];
        }

        INLINE HOSTDEVICE Vec3f ddPosition(float u) const
        {
            return Vec3f(doubleDerivative(u));
        }

        Vec4f e[4];
    };

#ifndef __CUDACC__
    // Bezier control point
    struct BezierPoint {
        Vec3f co;            // Point position
        Vec3f handle_left;    // Left control handle
        Vec3f handle_right;   // Right control handle
        float radius;        // Radius at this point

        BezierPoint()
            : co(0.0f), handle_left(0.0f), handle_right(0.0f), radius(1.0f) {}
        
        BezierPoint(const Vec3f& position, float r = 1.0f)
            : co(position), handle_left(position), handle_right(position), radius(r) {}

        BezierPoint(const BezierPoint& bp)
            : co(bp.co), handle_left(bp.handle_left), handle_right(bp.handle_right), radius(bp.radius) {}
    };

    // Interpolation mode for radius
    enum class RadiusInterpolation {
        LINEAR,
        CARDINAL,
        BSPLINE
    };

    // Curve dimension
    enum class CurveDimension {
        Curve2D,
        Curve3D
    };

    // Fill mode
    enum class FillMode {
        FULL,
        HALF,
        NONE
    };

    // Bezier spline (one continuous curve segment)
    class BezierSpline {
    public:
        std::vector<BezierPoint> bezier_points;
        int resolution_u;
        RadiusInterpolation radius_interpolation;

        BezierSpline() 
            : resolution_u(12), radius_interpolation(RadiusInterpolation::LINEAR) {}

        // Add a new bezier point and return pointer to it
        BezierPoint* addBezierPoint(const Vec3f& pos = Vec3f(0.0f), float radius = 1.0f) {
            bezier_points.emplace_back(pos, radius);
            return &bezier_points.back();
        }

        // Get point by index
        BezierPoint* getPoint(size_t index) {
            if (index < bezier_points.size())
                return &bezier_points[index];
            return nullptr;
        }

        // Evaluate position on curve at parameter t [0, 1]
        Vec3f evaluate(float t) const {
            if (bezier_points.empty()) return Vec3f(0.0f);
            if (bezier_points.size() == 1) return bezier_points[0].co;

            // Find segment
            int num_segments = (int)bezier_points.size() - 1;
            float scaled_t = t * num_segments;
            int seg_index = min((int)scaled_t, num_segments - 1);
            float local_t = scaled_t - seg_index;

            // Cubic Bezier interpolation
            const BezierPoint& p0 = bezier_points[seg_index];
            const BezierPoint& p1 = bezier_points[seg_index + 1];

            Vec3f a = p0.co;
            Vec3f b = p0.handle_right;
            Vec3f c = p1.handle_left;
            Vec3f d = p1.co;

            float u = 1.0f - local_t;
            float u2 = u * u;
            float u3 = u2 * u;
            float t2 = local_t * local_t;
            float t3 = t2 * local_t;

            return a * u3 + b * (3.0f * u2 * local_t) + c * (3.0f * u * t2) + d * t3;
        }

        // Evaluate radius at parameter t [0, 1]
        float evaluateRadius(float t) const {
            if (bezier_points.empty()) return 1.0f;
            if (bezier_points.size() == 1) return bezier_points[0].radius;

            int num_segments = (int)bezier_points.size() - 1;
            float scaled_t = t * num_segments;
            int seg_index = min((int)scaled_t, num_segments - 1);
            float local_t = scaled_t - seg_index;

            float r0 = bezier_points[seg_index].radius;
            float r1 = bezier_points[seg_index + 1].radius;

            // Linear interpolation for radius
            return r0 * (1.0f - local_t) + r1 * local_t;
        }

        // Evaluate tangent (derivative) at parameter t [0, 1]
        Vec3f evaluateTangent(float t) const {
            if (bezier_points.empty()) return Vec3f(0.0f, 1.0f, 0.0f);
            if (bezier_points.size() == 1) return Vec3f(0.0f, 1.0f, 0.0f);

            // Find segment
            int num_segments = (int)bezier_points.size() - 1;
            float scaled_t = t * num_segments;
            int seg_index = min((int)scaled_t, num_segments - 1);
            float local_t = scaled_t - seg_index;

            // Cubic Bezier derivative
            const BezierPoint& p0 = bezier_points[seg_index];
            const BezierPoint& p1 = bezier_points[seg_index + 1];

            Vec3f a = p0.co;
            Vec3f b = p0.handle_right;
            Vec3f c = p1.handle_left;
            Vec3f d = p1.co;

            float u = 1.0f - local_t;
            float u2 = u * u;
            float t2 = local_t * local_t;

            // Derivative: 3(1-t)^2(b-a) + 6(1-t)t(c-b) + 3t^2(d-c)
            return (b - a) * (3.0f * u2) + 
                   (c - b) * (6.0f * u * local_t) + 
                   (d - c) * (3.0f * t2);
        }

        // Evaluate position on specific segment at local parameter [0, 1]
        Vec3f evaluateSegment(int segIndex, float localT) const {
            if (segIndex < 0 || segIndex >= (int)bezier_points.size() - 1)
                return Vec3f(0.0f);

            const BezierPoint& p0 = bezier_points[segIndex];
            const BezierPoint& p1 = bezier_points[segIndex + 1];

            Vec3f a = p0.co;
            Vec3f b = p0.handle_right;
            Vec3f c = p1.handle_left;
            Vec3f d = p1.co;

            float u = 1.0f - localT;
            float u2 = u * u;
            float u3 = u2 * u;
            float t2 = localT * localT;
            float t3 = t2 * localT;

            return a * u3 + b * (3.0f * u2 * localT) + c * (3.0f * u * t2) + d * t3;
        }

        // Evaluate tangent on specific segment at local parameter [0, 1]
        Vec3f evaluateSegmentTangent(int segIndex, float localT) const {
            if (segIndex < 0 || segIndex >= (int)bezier_points.size() - 1)
                return Vec3f(0.0f, 1.0f, 0.0f);

            const BezierPoint& p0 = bezier_points[segIndex];
            const BezierPoint& p1 = bezier_points[segIndex + 1];

            Vec3f a = p0.co;
            Vec3f b = p0.handle_right;
            Vec3f c = p1.handle_left;
            Vec3f d = p1.co;

            float u = 1.0f - localT;
            float u2 = u * u;
            float t2 = localT * localT;

            return (b - a) * (3.0f * u2) + 
                   (c - b) * (6.0f * u * localT) + 
                   (d - c) * (3.0f * t2);
        }

        // Static helper: Evaluate cubic Bezier between two points at t [0, 1]
        static Vec3f evaluateCubicBezier(float t, const BezierPoint& p0, const BezierPoint& p1) {
            Vec3f a = p0.co;
            Vec3f b = p0.handle_right;
            Vec3f c = p1.handle_left;
            Vec3f d = p1.co;

            float u = 1.0f - t;
            float u2 = u * u;
            float u3 = u2 * u;
            float t2 = t * t;
            float t3 = t2 * t;

            return a * u3 + b * (3.0f * u2 * t) + c * (3.0f * u * t2) + d * t3;
        }

        // Static helper: Evaluate cubic Bezier tangent (derivative) between two points at t [0, 1]
        static Vec3f evaluateCubicBezierTangent(float t, const BezierPoint& p0, const BezierPoint& p1) {
            Vec3f a = p0.co;
            Vec3f b = p0.handle_right;
            Vec3f c = p1.handle_left;
            Vec3f d = p1.co;

            float u = 1.0f - t;
            float u2 = u * u;
            float t2 = t * t;

            // Derivative: 3(1-t)^2(b-a) + 6(1-t)t(c-b) + 3t^2(d-c)
            return (b - a) * (3.0f * u2) + 
                   (c - b) * (6.0f * u * t) + 
                   (d - c) * (3.0f * t2);
        }
    };

    // Bezier curve (collection of splines)
    class BezierCurve {
    public:
        std::string name;
        std::vector<std::shared_ptr<BezierSpline>> splines;
        
        CurveDimension dimensions;
        int resolution_u;
        FillMode fill_mode;
        float bevel_depth;
        int bevel_resolution;

        BezierCurve(const std::string& curveName = "Curve")
            : name(curveName)
            , dimensions(CurveDimension::Curve3D)
            , resolution_u(12)
            , fill_mode(FillMode::FULL)
            , bevel_depth(1.0f)
            , bevel_resolution(4)
        {}

        // Add a new spline and return pointer to it
        std::shared_ptr<BezierSpline> addSpline() {
            splines.push_back(std::make_shared<BezierSpline>());
            splines.back()->resolution_u = resolution_u;
            return splines.back();
        }

        // Get spline by index
        std::shared_ptr<BezierSpline> getSpline(size_t index) {
            if (index < splines.size())
                return splines[index];
            return nullptr;
        }

        // Get total number of splines
        size_t numSplines() const {
            return splines.size();
        }
    };

#endif // __CUDACC__

} // namespace prayground
