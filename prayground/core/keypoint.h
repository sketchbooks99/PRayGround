#pragma once

#include <prayground/optix/macros.h>
#ifndef __CUDACC__
#include <vector>
#include <prayground/math/bezier.h>
#endif

namespace prayground {

    enum class EaseType : uint32_t {
        Linear = 0,
        InSine = 1,
        OutSine = 2,
        InOutSine = 3,
        InQuad = 4,
        OutQuad = 5,
        InOutQuad = 6,
        InCubic = 7,
        OutCubic = 8,
        InOutCubic = 9,
        InQuart = 10,
        OutQuart = 11,
        InOutQuart = 12,
        InExpo = 13,
        OutExpo = 14,
        InOutExpo = 15
    };

    template <typename T>
    struct KeyPoint {
        T value;
        float t;

        static INLINE HOSTDEVICE T easeLinear(KeyPoint<T> a, KeyPoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            return a.value * (1.0f - p) + b.value * p;
        }

        // Sine easing functions
        static INLINE HOSTDEVICE T easeInSine(KeyPoint<T> a, KeyPoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            float x = 1.0f - cosf(p * math::pi * 0.5f);
            return a.value * (1.0f - x) + b.value * x;
        }

        static INLINE HOSTDEVICE T easeOutSine(KeyPoint<T> a, KeyPoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            float x = sinf(p * math::pi * 0.5f);
            return a.value * (1.0f - x) + b.value * x;
        }

        static INLINE HOSTDEVICE T easeInOutSine(KeyPoint<T> a, KeyPoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            float x = -(cosf(math::pi * p) - 1.0f) * 0.5f;
            return a.value * (1.0f - x) + b.value * x;
        }

        // Quadratic easing functions
        static INLINE HOSTDEVICE T easeInQuad(KeyPoint<T> a, KeyPoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            float x = p * p;
            return a.value * (1.0f - x) + b.value * x;
        }

        static INLINE HOSTDEVICE T easeOutQuad(KeyPoint<T> a, KeyPoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            float x = 1.0f - (1.0f - p) * (1.0f - p);
            return a.value * (1.0f - x) + b.value * x;
        }

        static INLINE HOSTDEVICE T easeInOutQuad(KeyPoint<T> a, KeyPoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            float x = p < 0.5f ? 2.0f * p * p : 1.0f - powf(-2.0f * p + 2.0f, 2.0f) * 0.5f;
            return a.value * (1.0f - x) + b.value * x;
        }

        // Cubic easing functions
        static INLINE HOSTDEVICE T easeInCubic(KeyPoint<T> a, KeyPoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            float x = p * p * p;
            return a.value * (1.0f - x) + b.value * x;
        }

        static INLINE HOSTDEVICE T easeOutCubic(KeyPoint<T> a, KeyPoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            float x = 1.0f - powf(1.0f - p, 3.0f);
            return a.value * (1.0f - x) + b.value * x;
        }

        static INLINE HOSTDEVICE T easeInOutCubic(KeyPoint<T> a, KeyPoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            float x = p < 0.5f ? 4.0f * p * p * p : 1.0f - powf(-2.0f * p + 2.0f, 3.0f) * 0.5f;
            return a.value * (1.0f - x) + b.value * x;
        }

        // Quartic easing functions
        static INLINE HOSTDEVICE T easeInQuart(KeyPoint<T> a, KeyPoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            float x = p * p * p * p;
            return a.value * (1.0f - x) + b.value * x;
        }

        static INLINE HOSTDEVICE T easeOutQuart(KeyPoint<T> a, KeyPoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            float x = 1.0f - powf(1.0f - p, 4.0f);
            return a.value * (1.0f - x) + b.value * x;
        }

        static INLINE HOSTDEVICE T easeInOutQuart(KeyPoint<T> a, KeyPoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            float x = p < 0.5f ? 8.0f * p * p * p * p : 1.0f - powf(-2.0f * p + 2.0f, 4.0f) * 0.5f;
            return a.value * (1.0f - x) + b.value * x;
        }

        // Exponential easing functions
        static INLINE HOSTDEVICE T easeInExpo(KeyPoint<T> a, KeyPoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            float x = p == 0.0f ? 0.0f : powf(2.0f, 10.0f * (p - 1.0f));
            return a.value * (1.0f - x) + b.value * x;
        }

        static INLINE HOSTDEVICE T easeOutExpo(KeyPoint<T> a, KeyPoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            float x = p == 1.0f ? 1.0f : 1.0f - powf(2.0f, -10.0f * p);
            return a.value * (1.0f - x) + b.value * x;
        }

        static INLINE HOSTDEVICE T easeInOutExpo(KeyPoint<T> a, KeyPoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            if (p == 0.0f)
                return a.value;
            if (p == 1.0f)
                return b.value;
            float x = p < 0.5f ? powf(2.0f, 20.0f * p - 10.0f) * 0.5f : (2.0f - powf(2.0f, -20.0f * p + 10.0f)) * 0.5f;
            return a.value * (1.0f - x) + b.value * x;
        }

        static INLINE HOSTDEVICE T ease(KeyPoint<T> a, KeyPoint<T> b, float t, EaseType type) {
            switch (type) {
            case EaseType::Linear: return easeLinear(a, b, t);
            case EaseType::InSine: return easeInSine(a, b, t);
            case EaseType::OutSine: return easeOutSine(a, b, t);
            case EaseType::InOutSine: return easeInOutSine(a, b, t);
            case EaseType::InQuad: return easeInQuad(a, b, t);
            case EaseType::OutQuad: return easeOutQuad(a, b, t);
            case EaseType::InOutQuad: return easeInOutQuad(a, b, t);
            case EaseType::InCubic: return easeInCubic(a, b, t);
            case EaseType::OutCubic: return easeOutCubic(a, b, t);
            case EaseType::InOutCubic: return easeInOutCubic(a, b, t);
            case EaseType::InQuart: return easeInQuart(a, b, t);
            case EaseType::OutQuart: return easeOutQuart(a, b, t);
            case EaseType::InOutQuart: return easeInOutQuart(a, b, t);
            case EaseType::InExpo: return easeInExpo(a, b, t);
            case EaseType::OutExpo: return easeOutExpo(a, b, t);
            case EaseType::InOutExpo: return easeInOutExpo(a, b, t);
            default: return easeLinear(a, b, t);
            }
        }
    };

#ifndef __CUDACC__
    template <typename T>
    inline T getValueFromKeypoints(float time, float min_time, float max_time, const std::vector<KeyPoint<T>>& points, EaseType ease) {
        if (points.size() == 0)
            return T();

        for (size_t i = 0; i < points.size() - 1; i++) {
            if (points[i].t <= time && time <= points[i + 1].t) {
                T value = KeyPoint<T>::ease(points[i], points[i + 1], time, ease);
                return value;
            }
        }

        float _min_time = fmaxf(min_time, points.back().t);
        float _max_time = fminf(max_time, points.back().t);

        if (time < _min_time)
            return points.front().value;
        else if (time > _max_time)
            return points.back().value;
        return T();
    }

    // Bezier curve interpolation between keypoints
    // Uses cubic Bezier with automatic tangent calculation
    template <typename T>
    inline T getValueFromKeypointsBezier(
        float time, 
        float min_time, 
        float max_time, 
        const std::vector<KeyPoint<T>>& points,
        float tension = 0.3f  // Control handle length (0.0 = linear, higher = more curve)
    ) {
        if (points.size() == 0)
            return T();
        
        if (points.size() == 1)
            return points[0].value;

        // Find the segment containing 'time'
        for (size_t i = 0; i < points.size() - 1; i++) {
            if (points[i].t <= time && time <= points[i + 1].t) {
                const KeyPoint<T>& p0 = points[i];
                const KeyPoint<T>& p1 = points[i + 1];
                
                // Normalize t to [0, 1] within this segment
                float t = (time - p0.t) / (p1.t - p0.t);
                
                // Calculate control handles based on neighboring points
                T handle_right_offset = T();
                T handle_left_offset = T();
                
                // Right handle for p0 (outgoing tangent)
                if (i > 0) {
                    // Use previous point for tangent
                    const KeyPoint<T>& p_prev = points[i - 1];
                    T tangent = (p1.value - p_prev.value) * (1.0f / (p1.t - p_prev.t));
                    handle_right_offset = tangent * (p1.t - p0.t) * tension;
                } else {
                    // First segment: use forward difference
                    handle_right_offset = (p1.value - p0.value) * tension;
                }
                
                // Left handle for p1 (incoming tangent)
                if (i < points.size() - 2) {
                    // Use next point for tangent
                    const KeyPoint<T>& p_next = points[i + 2];
                    T tangent = (p_next.value - p0.value) * (1.0f / (p_next.t - p0.t));
                    handle_left_offset = tangent * (p1.t - p0.t) * tension;
                } else {
                    // Last segment: use backward difference
                    handle_left_offset = (p1.value - p0.value) * tension;
                }
                
                // Cubic Bezier control points
                T a = p0.value;
                T b = p0.value + handle_right_offset;
                T c = p1.value - handle_left_offset;
                T d = p1.value;
                
                // Evaluate cubic Bezier at t
                float u = 1.0f - t;
                float u2 = u * u;
                float u3 = u2 * u;
                float t2 = t * t;
                float t3 = t2 * t;
                
                return a * u3 + b * (3.0f * u2 * t) + c * (3.0f * u * t2) + d * t3;
            }
        }

        float _min_time = fmaxf(min_time, points.front().t);
        float _max_time = fminf(max_time, points.back().t);

        if (time < _min_time)
            return points.front().value;
        else if (time > _max_time)
            return points.back().value;
        
        return T();
    }

    // Bezier curve interpolation with explicit control handles
    // More control but requires manual handle specification per keypoint
    template <typename T>
    struct KeyPointBezier {
        T value;
        T handle_left;   // Incoming control handle (relative to value)
        T handle_right;  // Outgoing control handle (relative to value)
        float t;

        KeyPointBezier() : value(T()), handle_left(T()), handle_right(T()), t(0.0f) {}
        KeyPointBezier(const T& v, float time) 
            : value(v), handle_left(T()), handle_right(T()), t(time) {}
        KeyPointBezier(const T& v, const T& h_left, const T& h_right, float time)
            : value(v), handle_left(h_left), handle_right(h_right), t(time) {}
    };

    template <typename T>
    inline T getValueFromKeypointsBezierExplicit(
        float time,
        float min_time,
        float max_time,
        const std::vector<KeyPointBezier<T>>& points
    ) {
        if (points.size() == 0)
            return T();
        
        if (points.size() == 1)
            return points[0].value;

        // Find the segment containing 'time'
        for (size_t i = 0; i < points.size() - 1; i++) {
            if (points[i].t <= time && time <= points[i + 1].t) {
                const KeyPointBezier<T>& p0 = points[i];
                const KeyPointBezier<T>& p1 = points[i + 1];
                
                // Normalize t to [0, 1] within this segment
                float t = (time - p0.t) / (p1.t - p0.t);
                
                // Cubic Bezier control points (handles are relative offsets)
                T a = p0.value;
                T b = p0.value + p0.handle_right;
                T c = p1.value + p1.handle_left;  // Note: handle_left is incoming, so it's already pointing towards value
                T d = p1.value;
                
                // Evaluate cubic Bezier at t
                float u = 1.0f - t;
                float u2 = u * u;
                float u3 = u2 * u;
                float t2 = t * t;
                float t3 = t2 * t;
                
                return a * u3 + b * (3.0f * u2 * t) + c * (3.0f * u * t2) + d * t3;
            }
        }

        float _min_time = fmaxf(min_time, points.front().t);
        float _max_time = fminf(max_time, points.back().t);

        if (time < _min_time)
            return points.front().value;
        else if (time > _max_time)
            return points.back().value;
        
        return T();
    }
#endif

} // namespace prayground