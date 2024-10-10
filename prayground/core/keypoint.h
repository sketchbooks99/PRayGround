#pragma once

#include <prayground/optix/macros.h>

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
    struct Keypoint {
        T value;
        float t;

        static INLINE HOSTDEVICE T easeLinear(Keypoint<T> a, Keypoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            return a.value * (1.0f - p) + b.value * p;
        }

        // Sine easing functions
        static INLINE HOSTDEVICE T easeInSine(Keypoint<T> a, Keypoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            float x = 1.0f - cosf(p * math::pi * 0.5f);
            return a.value * (1.0f - x) + b.value * x;
        }

        static INLINE HOSTDEVICE T easeOutSine(Keypoint<T> a, Keypoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            float x = sinf(p * math::pi * 0.5f);
            return a.value * (1.0f - x) + b.value * x;
        }

        static INLINE HOSTDEVICE T easeInOutSine(Keypoint<T> a, Keypoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            float x = -(cosf(math::pi * p) - 1.0f) * 0.5f;
            return a.value * (1.0f - x) + b.value * x;
        }

        // Quadratic easing functions
        static INLINE HOSTDEVICE T easeInQuad(Keypoint<T> a, Keypoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            float x = p * p;
            return a.value * (1.0f - x) + b.value * x;
        }

        static INLINE HOSTDEVICE T easeOutQuad(Keypoint<T> a, Keypoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            float x = 1.0f - (1.0f - p) * (1.0f - p);
            return a.value * (1.0f - x) + b.value * x;
        }

        static INLINE HOSTDEVICE T easeInOutQuad(Keypoint<T> a, Keypoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            float x = p < 0.5f ? 2.0f * p * p : 1.0f - powf(-2.0f * p + 2.0f, 2.0f) * 0.5f;
            return a.value * (1.0f - x) + b.value * x;
        }

        // Cubic easing functions
        static INLINE HOSTDEVICE T easeInCubic(Keypoint<T> a, Keypoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            float x = p * p * p;
            return a.value * (1.0f - x) + b.value * x;
        }

        static INLINE HOSTDEVICE T easeOutCubic(Keypoint<T> a, Keypoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            float x = 1.0f - powf(1.0f - p, 3.0f);
            return a.value * (1.0f - x) + b.value * x;
        }

        static INLINE HOSTDEVICE T easeInOutCubic(Keypoint<T> a, Keypoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            float x = p < 0.5f ? 4.0f * p * p * p : 1.0f - powf(-2.0f * p + 2.0f, 3.0f) * 0.5f;
            return a.value * (1.0f - x) + b.value * x;
        }

        // Quartic easing functions
        static INLINE HOSTDEVICE T easeInQuart(Keypoint<T> a, Keypoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            float x = p * p * p * p;
            return a.value * (1.0f - x) + b.value * x;
        }

        static INLINE HOSTDEVICE T easeOutQuart(Keypoint<T> a, Keypoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            float x = 1.0f - powf(1.0f - p, 4.0f);
            return a.value * (1.0f - x) + b.value * x;
        }

        static INLINE HOSTDEVICE T easeInOutQuart(Keypoint<T> a, Keypoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            float x = p < 0.5f ? 8.0f * p * p * p * p : 1.0f - powf(-2.0f * p + 2.0f, 4.0f) * 0.5f;
            return a.value * (1.0f - x) + b.value * x;
        }

        // Exponential easing functions
        static INLINE HOSTDEVICE T easeInExpo(Keypoint<T> a, Keypoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            float x = p == 0.0f ? 0.0f : powf(2.0f, 10.0f * (p - 1.0f));
            return a.value * (1.0f - x) + b.value * x;
        }

        static INLINE HOSTDEVICE T easeOutExpo(Keypoint<T> a, Keypoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            float x = p == 1.0f ? 1.0f : 1.0f - powf(2.0f, -10.0f * p);
            return a.value * (1.0f - x) + b.value * x;
        }

        static INLINE HOSTDEVICE T easeInOutExpo(Keypoint<T> a, Keypoint<T> b, float t) {
            float p = (t - a.t) / (b.t - a.t);
            if (p == 0.0f)
                return a.value;
            if (p == 1.0f)
                return b.value;
            float x = p < 0.5f ? powf(2.0f, 20.0f * p - 10.0f) * 0.5f : (2.0f - powf(2.0f, -20.0f * p + 10.0f)) * 0.5f;
            return a.value * (1.0f - x) + b.value * x;
        }

        static INLINE HOSTDEVICE T ease(Keypoint<T> a, Keypoint<T> b, float t, EaseType type) {
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



} // namespace prayground