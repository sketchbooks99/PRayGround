#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <sutil/vec_math.h>
#include <core/util.h>
#include <optix/util.h>

namespace pt {

/** \note Currently, CPU tracing is not implemented. */
bool cpu_trace_occlusion(unsigned long long, float3, float3, float, float) { }
float3 cpu_trace_radiance(unsigned long long, float3, float3, float, float) { }

using TraceOcclusionFunc = bool (*) (unsigned long long, float3, float3, float, float);
#ifdef __CUDACC__
    TraceOcclusionFunc _trace_occlusion = trace_occlusion;
#else 
    TraceOcclusionFunc _trace_occlusion = cpu_trace_occlusion;
#endif

// Forward declaration
class Material;
using MaterialPtr = Material*;

enum class MaterialType {
    Diffuse = 1u << 0,
    Conductor = 1u << 1,
    Dielectric = 1u << 2,
    Emitter = 1u << 3,
    Disney = 1u << 4
};

#if !defined(__CUDACC__)
inline std::ostream& operator<<(std::ostream& out, MaterialType type) {
    switch(type) {
    case MaterialType::Diffuse:
        return out << "MaterialType::Diffuse";
    case MaterialType::Conductor:
        return out << "MaterialType::Conductor";
    case MaterialType::Dielectric:
        return out << "MaterialType::Sphere";
    case MaterialType::Emitter:
        return out << "MaterialType::Emitter";
    case MaterialType::Disney:
        return out << "MaterialType::Disney";
    default:
        Throw("This MaterialType is not supported\n");
        return out << "";
    }
}
#endif

/** 
 * \brief Initialize object on device.
 * 
 * \note Initailization must be excecuted only once.
 */
template <typename T, typename... Args>
__global__ void setup_material_on_device(T** d_ptr, Args... args) {
    (*d_ptr) = new T(args...);
}

template <typename T>
__global__ void delete_material_on_device(T** d_ptr) {
    delete (void*)*d_ptr;
}

// Abstract class to compute scattering properties.
class Material {
protected:
    CUdeviceptr d_ptr { 0 }; // device pointer.

    virtual void setup_on_device() = 0;
    virtual void delete_on_device() = 0;

public:
    virtual HOSTDEVICE void sample(SurfaceInteraction& si) const = 0;
    virtual HOSTDEVICE float3 emittance(SurfaceInteraction& si) const = 0;
    /// FUTURE:
    // virtual HOSTDEVICE float pdf(const Ray& r, const SurfaceInteraction& si) const = 0; */
    virtual HOST MaterialType type() const = 0;

    HOST CUdeviceptr get_dptr() { return d_ptr; }
};

}