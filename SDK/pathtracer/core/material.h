#pragma once

#include <optix.h>
#include <sutil/vec_math.h>
#include "core_util.h"
#include "object.h"
#include "../cuda/cuda_util.cu"

namespace pt {

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

// Abstract class to compute scattering properties.
class Material {
public:    
    virtual HOSTDEVICE void sample(SurfaceInteraction& si) const = 0;
    virtual HOSTDEVICE float3 emittance(SurfaceInteraction& si) const = 0;
    /// FUTURE:
    // virtual HOSTDEVICE float pdf(const Ray& r, const SurfaceInteraction& si) const = 0; */
    virtual HOST MaterialType type() const = 0;
};

}