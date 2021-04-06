#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <sutil/vec_math.h>
#include <include/core/util.h>
#include <include/optix/util.h>

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

#ifndef __CUDACC__
inline std::ostream& operator<<(std::ostream& out, const MaterialType& type) {
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

#ifndef __CUDACC__
    virtual MaterialType type() const = 0;
    HOST MaterialPtr get_dptr() const { return d_ptr; }
    HOST MaterialPtr& get_dptr() { return d_ptr; }
#endif

protected:
    MaterialPtr d_ptr { 0 }; // device pointer.
    /**
     * \brief Allocation and release of device side object.
     */
    virtual void setup_on_device() = 0;
    virtual void delete_on_device() = 0;
};

}