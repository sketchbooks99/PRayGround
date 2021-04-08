#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <sutil/vec_math.h>
#include <include/core/util.h>
#include <include/optix/util.h>

namespace pt {

/**
 * \note Material class is used only in host side, 
 * because OptiX 7~ doesn't support virtual functions.
 * Material of Primitive should store the index of 
 * direct callable functions on a device.
 */

enum class MaterialType {
    Diffuse = 0,
    Conductor = 1,
    Dielectric = 2,
    Emitter = 3,
    Disney = 4
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
    virtual void sample(SurfaceInteraction& si) const = 0;
    virtual float3 emittance(SurfaceInteraction& si) const = 0;
    /// FUTURE:
    // virtual HOSTDEVICE float pdf(const Ray& r, const SurfaceInteraction& si) const = 0; */

    virtual void prepare_data() = 0;
    virtual MaterialType type() const = 0;
    
    CUdeviceptr get_dptr() const { return d_data; }
    CUdeviceptr& get_dptr() { return d_data; }
private:
    CUdeviceptr d_data { 0 };
};

}