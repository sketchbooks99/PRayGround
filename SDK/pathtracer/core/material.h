#pragma once

#include <optix.h>
#include <sutil/vec_math.h>
#include "core_util.h"
#include "object.h"

namespace pt {

// Forward declaration
class Material;
using MaterialPtr = Material*;

/** MEMO: 
 * If we need to take into account spectral property (not RGB), we should
 * switch Spectrum representation.
 * 
 * If Spectrum is rgb, rgb is float3? char3? I'm not sure which one is better.
 * 
 * NOTE: Currently, `Spectrum` must be a float3 */
struct SurfaceInteraction {
    /** position of intersection point in world coordinates. */
    float3 p;

    /** Surface normal of primitive at intersection point. */
    float3 n;

    /** Incident and outgoing directions at a surface. */
    float3 wi;
    float3 wo;

    /** Spectrum information of ray. */
    float3 spectrum;

    /** radiance and attenuation term computed by a material attached with surface. */
    float3 radiance;
    float3 attenuation;
    float3 emission;

    /** UV coordinate at intersection point. */
    float2 uv;

    /** Derivatives on texture coordinates. */
    float dpdu, dpdv;

    /** seed for random */
    unsigned int seed;

    int trace_terminate;
};

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