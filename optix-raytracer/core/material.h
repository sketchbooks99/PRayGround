#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <sutil/vec_math.h>
#include "../core/util.h"
#include "../optix/util.h"

#ifndef __CUDACC__
    #include <map>
#endif

namespace oprt {

/**
 * \note 
 * [EN] 
 * Material class is used only in host side, because OptiX 7~ doesn't support virtual functions.
 * So, computation of bsdf are performed using direct/continuation callables, and indices of them
 * are the indices of MaterialType.
 * ex) The index of Diffuse function = (int)MaterialType::Diffuse = 0
 * 
 * [JP]
 * Optix 7~ は仮想関数をサポートしていないため、Materialクラスはホスト側でのみ使用されます。
 * BSDFの計算はcontinuation/direct callablesにを用いて行われ、それら関数のインデックスは
 * MaterialTypeのインデックスになります。
 * 例) Diffuse関数のインデックス = (int)MaterialType::Diffuse = 0
 */

enum class MaterialType {
    Diffuse = 0,
    Conductor = 1,
    Dielectric = 2,
    Disney = 3,
    Emitter = 4,
    Count = 5
};

#ifndef __CUDACC__
/**
 * \brief A map connects MaterialType to names of entry functions that describe bsdf properties.
 */
static std::map<MaterialType, const char*> sample_func_map = {
    { MaterialType::Diffuse, "sample_diffuse" },    
    { MaterialType::Conductor, "sample_conductor" },    
    { MaterialType::Dielectric, "sample_dielectric" },    
    { MaterialType::Disney, "sample_disney"},
    { MaterialType::Emitter, "sample_emitter" }
};

static std::map<MaterialType, const char*> bsdf_func_map = {
    { MaterialType::Diffuse, "bsdf_diffuse" },    
    { MaterialType::Conductor, "bsdf_conductor" },    
    { MaterialType::Dielectric, "bsdf_dielectric" },    
    { MaterialType::Disney, "bsdf_disney"},
    { MaterialType::Emitter, "bsdf_emitter" }
};

static std::map<MaterialType, const char*> pdf_func_map = {
    { MaterialType::Diffuse, "pdf_diffuse" },    
    { MaterialType::Conductor, "pdf_conductor" },    
    { MaterialType::Dielectric, "pdf_dielectric" },    
    { MaterialType::Disney, "pdf_disney"},
    { MaterialType::Emitter, "pdf_emitter" }
};



inline std::ostream& operator<<(std::ostream& out, const MaterialType& type) {
    switch(type) {
    case MaterialType::Diffuse:     return out << "MaterialType::Diffuse";
    case MaterialType::Conductor:   return out << "MaterialType::Conductor";
    case MaterialType::Dielectric:  return out << "MaterialType::Sphere";
    case MaterialType::Disney:      return out << "MaterialType::Disney";
    case MaterialType::Emitter:     return out << "MaterialType::Emitter";
    default:
        Throw("This MaterialType is not supported\n");
        return out << "";
    }
}

#endif

// Abstract class to compute scattering properties.
class Material {
public:
    virtual ~Material() {}
    
    virtual void prepareData() = 0;
    virtual void freeData() = 0;
    virtual MaterialType type() const = 0;
    
    void* devicePtr() const { return d_data; }
protected:
    void* d_data { 0 };
};

}