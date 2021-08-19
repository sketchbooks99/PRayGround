#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <sutil/vec_math.h>
#include <oprt/core/util.h>

#ifndef __CUDACC__
    #include <map>
#endif

namespace oprt {

/**
 * @todo
 * @c MaterialType will be deprecated for extendability of the oprt app
 */

enum class MaterialType {
    Diffuse = 0,
    Conductor = 1,
    Dielectric = 2,
    Disney = 3,
    // Emitter = 4,
    Count = 4
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
    // { MaterialType::Emitter, "sample_emitter" }
};

static std::map<MaterialType, const char*> bsdf_func_map = {
    { MaterialType::Diffuse, "bsdf_diffuse" },    
    { MaterialType::Conductor, "bsdf_conductor" },    
    { MaterialType::Dielectric, "bsdf_dielectric" },    
    { MaterialType::Disney, "bsdf_disney"},
    // { MaterialType::Emitter, "bsdf_emitter" }
};

static std::map<MaterialType, const char*> pdf_func_map = {
    { MaterialType::Diffuse, "pdf_diffuse" },    
    { MaterialType::Conductor, "pdf_conductor" },    
    { MaterialType::Dielectric, "pdf_dielectric" },    
    { MaterialType::Disney, "pdf_disney"},
    // { MaterialType::Emitter, "pdf_emitter" }
};



inline std::ostream& operator<<(std::ostream& out, const MaterialType& type) {
    switch(type) {
    case MaterialType::Diffuse:     return out << "MaterialType::Diffuse";
    case MaterialType::Conductor:   return out << "MaterialType::Conductor";
    case MaterialType::Dielectric:  return out << "MaterialType::Sphere";
    case MaterialType::Disney:      return out << "MaterialType::Disney";
    // case MaterialType::Emitter:     return out << "MaterialType::Emitter";
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
    
    virtual void copyToDevice() = 0;
    virtual MaterialType type() const = 0;

    virtual void free()
    {
        if (d_data) cuda_free(d_data);
    }
    
    void* devicePtr() const { return d_data; }

    void addProgramId(const uint32_t idx)
    {
        m_prg_ids.push_back(idx);
    }

    int32_t programIdAt(const int32_t idx) const
    {
        if (idx >= m_prg_ids.size())
        {
            Message(MSG_ERROR, "oprt::Material::funcIdAt(): Index to get function id of material exceeds the number of functions");
            return -1;
        }
        return static_cast<int32_t>(m_prg_ids[idx]);
    }
protected:
    void* d_data { 0 };
    std::vector<uint32_t> m_prg_ids;
};

} // ::oprt