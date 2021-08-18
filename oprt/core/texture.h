#pragma once 

#include <oprt/core/util.h>
#include <oprt/optix/util.h>
#include <sutil/vec_math.h>

#ifndef __CUDACC__
    #include <map>
#endif

namespace oprt {

/** 
 * @todo @c TextureType will be deprecated for extendability of the oprt app
 */
enum class TextureType {
    Constant = 0,
    Checker = 1, 
    Bitmap = 2,
    Count = 3
};

#ifndef __CUDACC__
/**
 * @brief A map connects TextureType with names of entry functions that evaluate texture albedo.
 */
static std::map<TextureType, const char*> tex_eval_map = {
    { TextureType::Constant, "eval_constant" },
    { TextureType::Checker, "eval_checker" },
    { TextureType::Bitmap, "eval_bitmap" }
};

inline std::ostream& operator<<(std::ostream& out, TextureType type) {
    switch (type) {
    case TextureType::Constant: return out << "TextureType::Constant";
    case TextureType::Checker:  return out << "TextureType::Checker";
    case TextureType::Bitmap:   return out << "TextureType::Bitmap";
    default:                    return out << "";
    }
}
#endif

class Texture {
public:
    virtual TextureType type() const = 0;

    // Preparing texture data on the device.
    virtual void prepareData() = 0;
    virtual void freeData() {}

    // Get data pointer on the device.
    void* devicePtr() const 
    {
        return d_data;
    }

    void setProgramId(const int32_t prg_id)
    {
        m_prg_id = prg_id;
    }
    int32_t programId() const 
    {
        return m_prg_id;
    }
protected:
    // CUdeviceptr d_data { 0 };
    void* d_data;
    int32_t m_prg_id { -1 };
};

} // ::oprt
