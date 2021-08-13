#pragma once 

#include "../core/util.h"
// #include "../core/cudabuffer.h"
#include "../optix/util.h"
#include <sutil/vec_math.h>

#ifndef __CUDACC__
    #include <map>
#endif

namespace oprt {

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
    case TextureType::Bitmap:    return out << "TextureType::Bitmap";
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
    void* devicePtr() const { return d_data; }
protected:
    // CUdeviceptr d_data { 0 };
    void* d_data;
};

using TexturePtr = std::shared_ptr<Texture>;

} // ::oprt
