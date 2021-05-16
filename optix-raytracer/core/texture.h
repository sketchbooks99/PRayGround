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
    Image = 2,
    Count = 3
};

#ifndef __CUDACC__
/**
 * @brief A map connects TextureType with names of entry functions that evaluate texture albedo.
 */
static std::map<TextureType, const char*> tex_eval_map = {
    { TextureType::Constant, "eval_constant" }, 
    { TextureType::Checker, "eval_checker" },
    { TextureType::Image, "eval_image" }
};

inline std::ostream& operator<<(std::ostream& out, TextureType type) {
    switch (type) {
    case TextureType::Constant: return out << "TextureType::Constant";
    case TextureType::Checker:  return out << "TextureType::Checker";
    case TextureType::Image:    return out << "TextureType::Image";
    default:                    return out << "";
    }
}
#endif

class Texture {
public:
    virtual TextureType type() const = 0;

    // Preparing texture data on the device.
    virtual void prepare_data() = 0;

    // Get data pointer on the device.
    void* get_dptr() const { return d_data; }
protected:
    // CUdeviceptr d_data { 0 };
    void* d_data;
};

}
