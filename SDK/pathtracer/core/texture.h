#pragma once 

#include "core_util.h"

namespace pt {

enum class TextureType {
    Constant,
    Checker,
    Image
};

#if !defined(__CUDACC__)
inline std::ostream& operator<<(std::ostream& out, TextureType type) {
    switch(type) {
    case TextureType::Constant:
        return out << "TextureType::Constant";
    case TextureType::Checker:
        return out << "TextureType::Checker";
    case TextureType::Image:
        return out << "TextureType::Image";
    }
}
#endif

class Texture {
    virtual TextureType type() const = 0;
};

}
