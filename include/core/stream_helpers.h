#pragma once

#include <optix.h>
#include <iostream>

namespace pt {

template <typename T>
std::string to_str(T t) {
    std::ostringstream oss;
    oss << t;
    return oss.str();
}

/**
 * \enum OptixProgramGroupKind
 **/
inline std::ostream& operator<<(std::ostream& out, const OptixProgramGroupKind& kind) {
    switch(kind) {
    case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
        return out << "OPTIX_PROGRAM_GROUP_KIND_RAYGEN";
    case OPTIX_PROGRAM_GROUP_KIND_MISS:
        return out << "OPTIX_PROGRAM_GROUP_KIND_MISS";
    case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION:
        return out << "OPTIX_PROGRAM_GROUP_KIND_EXCEPTION";
    case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
        return out << "OPTIX_PROGRAM_GROUP_KIND_HITGROUP";
    case OPTIX_PROGRAM_GROUP_KIND_CALLABLES:
        return out << "OPTIX_PROGRAM_GROUP_KIND_CALLABLES";
    default:
        return out << "";
    }
}

/**
 * \struct OptixModuleCompileOptions
 */

inline std::ostream& operator<<(std::ostream& out, const OptixModuleCompileOptions& cop) {
    out << 
}

}



