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
    }
}

/** 
 * \enum OptixCompileOptimizationLevel
 */
inline std::ostream& operator<<(std::ostream& out, const OptixCompileOptimizationLevel& level) {
    switch (level) {
    case OPTIX_COMPILE_OPTIMIZATION_DEFAULT: 
        return out << "OPTIX_COMPILE_OPTIMIZATION_DEFAULT";
    case OPTIX_COMPILE_OPTIMIZATION_LEVEL_0: 
        return out << "OPTIX_COMPILE_OPTIMIZATION_LEVEL_0";
    case OPTIX_COMPILE_OPTIMIZATION_LEVEL_1: 
        return out << "OPTIX_COMPILE_OPTIMIZATION_LEVEL_1";
    case OPTIX_COMPILE_OPTIMIZATION_LEVEL_2: 
        return out << "OPTIX_COMPILE_OPTIMIZATION_LEVEL_2";
    case OPTIX_COMPILE_OPTIMIZATION_LEVEL_3: 
        return out << "OPTIX_COMPILE_OPTIMIZATION_LEVEL_3";
    }
}

/** 
 * \struct OptixCompileDebugLevel
 */
inline std::ostream& operator<<(std::ostream& out, const OptixCompileDebugLevel& level) {
    switch (level) {
    case OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT:  
        return out << "OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT";
    case OPTIX_COMPILE_DEBUG_LEVEL_NONE:     
        return out << "OPTIX_COMPILE_DEBUG_LEVEL_NONE";
    case OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO: 
        return out << "OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO";
    case OPTIX_COMPILE_DEBUG_LEVEL_FULL:     
        return out << "OPTIX_COMPILE_DEBUG_LEVEL_FULL";
    }
}

/**
 * \struct OptixModuleCompileOptions
 */
inline std::ostream& operator<<(std::ostream& out, const OptixModuleCompileOptions& cop) {
            out << "maxRegisterCount: " << cop.maxRegisterCount << std::endl;
            out << "optLevel: "         << cop.optLevel << std::endl;
            out << "debugLevel: "       << cop.debugLevel << std::endl;
    return  out << "numBoundValues: "   << cop.numBoundValues;
}

/**
 * \struct OptixPipelineCompileOptions
 */
inline std::ostream& operator<<(std::ostream& out, const OptixPipelineCompileOptions& cop) {
            out << "usesMotionBlur: " << cop.usesMotionBlur << std::endl;
            out << "traversableGraphFlags: " << cop.traversableGraphFlags << std::endl;
            out << "numPayloadValues: " << cop.numPayloadValues << std::endl;
            out << "numAttributeValues: " << cop.numAttributeValues << std::endl;
            out << "exceptionFlags: " << cop.exceptionFlags << std::endl;
            out << "pipelineLaunchParamVariableName: " << cop.pipelineLaunchParamsVariableName << std::endl;
    return  out << "usePrimitiveTypeFlags: " << cop.usesPrimitiveTypeFlags;
}

}



