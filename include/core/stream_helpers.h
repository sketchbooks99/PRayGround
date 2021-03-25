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
 * \struct float3
 */
inline std::ostream& operator<<(std::ostream& out, const float3& v) {
    return out << v.x << ' ' << v.y << ' ' << v.z;
}

/**
 * \struct OptixAabb
 */
inline std::ostream& operator<<(std::ostream& out, const OptixAabb& aabb) {
            out << "min: " << aabb.minX << ' ' << aabb.minY << ' ' << aabb.minZ;
    return  out << ', max: ' << aabb.maxX << ' ' << aabb.maxY << ' ' << aabb.maxZ;
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

/**
 * \enum OptixVertexFormat
 */
inline std::ostream& operator<<(std::ostream& out, const OptixVertexFormat& type) {
    switch (type) {
    case OPTIX_VERTEX_FORMAT_NONE:      return out << "OPTIX_VERTEX_FORMAT_NONE";
    case OPTIX_VERTEX_FORMAT_FLOAT3:    return out << "OPTIX_VERTEX_FORMAT_FLOAT3";
    case OPTIX_VERTEX_FORMAT_FLOAT2:    return out << "OPTIX_VERTEX_FORMAT_FLOAT2";
    case OPTIX_VERTEX_FORMAT_HALF3:     return out << "OPTIX_VERTEX_FORMAT_HALF3";
    case OPTIX_VERTEX_FORMAT_HALF2:     return out << "OPTIX_VERTEX_FORMAT_HALF2";
    case OPTIX_VERTEX_FORMAT_SNORM16_3: return out << "OPTIX_VERTEX_FORMAT_SNORM16_3";
    case OPTIX_VERTEX_FORMAT_SNORM16_2: return out << "OPTIX_VERTEX_FORMAT_SNORM16_2";
    }
}

/**
 * \enum OptixBuildInputType
 */
inline std::ostream& operator<<(std::ostream& out, const OptixBuildInputType& type) {
    switch (type) {
    case OPTIX_BUILD_INPUT_TYPE_TRIANGLES:          return out << "OPTIX_BUILD_INPUT_TYPE_TRIANGLES"; 
    case OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES:  return out << "OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES";
    case OPTIX_BUILD_INPUT_TYPE_INSTANCES:          return out << "OPTIX_BUILD_INPUT_TYPE_INSTANCES";
    case OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS:  return out << "OPTIX_BUILD_INPUT_TYPE_POINTERS";
    case OPTIX_BUILD_INPUT_TYPE_CURVES:             return out << "OPTIX_BUILD_INPUT_TYPE_CURVES";
    }
}

/**
 * \enum OptixIndicesFormat
 */
inline std::ostream& operator<<(std::ostream& out, const OptixIndicesFormat& format) {
    switch (format) {
    case OPTIX_INDICES_FORMAT_NONE:             return out << "OPTIX_INDICES_FORMAT_NONE";
    case OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3:  return out << "OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3";
    case OPTIX_INDICES_FORMAT_UNSIGNED_INT3:    return out << "OPTIX_INDICES_FORMAT_UNSIGNED_INT3";
    }
}

/**
 * \struct OPtixBuildInputTriangleArray
 */
inline std::ostream& operator<<(std::ostream& out, const OptixBuildInputTriangleArray& triangleArray) {
    out << "numVertices: " << triangleArray.numVertices << std::endl;
    out << "vertexFormat: " << triangleArray.vertexFormat << std::endl;
    out << "vertexStrideInBytes: " << triangleArray.vertexStrideInBytes << std::endl;
    out << "indexBuffer: " << triangleArray.indexBuffer << std::endl;
    out << "numIndexTriplets: " << triangleArray.numIndexTriplets << std::endl;
    out << "indexFormat: " << triangleArray.indexFormat << std::endl;
    out << "numSbtRecords: " << triangleArray.numSbtRecords << std::endl;
    out << "sbtIndexOffsetSizeInBytes: " << triangleArray.sbtIndexOffsetSizeInBytes << std::endl;
    out << "sbtIndexOffsetStrideInBytes: " << triangleArray.sbtIndexOffsetStrideInBytes << std::endl;
    out << "primitiveIndexOffset: " << triangleArray.primitiveIndexOffset << std::endl;
    return out;
}

/**
 * \struct OptixBuildInputCustomPrimitiveArray
 */
inline std::ostream& operator<<(std::ostream& out, const OptixBuildInputCustomPrimitiveArray& customArray) {
    out << "numPrimitives: " << customArray.numPrimitives << std::endl;
    out << "strideInBytes: " << customArray.strideInBytes << std::endl;
    out << "numSbtRecords: " << customArray.numSbtRecords << std::endl;
    out << "sbtIndexOffsetSizeInBytes: " << customArray.sbtIndexOffsetSizeInBytes << std::endl;
    out << "sbtIndexOffsetStrideInBytes: " << customArray.sbtIndexOffsetStrideInBytes << std::endl;
    out << "primitiveIndexOffset: " << customArray.primitiveIndexOffset << std::endl;
    return out;
}

/**
 * \struct OptixBuildInput
 */
inline std::ostream& operator<<(std::ostream& out, const OptixBuildInput& bi) {
    out << bi.type;
    switch (bi.type) {
    case OPTIX_BUILD_INPUT_TYPE_TRIANGLES:
        return out << bi.triangleArray;
    case OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES:
        return out << bi.customPrimitiveArray;
    /** \note Not implemented */
    case OPTIX_BUILD_INPUT_TYPE_INSTANCES:
    case OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS:
    case OPTIX_BUILD_INPUT_TYPE_CURVES:
        return out << "Not implemented";
    }
}

}



