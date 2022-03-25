#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>

namespace prayground {

template <typename T>
std::string toString(T t) {
    std::ostringstream oss;
    oss << t;
    return oss.str();
}

/** 
 * @struct float3
 */
inline std::ostream& operator<<(std::ostream& out, const float3& v) {
    return out << v.x << ' ' << v.y << ' ' << v.z;
}

/**
 * @struct float2
 */
inline std::ostream& operator<<(std::ostream& out, const float2& v) {
    return out << v.x << ' ' << v.y;
}

// For OptiX ======================================================================
/**
 * @struct OptixResult 
 */
inline std::ostream& operator<<(std::ostream& out, const OptixResult& result) {
    switch( result ) {
    case OPTIX_SUCCESS:                                 return out << "OPTIX_SUCCESS";
    case OPTIX_ERROR_INVALID_VALUE:                     return out << "OPTIX_ERROR_INVALID_VALUE";
    case OPTIX_ERROR_HOST_OUT_OF_MEMORY:                return out << "OPTIX_ERROR_HOST_OUT_OF_MEMORY";
    case OPTIX_ERROR_INVALID_OPERATION:                 return out << "OPTIX_ERROR_INVALID_OPERATION";
    case OPTIX_ERROR_FILE_IO_ERROR:                     return out << "OPTIX_ERROR_FILE_IO_ERROR";
    case OPTIX_ERROR_INVALID_FILE_FORMAT:               return out << "OPTIX_ERROR_INVALID_FILE_FORMAT";
    case OPTIX_ERROR_DISK_CACHE_INVALID_PATH:           return out << "OPTIX_ERROR_DISK_CACHE_INVALID_PATH";
    case OPTIX_ERROR_DISK_CACHE_PERMISSION_ERROR:       return out << "OPTIX_ERROR_DISK_CACHE_PERMISSION_ERROR";
    case OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR:         return out << "OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR";
    case OPTIX_ERROR_DISK_CACHE_INVALID_DATA:           return out << "OPTIX_ERROR_DISK_CACHE_INVALID_DATA";
    case OPTIX_ERROR_LAUNCH_FAILURE:                    return out << "OPTIX_ERROR_LAUNCH_FAILURE";
    case OPTIX_ERROR_INVALID_DEVICE_CONTEXT:            return out << "OPTIX_ERROR_INVALID_DEVICE_CONTEXT";
    case OPTIX_ERROR_CUDA_NOT_INITIALIZED:              return out << "OPTIX_ERROR_CUDA_NOT_INITIALIZED";
    case OPTIX_ERROR_VALIDATION_FAILURE:                return out << "OPTIX_ERROR_VALIDATION_FAILURE";
    case OPTIX_ERROR_INVALID_PTX:                       return out << "OPTIX_ERROR_INVALID_PTX";
    case OPTIX_ERROR_INVALID_LAUNCH_PARAMETER:          return out << "OPTIX_ERROR_INVALID_LAUNCH_PARAMETER";
    case OPTIX_ERROR_INVALID_PAYLOAD_ACCESS:            return out << "OPTIX_ERROR_INVALID_PAYLOAD_ACCESS";
    case OPTIX_ERROR_INVALID_ATTRIBUTE_ACCESS:          return out << "OPTIX_ERROR_INVALID_ATTRIBUTE_ACCESS";
    case OPTIX_ERROR_INVALID_FUNCTION_USE:              return out << "OPTIX_ERROR_INVALID_FUNCTION_USE";
    case OPTIX_ERROR_INVALID_FUNCTION_ARGUMENTS:        return out << "OPTIX_ERROR_INVALID_FUNCTION_ARGUMENTS";
    case OPTIX_ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY:   return out << "OPTIX_ERROR_PIPELINE_OUT_OF_CONSTANT_MEMORY";
    case OPTIX_ERROR_PIPELINE_LINK_ERROR:               return out << "OPTIX_ERROR_PIPELINE_LINK_ERROR";
    case OPTIX_ERROR_INTERNAL_COMPILER_ERROR:           return out << "OPTIX_ERROR_INTERNAL_COMPILER_ERROR";
    case OPTIX_ERROR_DENOISER_MODEL_NOT_SET:            return out << "OPTIX_ERROR_DENOISER_MODEL_NOT_SET";
    case OPTIX_ERROR_DENOISER_NOT_INITIALIZED:          return out << "OPTIX_ERROR_DENOISER_NOT_INITIALIZED";
    case OPTIX_ERROR_ACCEL_NOT_COMPATIBLE:              return out << "OPTIX_ERROR_ACCEL_NOT_COMPATIBLE";
    case OPTIX_ERROR_NOT_SUPPORTED:                     return out << "OPTIX_ERROR_NOT_SUPPORTED";
    case OPTIX_ERROR_UNSUPPORTED_ABI_VERSION:           return out << "OPTIX_ERROR_UNSUPPORTED_ABI_VERSION";
    case OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH:      return out << "OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH";
    case OPTIX_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS:    return out << "OPTIX_ERROR_INVALID_ENTRY_FUNCTION_OPTIONS";
    case OPTIX_ERROR_LIBRARY_NOT_FOUND:                 return out << "OPTIX_ERROR_LIBRARY_NOT_FOUND";
    case OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND:            return out << "OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND";
    case OPTIX_ERROR_LIBRARY_UNLOAD_FAILURE:            return out << "OPTIX_ERROR_LIBRARY_UNLOAD_FAILURE";
    case OPTIX_ERROR_CUDA_ERROR:                        return out << "OPTIX_ERROR_CUDA_ERROR";
    case OPTIX_ERROR_INTERNAL_ERROR:                    return out << "OPTIX_ERROR_INTERNAL_ERROR";
    case OPTIX_ERROR_UNKNOWN:                           return out << "OPTIX_ERROR_UNKNOWN";
    default:                                            return out << "";
    }
}

/**
 * @struct OptixAabb
 */
inline std::ostream& operator<<(std::ostream& out, const OptixAabb& aabb) {
            out << "min: " << aabb.minX << ' ' << aabb.minY << ' ' << aabb.minZ;
    return  out << ", max: " << aabb.maxX << ' ' << aabb.maxY << ' ' << aabb.maxZ;
}

/**
 * @enum OptixProgramGroupKind
 **/
inline std::ostream& operator<<(std::ostream& out, const OptixProgramGroupKind& kind) {
    switch(kind) {
    case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:    return out << "OPTIX_PROGRAM_GROUP_KIND_RAYGEN";
    case OPTIX_PROGRAM_GROUP_KIND_MISS:      return out << "OPTIX_PROGRAM_GROUP_KIND_MISS";
    case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION: return out << "OPTIX_PROGRAM_GROUP_KIND_EXCEPTION";
    case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:  return out << "OPTIX_PROGRAM_GROUP_KIND_HITGROUP";
    case OPTIX_PROGRAM_GROUP_KIND_CALLABLES: return out << "OPTIX_PROGRAM_GROUP_KIND_CALLABLES";
    default:                                 return out << "";
    }
}

/** 
 * @enum OptixCompileOptimizationLevel
 */
inline std::ostream& operator<<(std::ostream& out, const OptixCompileOptimizationLevel& level) {
    switch (level) {
    case OPTIX_COMPILE_OPTIMIZATION_DEFAULT: return out << "OPTIX_COMPILE_OPTIMIZATION_DEFAULT";
    case OPTIX_COMPILE_OPTIMIZATION_LEVEL_0: return out << "OPTIX_COMPILE_OPTIMIZATION_LEVEL_0";
    case OPTIX_COMPILE_OPTIMIZATION_LEVEL_1: return out << "OPTIX_COMPILE_OPTIMIZATION_LEVEL_1";
    case OPTIX_COMPILE_OPTIMIZATION_LEVEL_2: return out << "OPTIX_COMPILE_OPTIMIZATION_LEVEL_2";
    case OPTIX_COMPILE_OPTIMIZATION_LEVEL_3: return out << "OPTIX_COMPILE_OPTIMIZATION_LEVEL_3";
    default:                                 return out << "";
    }
}

/**
 * @enum OptixDeviceContextValidationMode
 */
inline std::ostream& operator<<(std::ostream& out, const OptixDeviceContextValidationMode& mode)
{
    switch (mode)
    {
        case OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF: return out << "OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF";
        case OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL: return out << "OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL";
    }
}

/**
 * @struct OptixDeviceContextOptions
 */
inline std::ostream& operator<<(std::ostream& out, const OptixDeviceContextOptions& options)
{
            out << "logCallbackLevel: " << options.logCallbackLevel << std::endl;
    return  out << "validationMode: "   << options.validationMode; 
}

/** 
 * @struct OptixCompileDebugLevel
 */
inline std::ostream& operator<<(std::ostream& out, const OptixCompileDebugLevel& level) {
    switch (level) {
    case OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT:  return out << "OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT";
    case OPTIX_COMPILE_DEBUG_LEVEL_NONE:     return out << "OPTIX_COMPILE_DEBUG_LEVEL_NONE";
#if OPTIX_VERSION == 70400
    case OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL:  return out << "OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL";
    case OPTIX_COMPILE_DEBUG_LEVEL_MODERATE:  return out << "OPTIX_COMPILE_DEBUG_LEVEL_MODERATE";
#else 
    case OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO: return out << "OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO";
#endif
    case OPTIX_COMPILE_DEBUG_LEVEL_FULL:     return out << "OPTIX_COMPILE_DEBUG_LEVEL_FULL";
    default:                                 return out << "";
    }
}

/**
 * @struct OptixModuleCompileOptions
 */
inline std::ostream& operator<<(std::ostream& out, const OptixModuleCompileOptions& cop) {
            out << "maxRegisterCount: " << cop.maxRegisterCount << std::endl;
            out << "optLevel: "         << cop.optLevel << std::endl;
            out << "debugLevel: "       << cop.debugLevel << std::endl;
    return  out << "numBoundValues: "   << cop.numBoundValues;
}

/**
 * @struct OptixPipelineCompileOptions
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
 * @struct
 */
inline std::ostream& operator<<(std::ostream& out, const OptixPipelineLinkOptions& lop) {
            out << "maxTraceDepth: " <<  lop.maxTraceDepth << std::endl;
    return  out << "debugLevel: " << lop.debugLevel;
}

/**
 * @enum OptixVertexFormat
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
    default:                            return out << "";
    }
}

/**
 * @enum OptixBuildInputType
 */
inline std::ostream& operator<<(std::ostream& out, const OptixBuildInputType& type) {
    switch (type) {
    case OPTIX_BUILD_INPUT_TYPE_TRIANGLES:          return out << "OPTIX_BUILD_INPUT_TYPE_TRIANGLES"; 
    case OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES:  return out << "OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES";
    case OPTIX_BUILD_INPUT_TYPE_INSTANCES:          return out << "OPTIX_BUILD_INPUT_TYPE_INSTANCES";
    case OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS:  return out << "OPTIX_BUILD_INPUT_TYPE_POINTERS";
    case OPTIX_BUILD_INPUT_TYPE_CURVES:             return out << "OPTIX_BUILD_INPUT_TYPE_CURVES";
    default:                                        return out << "";
    }
}

/**
 * @enum OptixIndicesFormat
 */
inline std::ostream& operator<<(std::ostream& out, const OptixIndicesFormat& format) {
    switch (format) {
    case OPTIX_INDICES_FORMAT_NONE:             return out << "OPTIX_INDICES_FORMAT_NONE";
    case OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3:  return out << "OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3";
    case OPTIX_INDICES_FORMAT_UNSIGNED_INT3:    return out << "OPTIX_INDICES_FORMAT_UNSIGNED_INT3";
    default:                                    return out << "";
    }
}

/**
 * @struct OptixBuildInputTriangleArray
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
    out << "primitiveIndexOffset: " << triangleArray.primitiveIndexOffset;
    return out;
}

/**
 * @struct OptixBuildInputCustomPrimitiveArray
 */
inline std::ostream& operator<<(std::ostream& out, const OptixBuildInputCustomPrimitiveArray& customArray) {
    out << "numPrimitives: " << customArray.numPrimitives << std::endl;
    out << "strideInBytes: " << customArray.strideInBytes << std::endl;
    out << "numSbtRecords: " << customArray.numSbtRecords << std::endl;
    out << "sbtIndexOffsetSizeInBytes: " << customArray.sbtIndexOffsetSizeInBytes << std::endl;
    out << "sbtIndexOffsetStrideInBytes: " << customArray.sbtIndexOffsetStrideInBytes << std::endl;
    out << "primitiveIndexOffset: " << customArray.primitiveIndexOffset;
    return out;
}

/**
 * @struct OptixBuildInput
 */
inline std::ostream& operator<<(std::ostream& out, const OptixBuildInput& bi) {
    out << "type: " << bi.type << std::endl;
    switch (bi.type) {
    case OPTIX_BUILD_INPUT_TYPE_TRIANGLES:
        return out << bi.triangleArray << std::endl;
    case OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES:
        return out << bi.customPrimitiveArray << std::endl;
    /** \note Not implemented */
    case OPTIX_BUILD_INPUT_TYPE_INSTANCES:
    case OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS:
    case OPTIX_BUILD_INPUT_TYPE_CURVES:
        return out << "Not implemented";
    default: 
        return out << "";
    }
}

/**
 * @struct OptixInstance
 */
inline std::ostream& operator<<(std::ostream& out, const OptixInstance& ins) {
    out << "transform: " << std::endl;
    out << ins.transform[0] << ' ' << ins.transform[1] << ' '<< ins.transform[2] << ' ' << ins.transform[3] << std::endl;
    out << ins.transform[4] << ' ' << ins.transform[5] << ' '<< ins.transform[6] << ' ' << ins.transform[7] << std::endl;
    out << ins.transform[8] << ' ' << ins.transform[9] << ' '<< ins.transform[10] << ' ' << ins.transform[11] << std::endl;
    out << "instanceId: " << ins.instanceId << std::endl;
    out << "sbtOffset: " << ins.sbtOffset << std::endl;
    out << "visibilityMask: " << ins.visibilityMask << std::endl;
    out << "flags: " << ins.flags << std::endl;
    out << "traversableHandle: " << ins.traversableHandle << std::endl;
    return out << "pad: " << ins.pad[0] << ' ' << ins.pad[1];
}

/**
 * @enum OptixBuildFlags 
 */
inline std::ostream& operator<<(std::ostream& out, OptixBuildFlags build_flags)
{
    switch (build_flags)
    {
    case OPTIX_BUILD_FLAG_NONE:                       return out << "OPTIX_BUILD_FLAG_NONE";
    case OPTIX_BUILD_FLAG_ALLOW_UPDATE:               return out << "OPTIX_BUILD_FLAG_ALLOW_UPDATE";
    case OPTIX_BUILD_FLAG_ALLOW_COMPACTION:           return out << "OPTIX_BUILD_FLAG_ALLOW_COMPACTION";
    case OPTIX_BUILD_FLAG_PREFER_FAST_TRACE:          return out << "OPTIX_BUILD_FLAG_PREFER_FAST_TRACE";
    case OPTIX_BUILD_FLAG_PREFER_FAST_BUILD:          return out << "OPTIX_BUILD_FLAG_PREFER_FAST_BUILD";
    case OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS: return out << "OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS";
    default:                                          return out << "";
    }
}

/**
 * @enum OptixBuildOperation
 */
inline std::ostream& operator<<(std::ostream& out, OptixBuildOperation build_operation)
{
    switch (build_operation)
    {
    case OPTIX_BUILD_OPERATION_BUILD:  return out << "OPTIX_BUILD_OPERATION_BUILD";
    case OPTIX_BUILD_OPERATION_UPDATE: return out << "OPTIX_BUILD_OPERATION_UPDATE";
    default:                           return out;
    }
}

// For CUDA ======================================================================
/**
 * @enum cudaTextureAddressMode
 */
inline std::ostream& operator<<( std::ostream& out, const cudaTextureAddressMode& address_mode ) 
{
    switch (address_mode) 
    {
    case cudaAddressModeWrap:   return out << "cudaAddressModeWrap";
    case cudaAddressModeClamp:  return out << "cudaAddressModeClamp";
    case cudaAddressModeMirror: return out << "cudaAddressModeMirror";
    case cudaAddressModeBorder: return out << "cudaAddressModeBorder";
    default:                    return out;
    }
}

/**
 * @enum cudaTextureFilterMode
 */
inline std::ostream& operator<<( std::ostream& out, const cudaTextureFilterMode& filter_mode )
{
    switch (filter_mode)
    {
    case cudaFilterModePoint:  return out << "cudaFilterModePoint";
    case cudaFilterModeLinear: return out << "cudaFilterModeLinear";
    default:                   return out;
    }
}

/**
 * @enum cudaTextureReadMode
 */
inline std::ostream& operator<<( std::ostream& out, const cudaTextureReadMode& read_mode )
{
    switch (read_mode)
    {
    case cudaReadModeElementType:     return out << "cudaReadModeElementType";
    case cudaReadModeNormalizedFloat: return out << "cudaReadModeNormalizedFloat";
    default:                          return out;
    }
}

/**
 * @struct cudaTextureDesc
 */
inline std::ostream& operator<<( std::ostream& out, const cudaTextureDesc& tex_desc )
{
    out << "cudaTextureDesc {" << std::endl;
    out << "\taddressMode[0]: " << tex_desc.addressMode[0] << std::endl;
    out << "\taddressMode[1]: " << tex_desc.addressMode[1] << std::endl;
    out << "\taddressMode[2]: " << tex_desc.addressMode[2] << std::endl;
    out << "\tfilterMode: " << tex_desc.filterMode << std::endl;
    out << "\treadMode: " << tex_desc.readMode << std::endl;
    out << "\tsRGB: " << tex_desc.sRGB << std::endl;
    out << "\tborderColor: " << tex_desc.borderColor[0] << ' ' << tex_desc.borderColor[1] << ' ' 
                           << tex_desc.borderColor[2] << ' ' << tex_desc.borderColor[3] << std::endl;
    out << "\tnormalizedCoords: " << tex_desc.normalizedCoords << std::endl;
    out << "\tmaxAnisotropy: " << tex_desc.maxAnisotropy << std::endl;
    out << "\tmipmapFilterMode: " << tex_desc.mipmapFilterMode << std::endl;
    out << "\tmipmapLevelBias: " << tex_desc.mipmapLevelBias << std::endl;
    out << "\tminMipmapLevelClamp: " << tex_desc.minMipmapLevelClamp << std::endl;
    out << "\tmaxMipmapLevelClamp: " << tex_desc.maxMipmapLevelClamp << std::endl;
    out << "\tdisableTrilinearOptimization: " << tex_desc.disableTrilinearOptimization << std::endl;
    out << "}";
    return out;
}

}



