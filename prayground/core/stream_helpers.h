#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>

namespace prayground {

    /** @todo Make this be pgToString() */
    template <typename T>
    std::string toString(T t) 
    {
        std::ostringstream oss;
        oss << t;
        return oss.str();
    }

    /**
     * @struct float2
     */
    inline std::ostream& operator<<(std::ostream& out, const float2& v) 
    {
        return out << v.x << ' ' << v.y;
    }

    /** 
     * @struct float3
     */
    inline std::ostream& operator<<(std::ostream& out, const float3& v) 
    {
        return out << v.x << ' ' << v.y << ' ' << v.z;
    }

    /**
     * @struct float4
     */
    inline std::ostream& operator<<(std::ostream& out, const float4& v)
    {
        return out << v.x << ' ' << v.y << ' ' << v.z << ' ' << v.w;
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

} // namespace prayground



