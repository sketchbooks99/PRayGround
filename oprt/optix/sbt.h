#pragma once

#include <optix.h>
#include "../core/material.h"
#include "../core/util.h"
#include "../optix/util.h"

namespace oprt {

/**
 * @note 
 * Should I implement a wrapper for Optix Shader Binding Table ??
 */

#ifndef __CUDACC__
template <typename T>
struct Record 
{
    __align__ (OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

#endif

}