#pragma once

#include <optix.h>
#include "core_util.h"

namespace pt {

template <typaname T>
struct Record 
{
    __align__ (OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

}