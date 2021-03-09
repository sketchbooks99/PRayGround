#pragma once 

#include <optix.h>
#include "core_util.h"

namespace pt {

/** \note
 * 
 */
template <typename T>
__global__ setup_env_on_device(T** object, )

/** \brief 
 *  This class enable to use virtual function on the device.
 */ 
class Object {
protected:
    virtual HOST_ONLY setup_env_on_device(){}
    virtual HOST_ONLY delete_env_on_device(){}
public:
    virtual DEVICE_FUNC ~Object() {};
};

}