#pragma once 

#include <optix.h>
#include "core_util.h"

namespace pt {

/** \note
 * 
 */
template <typename T>
__global__ void setup_object_on_device(CUdeviceptr d_ptr) {
    
}

/** \brief 
 *  This class enable to use virtual function on the device.
 */ 
class Object {
protected:
    virtual HOST setup_on_device(){}
    virtual HOST delete_on_device(){}
public:
    virtual DEVICE_FUNC ~Object() {};
};

}