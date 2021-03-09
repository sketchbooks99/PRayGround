#pragma once 

#include <optix.h>
#include "core_util.h"

namespace pt {

/** 
 * \brief
 * Initialize object in the device.
 * 
 * \note 
 * Object T must have copy constructor.
 */
template <typename T>
__global__ void setup_object_on_device(void** d_ptr, const T& obj) {
    (*d_ptr) = new T(obj);
}

/** \brief 
 *  This class enable to use virtual function on the device.
 */ 
class DeviceCallableObject {
protected:
    virtual HOST setup_on_device(){}
    virtual HOST delete_on_device(){}
public:
    virtual DEVICE_FUNC ~Object() {};
};

}