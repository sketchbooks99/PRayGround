#pragma once 

#include "core_util.h"

namespace pt {

/** 
 * \brief Initialize object on device.
 * 
 * \note Initailization must be excecuted only once.
 */
template <typename T, Args... args>
__global__ void setup_object_on_device(T** d_ptr, Args... args) {
    (*d_ptr) = new T(args...);
}

/** \brief 
 *  This class enable to use virtual function on the device.
 */ 
class DeviceCallableObject {
protected:
    virtual HOST setup_on_device(){}
    virtual HOST delete_on_device(){}
public:
    virtual DEVICE_FUNC ~Object() {}
};

}