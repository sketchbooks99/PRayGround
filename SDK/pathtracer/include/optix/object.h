#pragma once 

#include "core_util.h"

namespace pt {

/** \brief 
 *  This class enable to use virtual function on the device.
 */ 
class DeviceCallableObject {
protected:
    virtual HOST setup_on_device() {}
    virtual HOST delete_on_device() {}
public:
    virtual DEVICE_FUNC ~Object() {}
};

}