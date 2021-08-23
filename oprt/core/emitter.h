#pragma once

namespace oprt {

enum class EmitterType {
    Area = 0, 
    Point = 1,
    Envmap = 2,
    Count = 3
};

#ifndef __CUDACC__

class Emitter {
public:
    virtual void copyToDevice() = 0;
    virtual EmitterType type() const = 0;

    virtual void free() = 0;

    void* devicePtr() const { return reinterpret_cast<void*>(d_data); }
protected:
    void* d_data { nullptr };
};

#endif // __CUDACC__

} // ::oprt
