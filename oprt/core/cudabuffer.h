#pragma once

#include <sutil/Exception.h>
#include "../optix/util.h"

namespace oprt {

/**  
 * \brief
 * The class to easily manage data on the device from host.
 */

template <typename T>
class CUDABuffer {
public:
    CUDABuffer() {}
    explicit CUDABuffer(std::vector<T> vec) : CUDABuffer(vec.data(), vec.size()*sizeof(T)) {}
    explicit CUDABuffer(T* data, size_t size)
    {
        copyToDevice(data, size);
    }

    // Cast operator from CUDABuffer<T> to CUdeviceptr.
    operator CUdeviceptr() { return m_ptr; }

    void allocate(size_t size) {
        Assert(!isAllocated(), "This buffer is already allocated. Please init() before an allocation");
        m_size = size;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_ptr), m_size));
    }

    // To allocate memory and to copy data from the host to the device.
    void copyToDevice(std::vector<T> data_vec) {
        copyToDevice(data_vec.data(), sizeof(T) * data_vec.size());
    }
    void copyToDevice(T* data, size_t size) {
        allocate(size);
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(m_ptr),
            data, m_size, 
            cudaMemcpyHostToDevice
        ));
    }

    // Free data from the device.
    void free() {
        if (isAllocated())
            cuda_free(m_ptr);
        m_ptr = 0;
        m_size = 0;
    }

    // Get states of the buffer.
    bool isAllocated() { return (bool)m_ptr; }
    CUdeviceptr devicePtr() const { return m_ptr; }
    T* deviceData() { return reinterpret_cast<T*>(m_ptr); }
    size_t size() { return m_size; }
private:
    CUdeviceptr m_ptr { 0 };
    size_t m_size { 0 };
}; 

}