#pragma once

#include <sutil/Exception.h>
#include <include/optix/util.h>

namespace pt {

/**  
 * The class to easily manage data on device from host.
 * This is used in only host code.  
 */

template <typename T>
class CUDABuffer {
public:
    CUDABuffer() { _init(); }
    explicit CUDABuffer(std::vector<T> vec) : CUDABuffer(vec.data(), vec.size()*sizeof(T)) {}
    explicit CUDABuffer(T* data, size_t size)
    {
        _init(); 
        alloc_copy(data, size);
    }

    ~CUDABuffer() { free(); }

    // Enable cast to CUdeviceptr. 
    operator CUdeviceptr() { return d_ptr; }

    void alloc(size_t size) {
        Assert(!is_alloc, "This buffer is already allocated. Please use re_allocate() if you need.");
        m_size = size;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_ptr), m_size));
    }

    /// \brief Allocation of data on the device.
    void alloc_copy(std::vector<T> data_vec) {
        alloc_copy(data_vec.data(), sizeof(T) * data_vec.size());
    }
    void alloc_copy(T* data, size_t size) {
        Assert(!is_alloc, "This buffer is already allocated. Please use re_allocate() if you need.");
        m_size = size;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_ptr), m_size));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(m_ptr),
            data, m_size, 
            cudaMemcpyHostToDevice
        ));
        is_alloc = true;
    }

    /// \brief Re-allocatio of data on the device. 
    void re_allocate(std::vector<T> data_vec) {
        _init();
        allocate(data_vec);
    }
    void re_allocate(T* data, size_t size) {
        _init();
        allocate(data, size);
    }

    /// \brief Free data from the device.
    void free() {
        Assert(is_alloc, "This buffer still isn't allocated on device.");
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_ptr)));
        _init();
    }

    /// \brief Get state of the buffer.
    bool is_allocated() { return is_alloc; }
    CUdeviceptr d_ptr() { return m_ptr; }
    T* data() { return reinterpret_cast<T*>(m_ptr); }
    size_t size() { return m_size; }
private:
    void _init() {
        m_ptr = 0;
        is_alloc = false;
        m_size = 0;
    }

    CUdeviceptr m_ptr;
    bool is_alloc;
    size_t m_size;
}; 

}