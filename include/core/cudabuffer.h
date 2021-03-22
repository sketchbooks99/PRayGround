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

    // Enable cast to CUdeviceptr. 
    operator CUdeviceptr() { return m_ptr; }

    void alloc(size_t size) {
        Assert(!is_allocated(), "This buffer is already allocated. Please use re_alloc() if you need.");
        m_size = size;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_ptr), m_size));
    }

    /// \brief Allocation of data on the device.
    void alloc_copy(std::vector<T> data_vec) {
        alloc_copy(data_vec.data(), sizeof(T) * data_vec.size());
    }
    void alloc_copy(T* data, size_t size) {
        Assert(!is_allocated(), "This buffer is already allocated. Please use re_alloc_copy() if you need.");
        m_size = size;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_ptr), m_size));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(m_ptr),
            data, m_size, 
            cudaMemcpyHostToDevice
        ));
    }

    /// \brief Re-allocatio of data on the device. 
    void re_alloc(size_t size) {
        _init();
        this->alloc(size);
    }
    void re_alloc_copy(std::vector<T> data_vec) {
        _init();
        this->alloc_copy(data_vec);
    }
    void re_allocate(T* data, size_t size) {
        _init();
        alloc_copy(data, size);
    }

    /// \brief Free data from the device.
    void free() {
        // Assert(is_alloc, "This buffer still isn't allocated on device.");
        if (is_allocated())
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_ptr)));
        _init();
    }

    /// \brief Get state of the buffer.
    bool is_allocated() { return (bool)m_ptr; }
    CUdeviceptr d_ptr() const { return m_ptr; }
    CUdeviceptr& d_ptr() { return m_ptr; }
    T* data() { return reinterpret_cast<T*>(m_ptr); }
    size_t size() { return m_size; }
private:
    void _init() {
        m_ptr = 0;
        m_size = 0;
    }

    CUdeviceptr m_ptr { 0 };
    size_t m_size { 0 };
}; 

}