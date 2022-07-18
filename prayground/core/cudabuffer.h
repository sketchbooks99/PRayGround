#pragma once

#include <prayground/core/util.h>
#include <prayground/optix/macros.h>
#include <vector>

namespace prayground {

    /**  
     * @brief
     * The class to easily manage data on the device from host.
     * 
     * CUDABufferのみデバイス側のポインタをCUdeviceptrで管理している
     * 型変換後のデバイス側のポインタを取得したい場合はdeviceData()を使う
     */
    template <class T>
    class CUDABuffer {
    public:
        using Type = T;

        CUDABuffer();
        explicit CUDABuffer(const std::vector<T>& vec);
        explicit CUDABuffer(const T* data, size_t size);

        // Cast operator from CUDABuffer<T> to CUdeviceptr.
        operator CUdeviceptr() { return d_ptr; }

        void allocate(size_t size);
        void free();

        // To allocate memory and to copy data from the host to the device.
        void copyToDevice(const std::vector<T>& vec);
        void copyToDevice(const T* data, size_t size);
        void copyToDeviceAsync(const std::vector<T>& vec, const CUstream& stream);
        void copyToDeviceAsync(const T* data, size_t size, const CUstream& stream);
        T* copyFromDevice();
        /** @todo */
        // T* copyFromDeviceAsync(const CUstream& stream);

        // Get states of the buffer.
        bool isAllocated() const;
        CUdeviceptr devicePtr() const;
        T* deviceData();
        size_t size() const;
    private:
        CUdeviceptr d_ptr { 0 };
        size_t m_size { 0 };
    }; 

    // --------------------------------------------------------------------
    template <class T>
    inline CUDABuffer<T>::CUDABuffer()
    {

    }

    template <class T>
    inline CUDABuffer<T>::CUDABuffer(const std::vector<T>& vec)
    {
        copyToDevice(vec);
    }

    template <class T>
    inline CUDABuffer<T>::CUDABuffer(const T* data, size_t size)
    {
        copyToDevice(data, size);
    }

    // --------------------------------------------------------------------
    template <class T>
    inline void CUDABuffer<T>::allocate(size_t size)
    {
        if (isAllocated())
            free();
        m_size = size;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ptr), m_size));
    }

    template <class T>
    inline void CUDABuffer<T>::free()
    {
        if (isAllocated())
            cuda_free(d_ptr);
        d_ptr = 0;
        m_size = 0;
    }

    // --------------------------------------------------------------------
    template <class T>
    inline void CUDABuffer<T>::copyToDevice(const std::vector<T>& vec)
    {
        copyToDevice(vec.data(), sizeof(T) * vec.size());
    }

    template <class T>
    inline void CUDABuffer<T>::copyToDevice(const T* data, size_t size)
    {
        if (!isAllocated() || m_size != size)
            allocate(size);
    
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_ptr), 
            data, size, 
            cudaMemcpyHostToDevice
        ));
    }

    // --------------------------------------------------------------------
    template <class T>
    inline void CUDABuffer<T>::copyToDeviceAsync(const std::vector<T>& vec, const CUstream& stream)
    {
        copyToDeviceAsync(vec.data(), sizeof(T) * vec.size(), stream);
    }

    template <class T>
    inline void CUDABuffer<T>::copyToDeviceAsync(const T* data, size_t size, const CUstream& stream)
    {
        if (!isAllocated())
            allocate(size);
    
        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void*>(d_ptr), 
            data, size, 
            cudaMemcpyHostToDevice, stream
        )); 
    }

    // --------------------------------------------------------------------
    template <class T>
    inline T* CUDABuffer<T>::copyFromDevice()
    {
        T* h_ptr = static_cast<T*>(malloc(m_size));
        if (!isAllocated()) {
            pgLogFatal("The device-side data hasn't been allocated yet.");
            return nullptr;
        }
        CUDA_CHECK(cudaMemcpy(
            h_ptr, 
            reinterpret_cast<T*>(d_ptr), 
            m_size, 
            cudaMemcpyDeviceToHost 
        ));
        return h_ptr;
    }

    // template <class T>
    // T* CUDABuffer<T>::copyFromDeviceAsync()
    // {

    // }

    // --------------------------------------------------------------------
    template <class T>
    inline bool CUDABuffer<T>::isAllocated() const 
    {
        return static_cast<bool>(d_ptr);
    }

    template <class T>
    inline CUdeviceptr CUDABuffer<T>::devicePtr() const 
    {
        return d_ptr;
    }

    template <class T>
    inline T* CUDABuffer<T>::deviceData()
    {
        return reinterpret_cast<T*>(d_ptr);
    }

    template <class T>
    inline size_t CUDABuffer<T>::size() const
    {
        return m_size;
    }

} // namespace prayground
