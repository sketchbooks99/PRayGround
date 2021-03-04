#include <optix.h>

namespace pt {

/**  
 * The class to easily manage data on device from host.
 * This is used in only host code.  
 */

template <typename T>
class CUDABuffer {
public:
    explicit CUDABuffer(T* data, size_t size)
    {
        allocate(data, size);
    }

    void init() {
        m_ptr = 0;
        is_alloc = false;
        m_size = 0;
    }

    void allocate(std::vector<T> data_vec) {
        allocate(data_vec.size(), sizeof(T) * data_vec.size());
    }

    void allocate(T* data, size_t size) {
        Assert(!is_alloc, "This buffer is already allocated. Please use re_allocate() if you need.");
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_ptr), size));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(ptr),
            data, size, 
            cudaMemcpyHostToDevice
        ));
        is_alloc = true;
    }

    void re_allocate(std::vector<T> data_vec) {
        init();
        allocate(data_vec);
    }
    void re_allocate(T* data, size_t size) {
        init();
        allocate(data, size);
    }

    void free() {
        Assert(is_alloc, "This buffer still isn't allocated on device.");
        OPTIX_CHECK(cudaFree(reinterpret_cast<void*>(m_ptr)));
        is_alloc = false;
    }

    bool is_allocated() { return is_alloc; }
    CUdeviceptr d_ptr() { return m_ptr; }
    T* data() { return reinterpret_cast<T*>(m_ptr); }
    size_t size() { return m_size; }
private:
    CUdeviceptr m_ptr = 0;
    bool is_alloc = false;
    size_t m_size;
}; 

}