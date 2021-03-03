#include <optix.h>

template <typename T>
class CudaBuffer {
public:
    void allocate(T* data, size_t size) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), size));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(ptr),
            data, size, 
            cudaMemcpyHostToDevice
        ));
        isOnDevice = true;
    }

    bool isOnDevice() { return ptr; }
    CUdeviceptr d_ptr() { return ptr; }
    T* data() { return reinterpret_cast<T*>(ptr); }
private:
    CUdeviceptr ptr;
    bool onDevice() {}
}