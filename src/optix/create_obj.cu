#include <include/optix/util.h>

/** 
 * \brief Initialize object on device.
 * 
 * \note Initailization must be excecuted only once.
 */
template <typename T, typename... Args>
__global__ void setup_object(T** d_ptr, Args... args) {
    (*d_ptr) = new T(args...);
}

template <typename T>
__global__ void delete_object(T** d_ptr) {
    delete (void*)*d_ptr;
}

template <typename T, typename... Args>
void setup_object_on_device(T** d_ptr, Args... args) {
	setup_object<<<1,1>>>(d_ptr, args...);
}

template <typename T>
void delete_object_on_device(T** d_ptr) {
	delete_object<<<1,1>>>(d_ptr);
}