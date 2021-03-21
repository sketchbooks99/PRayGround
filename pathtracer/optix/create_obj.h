#pragma once

#include <optix.h>
#include <sutil/vec_math.h>

// Forward declaration 

template <typename T, typename... Args>
void setup_object_on_device(T** d_ptr, Args ...args);

template <typename T>
void delete_object_on_device(T* d_ptr);