#include <include/optix/util.h>
#include <sutil/Exception.h>
#include "../material/conductor.h"
#include "../material/dielectric.h"
#include "../material/diffuse.h"
#include "../material/emitter.h"


/** 
 * \brief Initialize object on device.
 * 
 * \note Initailization must be excecuted only once.
 */
template <typename T, typename... Args>
__global__ void setup_object(T* d_ptr, Args... args) {
    d_ptr = new T(args...);
}

__global__ void setup_conductor(pt::Material** d_ptr, float3 albedo, float fuzz) {
    *d_ptr = new pt::Conductor(albedo, fuzz);
}

__global__ void setup_dielectric(pt::Material** d_ptr, float3 albedo, float ior) {
    *d_ptr = new pt::Dielectric(albedo, ior);
}

__global__ void setup_diffuse(pt::Material** d_ptr, float3 albedo) {
    *d_ptr = new pt::Diffuse(albedo);
}

__global__ void setup_emitter(pt::Material** d_ptr, float3 color, float strength) {
    *d_ptr = new pt::Emitter(color, strength);
}

template <typename T>
__global__ void delete_object(T* d_ptr) {
    delete d_ptr;
}

__global__ void delete_conductor(pt::Material** d_ptr) {
    delete ((pt::Conductor*)(*d_ptr));
}

__global__ void delete_dielectric(pt::Material** d_ptr) {
    delete ((pt::Dielectric*)(*d_ptr));
}

__global__ void delete_diffuse(pt::Material** d_ptr) {
    delete ((pt::Diffuse*)(*d_ptr));
}

__global__ void delete_emitter(pt::Material** d_ptr) {
    delete ((pt::Emitter*)(*d_ptr));
}

void pt::Conductor::setup_on_device() {
    CUDA_CHECK(cudaMalloc(&d_ptr, sizeof(pt::Material**)));
    setup_conductor<<<1,1>>>((pt::Material**)d_ptr, m_albedo, m_fuzz);
    CUDA_SYNC_CHECK();
}

void pt::Conductor::delete_on_device() {
    delete_conductor<<<1,1>>>(d_ptr);
}

void pt::Dielectric::setup_on_device() {
    CUDA_CHECK(cudaMalloc(&d_ptr, sizeof(pt::Material**)));
    setup_dielectric<<<1,1>>>((pt::Material**)d_ptr, m_albedo, m_ior);
    CUDA_SYNC_CHECK();
}

void pt::Dielectric::delete_on_device() {
    delete_dielectric<<<1,1>>>(d_ptr);
}

void pt::Diffuse::setup_on_device() {
    CUDA_CHECK(cudaMalloc(&d_ptr, sizeof(pt::Material**)));
    setup_diffuse<<<1,1>>>((pt::Material**)d_ptr, m_albedo);
    CUDA_SYNC_CHECK();
}

void pt::Diffuse::delete_on_device() {
    delete_diffuse<<<1,1>>>(d_ptr);
}

void pt::Emitter::setup_on_device() {
    CUDA_CHECK(cudaMalloc(&d_ptr, sizeof(pt::Material**)));
    setup_emitter<<<1,1>>>((pt::Material**)d_ptr, m_color, m_strength);
    CUDA_SYNC_CHECK();
}

void pt::Emitter::delete_on_device() {
    delete_emitter<<<1,1>>>(d_ptr);
}
