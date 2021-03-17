#pragma once 

#include <sutil/Matrix.h>
#include <sutil/vec_math.h>
#include <core/util.h>
#include <core/ray.h>
#include <optix/macros.h>

namespace pt {

/// \brief Multiply matrix with positional vector.
HOSTDEVICE INLINE float3 point_mul(const sutil::Matrix4x4& m, const float3& p) {
    float x = m[0*4+0]*p.x + m[0*4+1]*p.y + m[0*4+2]*p.z + m[0*4+3];
    float y = m[1*4+0]*p.x + m[1*4+1]*p.y + m[1*4+2]*p.z + m[1*4+3];
    float z = m[2*4+0]*p.x + m[2*4+1]*p.y + m[2*4+2]*p.z + m[2*4+3];
    float w = m[3*4+0]*p.x + m[3*4+1]*p.y + m[3*4+2]*p.z + m[3*4+3];
    if (w == 1)
        return make_float3(x, y, z);
    else
        return make_float3(x, y, z) / w;
}

/// \brief Muitiply matrix with normal.
HOSTDEVICE INLINE float3 normal_mul(const sutil::Matrix4x4& m, const float3& n) {
    float x = n.x, y = n.y, z = n.z;
    return make_float3(m[0*4+0]*x + m[1*4*0]*y + m[2*4+0]*z,
                       m[0*4+1]*x + m[1*4*1]*y + m[2*4+1]*z,
                       m[0*4+2]*x + m[1*4*2]*y + m[2*4+2]*z);
}

/// \brief Multiply matrix with vector.
HOSTDEVICE INLINE float3 vector_mul(const sutil::Matrix4x4& m, const float3& v) {
    float x = v.x, y = v.y, z = v.z;
    return make_float3(m[0*4+0]*x + m[0*4+1]*y + m[0*4+2]*z,
                       m[1*4+0]*x + m[1*4+1]*y + m[1*4+2]*z,
                       m[2*4+0]*x + m[2*4+1]*y + m[2*4+2]*z);
}

class Transform {
public:
    sutil::Matrix4x4 mat, matInv;

    Transform() {}
    explicit Transform(sutil::Matrix4x4 m) : mat(m), matInv(m.inverse()) {}
    explicit Transform(sutil::Matrix4x4 m, sutil::Matrix4x4 mInv) : mat(m), matInv(mInv) {}

#ifdef __CUDACC__
    DEVICE Transform() {}
    DEVICE Transform(float m[12], float inv[12]) {
        for (int i=0; i<12; i++) {
            mat[i] = m[i];
            matInv[i] = inv[i];
        }
        mat[12] = mat[13] = mat[14] = 0.f;
        mat[15] = 1.f;
        matInv[12] = matInv[13] = matInv[14] = 0.f;
        matInv[15] = 1.f;
    }
#endif 

    void rotate_x(const float radians) { this->rotate(radians, make_float3(1, 0, 0)); }
    void rotate_y(const float radians) { this->rotate(radians, make_float3(0, 1, 0)); }
    void rotate_z(const float radians) { this->rotate(radians, make_float3(0, 0, 1)); }
    void rotate(const float radians, const float3& axis);
    
    void translate(const float3& v); 

    void scale(const float s) { this->scale(make_float3(s)); }
    void scale(const float3& v);

#ifdef __CUDAC__
    DEVICE float3 point_mul(const float3& p) {

    }

    DEVICE float3 vector_mul(const float3& v) {

    }

    DEVICE float3 normal_mul(const float3& n) {

    }

    DEVICE float3 transform_ray(const Ray& r) {
        
    }
#endif
};

HOSTDEVICE INLINE Transform operator*(const Transform& t1, const Transform& t2) {
    auto mat = t1.mat * t2.mat;
    auto matInv = t2.matInv * t1.matInv;
    return Transform(mat, matInv);
}

/// \note Ray transformation is performed only in the device.
#ifdef __CUDACC__ 
HOSTDEVICE INLINE Ray operator*(const Transform& t, const Ray& r) {
    float3 ro = point_mul(t.matInv, r.origin());
    float3 rd = vector_mul(t.matInv, r.direction());
    return Ray(ro, rd);
}
#endif

}
