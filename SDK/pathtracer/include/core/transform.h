#pragma once 

#include <sutil/Matrix.h>
#include <sutil/vec_math.h>
#include <core/util.h>
#include <core/ray.h>
#include <optix/macros.h>

namespace pt {

struct Transform {
    sutil::Matrix4x4 mat, matInv;

    HOSTDEVICE Transform() {}
    explicit HOSTDEVICE Transform(sutil::Matrix4x4 m) : mat(m), matInv(m.inverse()) {}
    explicit HOSTDEVICE Transform(sutil::Matrix4x4 m, sutil::Matrix4x4 mInv) : mat(m), matInv(mInv) {}

    HOSTDEVICE Transform operator*(const Transform& t) {
        sutil::Matrix4x4 m = mat * t.mat;
        sutil::Matrix4x4 mInv = t.matInv * matInv;
        return Transform(m, mInv);
    }

    HOSTDEVICE Transform operator*=(const Transform& t) {
        mat = mat * t.mat;
        matInv = t.matInv * matInv;
        return *this;
    }

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

#ifndef __CUDACC__
    HOST void rotate_x(const float radians) { this->rotate(radians, make_float3(1, 0, 0)); }
    HOST void rotate_y(const float radians) { this->rotate(radians, make_float3(0, 1, 0)); }
    HOST void rotate_z(const float radians) { this->rotate(radians, make_float3(0, 0, 1)); }
    HOST void rotate(const float radians, const float3& axis) {
        sutil::Matrix4x4 rotateMat = sutil::Matrix4x4::rotate(radians, axis);
        sutil::Matrix4x4 rotateMatInv = rotateMat.inverse();
        *this *= Transform(rotateMat, rotateMatInv);
    }
    
    HOST void translate(const float3& v) {
        sutil::Matrix4x4 translateMat = sutil::Matrix4x4::translate(v);
        sutil::Matrix4x4 translateMatInv = translateMat.inverse();
        *this *= Transform(translateMat, translateMatInv);
    }

    HOST void scale(const float s) { this->scale(make_float3(s)); }
    HOST void scale(const float3& s) {
        sutil::Matrix4x4 scaleMat = sutil::Matrix4x4::scale(s);
        sutil::Matrix4x4 scaleMatInv = scaleMat.inverse();
        *this *= Transform(scaleMat, scaleMatInv);
    }
#else
    DEVICE float3 point_mul(const float3& p) {
        float x = mat[0*4+0]*p.x + mat[0*4+1]*p.y + mat[0*4+2]*p.z + mat[0*4+3];
        float y = mat[1*4+0]*p.x + mat[1*4+1]*p.y + mat[1*4+2]*p.z + mat[1*4+3];
        float z = mat[2*4+0]*p.x + mat[2*4+1]*p.y + mat[2*4+2]*p.z + mat[2*4+3];
        float w = mat[3*4+0]*p.x + mat[3*4+1]*p.y + mat[3*4+2]*p.z + mat[3*4+3];
        if (w == 1)
            return make_float3(x, y, z);
        else
            return make_float3(x, y, z) / w;
    }

    DEVICE float3 vector_mul(const float3& v) {
        float x = v.x, y = v.y, z = v.z;
        return make_float3(mat[0*4+0]*x + mat[0*4+1]*y + mat[0*4+2]*z,
                           mat[1*4+0]*x + mat[1*4+1]*y + mat[1*4+2]*z,
                           mat[2*4+0]*x + mat[2*4+1]*y + mat[2*4+2]*z);
    }

    // Multiply itself and normal
    DEVICE float3 normal_mul(const float3& n) {
        float x = n.x, y = n.y, z = n.z;
        return make_float3(matInv[0*4+0]*x + matInv[1*4*0]*y + matInv[2*4+0]*z,
                           matInv[0*4+1]*x + matInv[1*4*1]*y + matInv[2*4+1]*z,
                           matInv[0*4+2]*x + matInv[1*4*2]*y + matInv[2*4+2]*z);
    }

    DEVICE float3 transform_ray(const Ray& r) {
        float ro = point_mul(r.o)
    }
#endif
};

}
