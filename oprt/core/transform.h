#pragma once 

#include <oprt/math/matrix.h>
#include <oprt/core/util.h>
#include <oprt/core/ray.h>
#include <oprt/optix/macros.h>

namespace oprt {

struct Transform {
    Matrix4f mat, matInv;

    explicit HOSTDEVICE Transform() : mat(Matrix4f::identity()), matInv(Matrix4f::identity()) {}
    explicit HOSTDEVICE Transform(Matrix4f m) : mat(m), matInv(m.inverse()) {}
    explicit HOSTDEVICE Transform(const Matrix4f& m, const Matrix4f& mInv) : mat(m), matInv(mInv) {}

    explicit HOSTDEVICE Transform(const float m[12])
    {
        for (int i = 0; i < 12; i++)
            mat[i] = m[i];
        mat[12] = mat[13] = mat[14] = 0.0f;
        mat[15] = 1.0f;
        matInv = mat.inverse();
    }

    explicit HOSTDEVICE Transform(float m[12], float inv[12]) {
        for (int i = 0; i < 12; i++) {
            mat[i] = m[i];
            matInv[i] = inv[i];
        }
        mat[12] = mat[13] = mat[14] = 0.0f;
        mat[15] = 1.0f;
        matInv[12] = matInv[13] = matInv[14] = 0.0f;
        matInv[15] = 1.0f;
    }

    HOSTDEVICE Transform operator*(const Transform& t) {
        Matrix4f m = mat * t.mat;
        Matrix4f mInv = t.matInv * matInv;
        return Transform(m, mInv);
    }

    HOSTDEVICE Transform operator*=(const Transform& t) {
        mat = mat * t.mat;
        matInv = t.matInv * matInv;
        return *this;
    }

    HOSTDEVICE bool operator==(const Transform& t) {
        return (t.mat == mat) && (t.matInv == matInv);
    }

    HOSTDEVICE bool isIdentity() {
        return (mat == Matrix4f::identity()) && (matInv == Matrix4f::identity());
    }

#ifndef __CUDACC__
    HOST void rotateX(const float radians) { rotate(radians, make_float3(1, 0, 0)); }
    HOST void rotateY(const float radians) { rotate(radians, make_float3(0, 1, 0)); }
    HOST void rotateZ(const float radians) { rotate(radians, make_float3(0, 0, 1)); }
    HOST void rotate(const float radians, const float3& axis) {
        Matrix4f rotateMat = Matrix4f::rotate(radians, axis);
        Matrix4f rotateMatInv = rotateMat.inverse();
        *this *= Transform(rotateMat, rotateMatInv);
    }
    
    HOST void translate(const float3& v) {
        Matrix4f translateMat = Matrix4f::translate(v);
        Matrix4f translateMatInv = translateMat.inverse();
        *this *= Transform(translateMat, translateMatInv);
    }

    HOST void scale(const float s) { this->scale(make_float3(s)); }
    HOST void scale(const float3& s) {
        Matrix4f scaleMat = Matrix4f::scale(s);
        Matrix4f scaleMatInv = scaleMat.inverse();
        *this *= Transform(scaleMat, scaleMatInv);
    }
#else
    DEVICE float3 pointMul(const float3& p) {
        float x = mat[0*4+0]*p.x + mat[0*4+1]*p.y + mat[0*4+2]*p.z + mat[0*4+3];
        float y = mat[1*4+0]*p.x + mat[1*4+1]*p.y + mat[1*4+2]*p.z + mat[1*4+3];
        float z = mat[2*4+0]*p.x + mat[2*4+1]*p.y + mat[2*4+2]*p.z + mat[2*4+3];
        float w = mat[3*4+0]*p.x + mat[3*4+1]*p.y + mat[3*4+2]*p.z + mat[3*4+3];
        if (w == 1)
            return make_float3(x, y, z);
        else
            return make_float3(x, y, z) / w;
    }

    DEVICE float3 vectorMul(const float3& v) {
        float x = v.x, y = v.y, z = v.z;
        return make_float3(mat[0*4+0]*x + mat[0*4+1]*y + mat[0*4+2]*z,
                           mat[1*4+0]*x + mat[1*4+1]*y + mat[1*4+2]*z,
                           mat[2*4+0]*x + mat[2*4+1]*y + mat[2*4+2]*z);
    }

    // Multiply itself and normal
    DEVICE float3 normalMul(const float3& n) {
        float x = n.x, y = n.y, z = n.z;
        return make_float3(matInv[0*4+0]*x + matInv[1*4*0]*y + matInv[2*4+0]*z,
                           matInv[0*4+1]*x + matInv[1*4*1]*y + matInv[2*4+1]*z,
                           matInv[0*4+2]*x + matInv[1*4*2]*y + matInv[2*4+2]*z);
    }

    DEVICE Ray transformRay(const Ray& r) {
        float3 ro = pointMul(r.o);
        float3 rd = vectorMul(r.d);
        return { ro, rd, r.tmin, r.tmax, r.t, r.spectrum };
    }
#endif
};

} // ::oprt
