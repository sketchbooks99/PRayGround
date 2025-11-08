#pragma once

#include <prayground/math/vec.h>
#include <prayground/optix/macros.h>
#include <cmath>

namespace prayground {

    // Quaternion class for 3D rotations
    template <typename T>
    class Quaternion {
    public:
        // w + xi + yj + zk
        T w, x, y, z;

        // Constructors
        HOSTDEVICE Quaternion() : w(1), x(0), y(0), z(0) {}
        HOSTDEVICE Quaternion(T w, T x, T y, T z) : w(w), x(x), y(y), z(z) {}
        
        // Create quaternion from axis-angle representation
        // axis: rotation axis (will be normalized)
        // angle: rotation angle in radians
        HOSTDEVICE Quaternion(const Vec3<T>& axis, T angle) {
            T half_angle = angle * T(0.5);
            T s = sin(half_angle);
            Vec3<T> normalized_axis = normalize(axis);
            w = cos(half_angle);
            x = normalized_axis.x() * s;
            y = normalized_axis.y() * s;
            z = normalized_axis.z() * s;
        }

        // Identity quaternion
        HOSTDEVICE static Quaternion identity() {
            return Quaternion(1, 0, 0, 0);
        }

        // Quaternion multiplication
        HOSTDEVICE Quaternion operator*(const Quaternion& q) const {
            return Quaternion(
                w * q.w - x * q.x - y * q.y - z * q.z,
                w * q.x + x * q.w + y * q.z - z * q.y,
                w * q.y - x * q.z + y * q.w + z * q.x,
                w * q.z + x * q.y - y * q.x + z * q.w
            );
        }

        // Conjugate
        HOSTDEVICE Quaternion conjugate() const {
            return Quaternion(w, -x, -y, -z);
        }

        // Norm
        HOSTDEVICE T norm() const {
            return sqrtf(w * w + x * x + y * y + z * z);
        }

        // Normalize
        HOSTDEVICE Quaternion normalized() const {
            T n = norm();
            if (n < T(1e-6)) return Quaternion::identity();
            return Quaternion(w / n, x / n, y / n, z / n);
        }

        // Rotate a vector by this quaternion
        HOSTDEVICE Vec3<T> rotate(const Vec3<T>& v) const {
            // v' = q * v * q^-1
            // For unit quaternions: q^-1 = q*
            Quaternion q_v(0, v.x(), v.y(), v.z());
            Quaternion result = (*this) * q_v * this->conjugate();
            return Vec3<T>(result.x, result.y, result.z);
        }

        HOSTDEVICE Quaternion inverse() const {
            T n2 = w * w + x * x + y * y + z * z;
            if (n2 < T(1e-6)) return Quaternion::identity();
            return Quaternion(w / n2, -x / n2, -y / n2, -z / n2);
        }

        // Convert to rotation matrix (for debugging/visualization)
        // Returns the 3x3 rotation matrix as 9 elements
        HOSTDEVICE void toRotationMatrix(T mat[9]) const {
            T xx = x * x, yy = y * y, zz = z * z;
            T xy = x * y, xz = x * z, yz = y * z;
            T wx = w * x, wy = w * y, wz = w * z;

            mat[0] = 1 - 2 * (yy + zz);
            mat[1] = 2 * (xy - wz);
            mat[2] = 2 * (xz + wy);
            
            mat[3] = 2 * (xy + wz);
            mat[4] = 1 - 2 * (xx + zz);
            mat[5] = 2 * (yz - wx);
            
            mat[6] = 2 * (xz - wy);
            mat[7] = 2 * (yz + wx);
            mat[8] = 1 - 2 * (xx + yy);
        }

        static HOSTDEVICE Quaternion trackTo(const Vec3<T>& v, const Vec3<T>& target, const Vec3<T>& up) {
            Vec3<T> from = normalize(v);
            Vec3<T> to = normalize(target);

            Vec3<T> axis = cross(from, to);
            T axis_len = length(axis);

            if (axis_len < T(1e-6)) {
                // Vectors are parallel
                if (dot(from, to) > 0) {
                    // Same direction
                    return Quaternion::identity();
                } else {
                    // Opposite direction
                    Vec3<T> ortho = cross(from, up);
                    if (length(ortho) < T(1e-6)) {
                        ortho = cross(from, Vec3<T>(1, 0, 0));
                        if (length(ortho) < T(1e-6)) {
                            ortho = cross(from, Vec3<T>(0, 1, 0));
                        }
                    }
                    ortho = normalize(ortho);
                    return Quaternion(0.0f, ortho.x(), ortho.y(), ortho.z()); // 180 degree rotation
                }
            }
            
            axis = normalize(axis);
            float angle = acosf(clamp(dot(from, to), -1.0f, 1.0f));
            float half_angle = angle * T(0.5);
            float sin_half = sin(half_angle);

            return Quaternion(cos(half_angle), axis.x() * sin_half, axis.y() * sin_half, axis.z() * sin_half).normalized();
        }
    };

    using Quatf = Quaternion<float>;
    using Quatd = Quaternion<double>;

    // Helper function: rotate vector by axis-angle
    template <typename T>
    HOSTDEVICE Vec3<T> rotateVector(const Vec3<T>& v, const Vec3<T>& axis, T angle) {
        Quaternion<T> q(axis, angle);
        return q.rotate(v);
    }

} // namespace prayground
