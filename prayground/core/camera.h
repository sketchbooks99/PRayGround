#pragma once

#include <prayground/math/vec_math.h>

#ifndef __CUDACC__
    #include <prayground/app/window.h>
#endif

namespace prayground {

/**
 * @brief 
 * Standard pinhole camera
 */
class Camera {
public:
    enum class FovAxis
    {
        Horizontal,
        Vertical,
        Diagonal
    };

    struct Data {
        Vec3f origin;
        Vec3f lookat;
        Vec3f up;
        Vec3f U;
        Vec3f V;
        Vec3f W;
        float fov;
        float aspect;
        float nearclip;
        float farclip;
        FovAxis fovaxis;
    };

#ifndef __CUDACC__
    Camera();

    Camera(const Vec3f& origin, const Vec3f& lookat, const Vec3f& up, float fov, float aspect,
        float nearclip = 0.01f, float farclip = 10000.0f, FovAxis fovaxis = FovAxis::Horizontal);

    Vec3f direction() const;

    const Vec3f& origin() const;
    void setOrigin(const Vec3f& origin);
    void setOrigin(float x, float y, float z);

    const Vec3f& lookat() const;
    void setLookat(const Vec3f& lookat);
    void setLookat(float x, float y, float z);

    const Vec3f& up() const;
    void setUp(const Vec3f& up);
    void setUp(float x, float y, float z);

    const float& fov() const;
    void setFov(float fov);

    const float& aspect() const;
    void setAspect( float aspect );

    const float& nearClip() const;
    void setNearClip( float nearclip );

    const float& farClip() const;
    void setFarClip( float farclip );

    const FovAxis& fovAxis() const;
    void setFovAxis( FovAxis fovaxis );

    void enableTracking(std::shared_ptr<Window> window);
    void disableTracking();

    void UVWFrame(Vec3f& U, Vec3f& V, Vec3f& W) const;

    Data getData() const;
protected:
    Vec3f m_origin;
    Vec3f m_lookat;
    Vec3f m_up;
    float m_fov;
    float m_aspect;
    float m_nearclip;
    float m_farclip;
    FovAxis m_fovaxis;
private:
    void mouseDragged(float x, float y, int button);
    void mouseScrolled(float xoffset, float yoffset);
#endif 

};

/**
 * @brief 
 * Lens camera to realize a DoF (Depth of Field) effect.
 * This class implemented based on \c Camera class and extend basic features of it.
 */
class LensCamera final : public Camera {
public:
    struct Data {
        Vec3f origin;
        Vec3f lookat;
        Vec3f up;
        Vec3f U;
        Vec3f V;
        Vec3f W;
        float fov;
        float aspect;
        float nearclip;
        float farclip;
        float aperture;
        float focus_distance;
        FovAxis fovaxis;
    };

#ifndef __CUDACC__
    LensCamera() : Camera(), m_aperture(0.01f), m_focus_distance(100.0) {}
    LensCamera(
        const Vec3f& origin, const Vec3f& lookat, const Vec3f& up, float fov, float aspect, 
        float nearclip = 0.01f, float farclip = 10000.0f, 
        float aperture = 0.01f, float focus_dist = 100.0f,
        FovAxis fovaxis=FovAxis::Horizontal)
    : Camera(origin, lookat, up, fov, aspect, nearclip, farclip, fovaxis), m_aperture(aperture), m_focus_distance(focus_dist)
    {}

    const float& aperture() const { return m_aperture; }
    void setAperture( const float& aperture ) { m_aperture = aperture; }

    const float& focusDistance() const { return m_focus_distance; }
    void setFocusDistance( const float& focus_dist ) { m_focus_distance = focus_dist; }

    Data getData() const;
private:
    float m_aperture;
    float m_focus_distance;

#endif // __CUDACC__
};

}