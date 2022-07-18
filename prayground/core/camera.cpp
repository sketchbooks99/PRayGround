#include "camera.h"
#include <prayground/app/app_runner.h>
#include <prayground/app/input.h>
#include <prayground/math/util.h>

namespace prayground {

    Camera::Camera()
        : m_origin(Vec3f(0.0f, 0.0f, -1.0f)),
        m_lookat(Vec3f(0.0f)),
        m_up(Vec3f(0.0f, 1.0f, 0.0f)),
        m_fov(40.0f),
        m_aspect(1.0f),
        m_nearclip(0.01f),
        m_farclip(10000.0f),
        m_fovaxis(FovAxis::Vertical)
    {

    }

    Camera::Camera(const Vec3f& origin, const Vec3f& lookat, const Vec3f& up, float fov, float aspect,
        float nearclip, float farclip, FovAxis fovaxis)
        : m_origin(origin), m_lookat(lookat), m_up(up), m_fov(fov), m_aspect(aspect)
        , m_nearclip(nearclip), m_farclip(farclip), m_fovaxis(fovaxis)
    {

    }

    // --------------------------------------------------------------------------------------
    Vec3f Camera::direction() const 
    { 
        return normalize(m_lookat - m_origin); 
    }

    const Vec3f& Camera::origin() const
    {
        return m_origin;
    }
    void Camera::setOrigin(const Vec3f& origin)
    {
        m_origin = origin;
    }
    void Camera::setOrigin(float x, float y, float z)
    {
        m_origin = Vec3f(x, y, z);
    }

    const Vec3f& Camera::lookat() const
    {
        return m_lookat;
    }
    void Camera::setLookat(const Vec3f& lookat)
    {
        m_lookat = lookat;
    }
    void Camera::setLookat(float x, float y, float z)
    {
        m_lookat = Vec3f(x, y, z);
    }

    const Vec3f& Camera::up() const
    {
        return m_up;
    }
    void Camera::setUp(const Vec3f& up)
    {
        m_up = up;
    }
    void Camera::setUp(float x, float y, float z)
    {
        m_up = Vec3f(x, y, z);
    }

    const float& Camera::fov() const
    {
        return m_fov;
    }
    void Camera::setFov(float fov)
    {
        m_fov = fov;
    }

    const float& Camera::aspect() const
    {
        return m_aspect;
    }
    void Camera::setAspect( float aspect )
    {
        m_aspect = aspect;
    }

    const float& Camera::nearClip() const
    {
        return m_nearclip;
    }
    void Camera::setNearClip( float nearclip )
    {
        m_nearclip = nearclip;
    }

    const float& Camera::farClip() const
    {
        return m_farclip;
    }
    void Camera::setFarClip( float farclip )
    {
        m_farclip = farclip;
    }

    const Camera::FovAxis& Camera::fovAxis() const
    {
        return m_fovaxis;
    }
    void Camera::setFovAxis( FovAxis fovaxis )
    {
        m_fovaxis = fovaxis;
    }

    // --------------------------------------------------------------------------------------
    void Camera::enableTracking(std::shared_ptr<Window> window)
    {
        window->events().mouseDragged.bindFunction([&](float x, float y, int button){ return this->mouseDragged(x, y, button); });
        window->events().mouseScrolled.bindFunction([&](float xoffset, float yoffset){ return this->mouseScrolled(xoffset, yoffset); });
    }

    void Camera::disableTracking()
    {
        UNIMPLEMENTED();
    }

    void Camera::UVWFrame(Vec3f& U, Vec3f& V, Vec3f& W) const
    {
        W = m_lookat - m_origin;
        float wlen = length(W);
        U = normalize(cross(W, m_up));
        V = normalize(cross(W, U));

        float vlen = wlen * tanf(math::radians(m_fov) / 2.0f);
        V *= vlen;
        float ulen = vlen * m_aspect;
        U *= ulen;
    }

    Camera::Data Camera::getData() const
    {
        Vec3f U, V, W;
        this->UVWFrame(U, V, W);
        return
        {
            m_origin, 
            m_lookat, 
            m_up, 
            U, 
            V, 
            W,
            m_fov, 
            m_aspect, 
            m_nearclip, 
            m_farclip, 
            m_fovaxis
        };
    }

    // --------------------------------------------------------------------------------------
    void Camera::mouseDragged(float x, float y, int button)
    {
        bool is_move = button == MouseButton::Middle;
        if (!is_move) return;

        bool is_slide = pgGetKey() != Key::Unknown && +(pgGetKey() & (Key::LeftShift | Key::RightShift));

        if (is_slide) {
            float deltaX = x - pgGetPreviousMousePosition().x();
            float deltaY = y - pgGetPreviousMousePosition().y();
            Vec3f cam_dir = normalize(m_origin - m_lookat);
            Vec3f cam_side = normalize(cross(cam_dir, this->up()));
            Vec3f cam_up = normalize(cross(cam_dir, cam_side));

            Vec3f offset = cam_side * deltaX + cam_up * deltaY;
        
            this->setOrigin(m_origin + offset * 0.1f);
            this->setLookat(m_lookat + offset * 0.1f);
        }
        else 
        {
            float deltaX = x - pgGetPreviousMousePosition().x();
            float deltaY = y - pgGetPreviousMousePosition().y();
            float cam_length = length(m_origin - m_lookat);
            Vec3f cam_dir = normalize(m_origin - m_lookat);

            float theta = acosf(cam_dir.y());
            float phi = atan2(cam_dir.z(), cam_dir.x());

            theta = clamp(theta - math::radians(deltaY * 0.25f), math::eps, math::pi - math::eps);
            phi += math::radians(deltaX * 0.25f);

            float cam_x = cam_length * sinf(theta) * cosf(phi);
            float cam_y = cam_length * cosf(theta);
            float cam_z = cam_length * sinf(theta) * sinf(phi);

            this->setOrigin(this->lookat() + Vec3f(cam_x, cam_y, cam_z));
        }

    }

    void Camera::mouseScrolled(float xoffset, float yoffset)
    {
        float zoom = yoffset < 0 ? 1.1f : 1.0f / 1.1f;
        this->setOrigin(this->lookat() + (this->origin() - this->lookat()) * zoom);

    }

    // --------------------------------------------------------------------------------------
    LensCamera::Data LensCamera::getData() const
    {
        Vec3f U, V, W;
        this->UVWFrame(U, V, W);
        return {
            m_origin,
            m_lookat,
            m_up,
            U,
            V,
            W,
            m_fov,
            m_aspect,
            m_nearclip,
            m_farclip,
            m_aperture, 
            m_focus_distance,
            m_fovaxis
        };
    }

} // namespace prayground