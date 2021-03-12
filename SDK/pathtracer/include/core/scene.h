#pragma once

#include <core/primitive.h>
#include <sutil/Camera.h>

namespace pt {

class Scene {
public:
    explicit Scene() {}

    void add_primitive(const Primitive& p) { m_primitives.push_back(p); }
    std::vector<Primitive> get_primitives() const { return m_primitives; }

    void set_camera(const sutil::Camera& cam) { m_camera = cam; }
    sutil::Camera get_camera() const { return m_camera; }

    void set_width(const unsigned int w) { m_width = w; }
    unsigned int get_width() const { return m_width; }

    void set_height(const unsigned int h) { m_height = h; }
    unsigned int get_height() const { return m_height; }

    void set_bgcolor(const float4& bg) { m_bgcolor = bg; }
    float4 get_bgcolor() const { return m_bgcolor; }
private:
    std::vector<Primitive> m_primitives;
    sutil::Camera m_camera;
    unsigned int m_width, m_height;
    float4 m_bgcolor;
};



}