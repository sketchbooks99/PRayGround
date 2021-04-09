#pragma once

#include <include/core/primitive.h>
#include <include/core/scene.h>

#include "pathtracer/shape/sphere.h"
#include "pathtracer/shape/trianglemesh.h"

#include "pathtracer/material/conductor.h"
#include "pathtracer/material/dielectric.h"
#include "pathtracer/material/diffuse.h"
#include "pathtracer/material/emitter.h"

pt::Scene my_scene() {
    pt::Scene scene;
    // utility settings    
    scene.set_bgcolor(make_float4(0.0f));
    scene.set_width(768);
    scene.set_height(768);
    scene.set_depth(5);
    scene.set_samples_per_launch(4);

    // Material pointers. They are constructed on the device at the same time.
    pt::Material* red_diffuse = new pt::Diffuse(make_float3(0.8f, 0.05f, 0.05f));
    pt::Material* green_diffuse = new pt::Diffuse(make_float3(0.05f, 0.8f, 0.05f));
    pt::Material* white_diffuse = new pt::Diffuse(make_float3(0.8f, 0.8f, 0.8f));
    pt::Material* emitter = new pt::Emitter(make_float3(0.8f, 0.8f, 0.7f), 15.0f);
    pt::Material* metal = new pt::Conductor(make_float3(0.8f, 0.8f, 0.2f), 0.01f);
    pt::Material* glass = new pt::Dielectric(make_float3(0.9f), 1.5f);

    float3 cornel_center = make_float3(278.0f, 274.4f, 279.6f);

    // Primitive instance for cornel box.
    pt::PrimitiveInstance cornel_ps = pt::PrimitiveInstance(pt::Transform());
    cornel_ps.set_sbt_index_base(0);

    // Floor ------------------------------------
    std::vector<float3> floor_vertices;
    std::vector<float3> floor_normals(6, make_float3(0.0f, 1.0f, 0.0f));
    std::vector<int3> floor_indices;
    floor_vertices.emplace_back(make_float3(  0.0f, 0.0f,   0.0f));
    floor_vertices.emplace_back(make_float3(  0.0f, 0.0f, 559.2f));
    floor_vertices.emplace_back(make_float3(556.0f, 0.0f, 559.2f));
    floor_vertices.emplace_back(make_float3(  0.0f, 0.0f,   0.0f));
    floor_vertices.emplace_back(make_float3(556.0f, 0.0f, 559.2f));
    floor_vertices.emplace_back(make_float3(556.0f, 0.0f,   0.0f));
    floor_indices.emplace_back(make_int3(0, 1, 2));
    floor_indices.emplace_back(make_int3(3, 4, 5));
    auto floor_mesh = new pt::TriangleMesh(floor_vertices, floor_indices, floor_normals);
    cornel_ps.add_primitive(floor_mesh, white_diffuse);

    // Ceiling ------------------------------------
    std::vector<float3> ceiling_vertices;
    std::vector<float3> ceiling_normals(6, make_float3(0.0f, -1.0f, 0.0f));
    std::vector<int3> ceiling_indices;
    ceiling_vertices.emplace_back(make_float3(  0.0f, 548.8f,   0.0f));
    ceiling_vertices.emplace_back(make_float3(556.0f, 548.8f,   0.0f));
    ceiling_vertices.emplace_back(make_float3(556.0f, 548.8f, 559.2f));
    ceiling_vertices.emplace_back(make_float3(  0.0f, 548.8f,   0.0f));
    ceiling_vertices.emplace_back(make_float3(556.0f, 548.8f, 559.2f));
    ceiling_vertices.emplace_back(make_float3(  0.0f, 548.8f, 559.2f));
    ceiling_indices.emplace_back(make_int3(0, 1, 2));
    ceiling_indices.emplace_back(make_int3(3, 4, 5));
    auto ceiling_mesh = new pt::TriangleMesh(ceiling_vertices, ceiling_indices, ceiling_normals);
    cornel_ps.add_primitive(ceiling_mesh, white_diffuse);

    // Back wall ------------------------------------
    std::vector<float3> back_wall_vertices;
    std::vector<float3> back_wall_normals(6, make_float3(0.0f, 0.0f, -1.0f));
    std::vector<int3> back_wall_indices;
    back_wall_vertices.emplace_back(make_float3(  0.0f,   0.0f, 559.2f));
    back_wall_vertices.emplace_back(make_float3(  0.0f, 548.8f, 559.2f));
    back_wall_vertices.emplace_back(make_float3(556.0f, 548.8f, 559.2f));
    back_wall_vertices.emplace_back(make_float3(  0.0f,   0.0f, 559.2f));
    back_wall_vertices.emplace_back(make_float3(556.0f, 548.8f, 559.2f));
    back_wall_vertices.emplace_back(make_float3(556.0f,   0.0f, 559.2f));
    back_wall_indices.emplace_back(make_int3(0, 1, 2));
    back_wall_indices.emplace_back(make_int3(3, 4, 5));
    auto back_wall_mesh = new pt::TriangleMesh(back_wall_vertices, back_wall_indices, back_wall_normals);
    cornel_ps.add_primitive(back_wall_mesh, white_diffuse);

    // Right wall ------------------------------------
    std::vector<float3> right_wall_vertices;
    std::vector<float3> right_wall_normals(6, make_float3(1.0f, 0.0f, 0.0f));
    std::vector<int3> right_wall_indices;
    right_wall_vertices.emplace_back(make_float3(0.0f,   0.0f,   0.0f));
    right_wall_vertices.emplace_back(make_float3(0.0f, 548.8f,   0.0f));
    right_wall_vertices.emplace_back(make_float3(0.0f, 548.8f, 559.2f));
    right_wall_vertices.emplace_back(make_float3(0.0f,   0.0f,   0.0f));
    right_wall_vertices.emplace_back(make_float3(0.0f, 548.8f, 559.2f));
    right_wall_vertices.emplace_back(make_float3(0.0f,   0.0f, 559.2f));
    right_wall_indices.emplace_back(make_int3(0, 1, 2));
    right_wall_indices.emplace_back(make_int3(3, 4, 5));
    auto right_wall_mesh = new pt::TriangleMesh(right_wall_vertices, right_wall_indices, right_wall_normals);
    cornel_ps.add_primitive(right_wall_mesh, red_diffuse);

    // Left wall ------------------------------------
    std::vector<float3> left_wall_vertices;
    std::vector<float3> left_wall_normals(6, make_float3(-1.0f, 0.0f, 0.0f));
    std::vector<int3> left_wall_indices;
    left_wall_vertices.emplace_back(make_float3(556.0f,   0.0f,   0.0f));
    left_wall_vertices.emplace_back(make_float3(556.0f,   0.0f, 559.2f));
    left_wall_vertices.emplace_back(make_float3(556.0f, 548.8f, 559.2f));
    left_wall_vertices.emplace_back(make_float3(556.0f,   0.0f,   0.0f));
    left_wall_vertices.emplace_back(make_float3(556.0f, 548.8f, 559.2f));
    left_wall_vertices.emplace_back(make_float3(556.0f, 548.8f,   0.0f));
    left_wall_indices.emplace_back(make_int3(0, 1, 2));
    left_wall_indices.emplace_back(make_int3(3, 4, 5));
    auto left_wall_mesh = new pt::TriangleMesh(left_wall_vertices, left_wall_indices, left_wall_normals);
    cornel_ps.add_primitive(left_wall_mesh, green_diffuse);

    // Ceiling light ------------------------------------
    std::vector<float3> ceiling_light_vertices;
    std::vector<float3> ceiling_light_normals(6, make_float3(0.0f, -1.0f, 0.0f));
    std::vector<int3> ceiling_light_indices;
    ceiling_light_vertices.emplace_back(make_float3(343.0f, 548.6f, 227.0f));
    ceiling_light_vertices.emplace_back(make_float3(213.0f, 548.6f, 227.0f));
    ceiling_light_vertices.emplace_back(make_float3(213.0f, 548.6f, 332.0f));
    ceiling_light_vertices.emplace_back(make_float3(343.0f, 548.6f, 227.0f));
    ceiling_light_vertices.emplace_back(make_float3(213.0f, 548.6f, 332.0f));
    ceiling_light_vertices.emplace_back(make_float3(343.0f, 548.6f, 332.0f));
    ceiling_light_indices.emplace_back(make_int3(0, 1, 2));
    ceiling_light_indices.emplace_back(make_int3(3, 4, 5));
    auto ceiling_light_mesh = new pt::TriangleMesh(ceiling_light_vertices, ceiling_light_indices, ceiling_light_normals);
    cornel_ps.add_primitive(ceiling_light_mesh, emitter);

    // pt::Shape* sphere = new pt::Sphere(cornel_center, 100.0f);
    // cornel_ps.add_primitive(sphere, white_diffuse);

    scene.add_primitive_instance(cornel_ps);

    // pt::PrimitiveInstance sphere_ps(pt::Transform(sutil::Matrix4x4::translate(cornel_center)));
    pt::PrimitiveInstance sphere_ps = pt::PrimitiveInstance(pt::Transform());
    sphere_ps.set_sbt_index_base(cornel_ps.sbt_index());
    // pt::Shape* sphere = new pt::Sphere(cornel_center + make_float3(0.0f, -90.0f, 0.0f), 80.0f);
    // sphere_ps.add_primitive(sphere, white_diffuse);

    pt::Shape* metal_sphere = new pt::Sphere(cornel_center + make_float3(-120.0f, 0.0f, 0.0f), 80.0f);
    sphere_ps.add_primitive(metal_sphere, metal);

    pt::Shape* glass_sphere = new pt::Sphere(cornel_center + make_float3(120.0f, 0.0f, 0.0f), 80.0f);
    sphere_ps.add_primitive(glass_sphere, glass);

    scene.add_primitive_instance(sphere_ps);

    /**
     * \note Multiple instances of sphere induced invalid argument error at cudaFree
     */
    // pt::PrimitiveInstance sphere2_ps(pt::Transform(sutil::Matrix4x4::translate(cornel_center + make_float3(100.0f, 0.0f, 0.0f))));
    // sphere2_ps.set_sbt_index_base(sphere_ps.sbt_index());
    // pt::Shape* sphere2 = new pt::Sphere(make_float3(0.0f), 80.0f);
    // sphere2_ps.add_primitive(sphere2, red_diffuse);
    // scene.add_primitive_instance(sphere2_ps);

    return scene;
}