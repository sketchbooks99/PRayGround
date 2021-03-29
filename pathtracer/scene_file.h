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
    scene.set_bgcolor(make_float4(0.f));
    scene.set_width(768);
    scene.set_height(768);
    scene.set_depth(5);
    scene.set_samples_per_launch(4);

    // Primitive instance that manage primitives with a transformation.
    pt::PrimitiveInstance primitive_instance = pt::PrimitiveInstance(pt::Transform());

    // Material pointers. They are constructed on the device at the same time.
    pt::MaterialPtr red_diffuse = new pt::Diffuse(make_float3(0.8f, 0.05f, 0.05f));
    pt::MaterialPtr green_diffuse = new pt::Diffuse(make_float3(0.05f, 0.8f, 0.05f));
    pt::MaterialPtr white_diffuse = new pt::Diffuse(make_float3(0.8f, 0.8f, 0.8f));
    pt::MaterialPtr emitter = new pt::Emitter(make_float3(1.0f), 10.0f);

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
    pt::ShapePtr floor_mesh = new pt::TriangleMesh(floor_vertices, floor_indices, floor_normals);
    primitive_instance.add_primitive(pt::Primitive(floor_mesh, white_diffuse, 0));

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
    pt::ShapePtr ceiling_mesh = new pt::TriangleMesh(ceiling_vertices, ceiling_indices, ceiling_normals);
    primitive_instance.add_primitive(pt::Primitive(ceiling_mesh, white_diffuse, 1));

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
    pt::ShapePtr back_wall_mesh = new pt::TriangleMesh(back_wall_vertices, back_wall_indices, back_wall_normals);
    primitive_instance.add_primitive(pt::Primitive(back_wall_mesh, white_diffuse, 2));

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
    pt::ShapePtr right_wall_mesh = new pt::TriangleMesh(right_wall_vertices, right_wall_indices, right_wall_normals);
    primitive_instance.add_primitive(pt::Primitive(right_wall_mesh, red_diffuse, 3));

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
    pt::ShapePtr left_wall_mesh = new pt::TriangleMesh(left_wall_vertices, left_wall_indices, left_wall_normals);
    primitive_instance.add_primitive(pt::Primitive(left_wall_mesh, green_diffuse, 4));

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
    pt::ShapePtr ceiling_light_mesh = new pt::TriangleMesh(ceiling_light_vertices, ceiling_light_indices, ceiling_light_normals);
    primitive_instance.add_primitive(pt::Primitive(ceiling_light_mesh, emitter, 5));

    // pt::ShapePtr sphere = new pt::Sphere(make_float3(0.0f, 0.0f, 0.0f), 100.0f);
    // primitive_instance.add_primitive(pt::Primitive(sphere, white_diffuse, 0));

    scene.add_primitive_instance(primitive_instance);

    return scene;
}