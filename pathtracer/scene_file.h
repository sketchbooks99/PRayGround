#pragma once

#include "core/primitive.h"
#include "core/scene.h"

#include "shape/sphere.h"
#include "shape/trianglemesh.h"

#include "material/conductor.h"
#include "material/dielectric.h"
#include "material/diffuse.h"
#include "material/emitter.h"

#include "texture/constant.h"
#include "texture/checker.h"
#include "texture/image.h"

/**
 * \note 
 * If a PrimitiveInstance store meshes and custom primitives (i.e. Sphere, Cylinder...), 
 * please sort primitives array to render geometries correctly as like follow code.
 * 
 * auto ps = pt::PrimitiveInstance(pt::Transform());
 * ps.set_sbt_index_base(0);
 * ps.add_primitive(sphere, white_lambert);
 * ps.add_primitive(mesh, metal);
 * ps.sort();
 * scene.add_primitive_instance(ps);
 */

pt::Scene my_scene() {
    pt::Scene scene;
    // utility settings    
    scene.set_bgcolor(make_float4(0.0f));
    scene.set_width(768);
    scene.set_height(768);
    scene.set_depth(5);
    scene.set_samples_per_launch(4);
    scene.set_num_samples(4096);

    auto checker1 = new pt::CheckerTexture(
        make_float3(0.3f), make_float3(0.9f), 10.0f
    );
    auto checker2 = new pt::CheckerTexture(
        make_float3(0.8f), make_float3(0.8f, 0.05, 0.05f), 10.0f
    );
    auto earth_image = new pt::ImageTexture("../../data/image/earth.jpg");

    // Material pointers. 
    auto red_diffuse = new pt::Diffuse(make_float3(0.8f, 0.05f, 0.05f));
    auto green_diffuse = new pt::Diffuse(make_float3(0.05f, 0.8f, 0.05f));
    auto white_diffuse = new pt::Diffuse(make_float3(0.8f, 0.8f, 0.8f));
    auto emitter = new pt::Emitter(make_float3(0.8f, 0.8f, 0.7f), 25.0f);
    auto metal = new pt::Conductor(make_float3(0.8f, 0.8f, 0.2f), 0.01f);
    auto glass = new pt::Dielectric(make_float3(0.9f), 1.5f);
    auto floor_checker = new pt::Diffuse(checker1);
    auto earth_diffuse = new pt::Diffuse(earth_image);
    auto teapot_diffuse = new pt::Diffuse(make_float3(1.0f, 0.8f, 0.3f));

    float3 cornel_center = make_float3(278.0f, 274.4f, 279.6f);

    // Primitive instance for cornel box.
    pt::PrimitiveInstance cornel_ps = pt::PrimitiveInstance(pt::Transform());
    cornel_ps.set_sbt_index_base(0);

    // Floor 
    auto floor_mesh = pt::createQuadMesh(0.0f, 556.0f, 0.0f, 559.2f, 0.0f, pt::Axis::Y);
    cornel_ps.add_primitive(floor_mesh, floor_checker);
    // Ceiling 
    auto ceiling_mesh = pt::createQuadMesh(0.0f, 556.0f, 0.0f, 559.2f, 548.8f, pt::Axis::Y);
    cornel_ps.add_primitive(ceiling_mesh, white_diffuse);
    // Back wall 
    auto back_wall_mesh = createQuadMesh(0.0f, 556.0f, 0.0f, 548.8f, 559.2f, pt::Axis::Z);
    cornel_ps.add_primitive(back_wall_mesh, white_diffuse);
    // Right wall 
    auto right_wall_mesh = createQuadMesh(0.0f, 548.8f, 0.0f, 559.2f, 0.0f, pt::Axis::X);
    cornel_ps.add_primitive(right_wall_mesh, red_diffuse);
    // Left wall 
    auto left_wall_mesh = createQuadMesh(0.0f, 548.8f, 0.0f, 559.2f, 556.0f, pt::Axis::X);
    cornel_ps.add_primitive(left_wall_mesh, green_diffuse);
    // Ceiling light
    auto ceiling_light_mesh = createQuadMesh(213.0f, 343.0f, 227.0f, 332.0f, 548.6f, pt::Axis::Y);
    cornel_ps.add_primitive(ceiling_light_mesh, emitter);
    scene.add_primitive_instance(cornel_ps);

    // Armadillo
    auto bunny1_matrix = sutil::Matrix4x4::translate(cornel_center + make_float3(150.0f, -210.0f, -130.0f)) 
                       * sutil::Matrix4x4::scale(make_float3(1.2f));
    auto bunny1_ps = pt::PrimitiveInstance(bunny1_matrix);
    bunny1_ps.set_sbt_index_base(cornel_ps.sbt_index());
    auto bunny1 = new pt::TriangleMesh("../../model/Armadillo.ply");
    bunny1_ps.add_primitive(bunny1, metal);
    scene.add_primitive_instance(bunny1_ps);

    // Center bunny with lambert material
    auto bunny2_matrix = sutil::Matrix4x4::translate(cornel_center + make_float3(0.0f, -270.0f, 100.0f)) 
                       * sutil::Matrix4x4::rotate(M_PIf, make_float3(0.0f, 1.0f, 0.0f))
                       * sutil::Matrix4x4::scale(make_float3(1200.0f));
    auto bunny2_ps = pt::PrimitiveInstance(bunny2_matrix);
    bunny2_ps.set_sbt_index_base(bunny1_ps.sbt_index());
    auto bunny2 = new pt::TriangleMesh("../../model/bunny.obj");
    bunny2_ps.add_primitive(bunny2, white_diffuse);
    scene.add_primitive_instance(bunny2_ps);

    // Teapot
    auto teapot_matrix = sutil::Matrix4x4::translate(cornel_center + make_float3(-150.0f, -260.0f, -120.0f)) 
                    //    * sutil::Matrix4x4::rotate(M_PIf, make_float3(0.0f, 1.0f, 0.0f))
                       * sutil::Matrix4x4::scale(make_float3(40.0f));
    auto teapot_ps = pt::PrimitiveInstance(teapot_matrix);
    teapot_ps.set_sbt_index_base(bunny2_ps.sbt_index());
    auto teapot = new pt::TriangleMesh("../../model/teapot_normal_merged.obj");
    teapot_ps.add_primitive(teapot, teapot_diffuse);
    scene.add_primitive_instance(teapot_ps);

    auto sphere_ps = pt::PrimitiveInstance(pt::Transform());
    sphere_ps.set_sbt_index_base(teapot_ps.sbt_index());

    auto sphere = new pt::Sphere(cornel_center + make_float3(120.0f, 80.0f, 100.0), 90.0f);
    sphere_ps.add_primitive(sphere, earth_diffuse);

    auto glass_sphere = new pt::Sphere(cornel_center + make_float3(-150.0f, 0.0f, 80.0f), 80.0f);
    sphere_ps.add_primitive(glass_sphere, glass);
    scene.add_primitive_instance(sphere_ps);

    return scene;
}