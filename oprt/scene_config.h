#pragma once

#include "oprt.h"

Scene my_scene() {
    Scene scene;

    std::shared_ptr<Bitmap> film = std::make_shared<Bitmap>(Bitmap::Format::RGBA, 1024, 1024);
    scene.setFilm(film);

    // シーンの一般的な設定    
    scene.setEnvironment("image/107_hdrmaps_com_free.exr");
    scene.setDepth(5);
    scene.setSamplesPerLaunch(1);
    scene.setNumSamples(1000);

    // カメラの設定
    sutil::Camera camera;
    camera.setEye(make_float3(0.0f, 0.0f, -1000.0f));
    camera.setLookat(make_float3(0.0f, -225.0f, 0.0f));
    camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
    camera.setFovY(35.0f);
    scene.setCamera(camera);

    // テクスチャの準備
    auto checker1 = std::make_shared<CheckerTexture>(
        make_float3(0.3f), make_float3(0.9f), 10.0f
    );
    auto checker2 = std::make_shared<CheckerTexture>(
        make_float3(0.8f, 0.05f, 0.05f), make_float3(0.8f), 10.0f
    );
    auto earth_image = std::make_shared<BitmapTexture>("image/earth.jpg");
    auto skyblue_constant = std::make_shared<ConstantTexture>(make_float3(83.0f/255.0f, 179.0f/255.0f, 181.0f/255.0f));
    auto white_constant = std::make_shared<ConstantTexture>(make_float3(1.0f));
    auto disney_checker = std::make_shared<CheckerTexture>(
        make_float3(83.0f/255.0f, 179.0f/255.0f, 181.0f/255.0f), 
        make_float3(0.3f), 10.0f);

    // マテリアルの準備 
    auto red_diffuse = std::make_shared<Diffuse>(make_float3(0.8f, 0.05f, 0.05f));
    auto green_diffuse = std::make_shared<Diffuse>(make_float3(0.05f, 0.8f, 0.05f));
    auto white_diffuse = std::make_shared<Diffuse>(make_float3(0.8f, 0.8f, 0.8f));
    // auto emitter = std::make_shared<Emitter>(make_float3(0.8f, 0.8f, 0.7f), 15.0f);
    auto glass = std::make_shared<Dielectric>(make_float3(0.9f), 1.5f);
    auto floor_checker = std::make_shared<Diffuse>(checker1);
    auto plane_checker = std::make_shared<Diffuse>(checker2);
    auto earth_diffuse = std::make_shared<Diffuse>(earth_image);
    auto disney = std::make_shared<Disney>(disney_checker);
    disney->setMetallic(0.8f);
    disney->setRoughness(0.4f);
    auto teapot_diffuse = std::make_shared<Diffuse>(make_float3(0.9f, 0.8f, 0.3f));

    // コーネルボックス用のプリミティブインスタンス

    // // Floor 
    // auto floor_mesh = createQuadMesh(0.0f, 556.0f, 0.0f, 559.2f, 0.0f, Axis::Y);
    // cornel_ps.addPrimitive(floor_mesh, floor_checker);
    // // Ceiling 
    // auto ceiling_mesh = createQuadMesh(0.0f, 556.0f, 0.0f, 559.2f, 548.8f, Axis::Y);
    // cornel_ps.addPrimitive(ceiling_mesh, white_diffuse);
    // // Back wall 
    // auto back_wall_mesh = createQuadMesh(0.0f, 556.0f, 0.0f, 548.8f, 559.2f, Axis::Z);
    // cornel_ps.addPrimitive(back_wall_mesh, white_diffuse);
    // // Right wall 
    // auto right_wall_mesh = createQuadMesh(0.0f, 548.8f, 0.0f, 559.2f, 0.0f, Axis::X);
    // cornel_ps.addPrimitive(right_wall_mesh, red_diffuse);
    // // Left wall 
    // auto left_wall_mesh = createQuadMesh(0.0f, 548.8f, 0.0f, 559.2f, 556.0f, Axis::X);
    // cornel_ps.addPrimitive(left_wall_mesh, green_diffuse);
    // // Ceiling light
    // auto ceiling_light_mesh = createQuadMesh(-100.0f, 100.0f, -100.0f, 100.0f, 100.0f, Axis::Y);
    // auto cornel_ps = PrimitiveInstance(Transform());
    // auto light_sphere1 = std::make_shared<Sphere>(make_float3(0.0f, 200.0f, 0.0f), 50.0f);
    // auto light1 = std::make_shared<AreaEmitter>(make_float3(1.0, 0.1f, 0.0f), 10.0f);
    // cornel_ps.addPrimitive(light_sphere1, light1);

    // scene.addPrimitiveInstance(cornel_ps);

    auto ground_matrix = sutil::Matrix4x4::translate(make_float3(0.0f, -275.0f, 0.0f));
    auto ground_ps = PrimitiveInstance(ground_matrix);
    auto ground = std::make_shared<Plane>(make_float2(-500.0f, -500.0f), make_float2(500.0f, 500.0f));
    ground_ps.addPrimitive(ground, white_diffuse);
    scene.addPrimitiveInstance(ground_ps);

    // Armadillo
    auto armadillo_matrix = sutil::Matrix4x4::translate(make_float3(250.0f, -210.0f, -150.0f)) 
                          * sutil::Matrix4x4::scale(make_float3(1.2f));
    auto armadillo_ps = PrimitiveInstance(armadillo_matrix);
    auto armadillo = createTriangleMesh("model/Armadillo.ply");
    auto metal = std::make_shared<Conductor>(make_float3(0.8f, 0.8f, 0.2f), 0.01f);
    armadillo_ps.addPrimitive(armadillo, metal);
    scene.addPrimitiveInstance(armadillo_ps);

    // Center bunny with lambert material
    auto bunny2_matrix = sutil::Matrix4x4::translate(make_float3(-50.0f, -275.0f, 300.0f)) 
                       * sutil::Matrix4x4::rotate(M_PIf, make_float3(0.0f, 1.0f, 0.0f))
                       * sutil::Matrix4x4::scale(make_float3(1200.0f));
    auto bunny2_ps = PrimitiveInstance(bunny2_matrix);
    auto bunny2 = createTriangleMesh("model/uv_bunny.obj");
    bunny2_ps.addPrimitive(bunny2, disney);
    scene.addPrimitiveInstance(bunny2_ps);

    // Teapot
    auto teapot_matrix = sutil::Matrix4x4::translate(make_float3(-250.0f, -275.0f, -150.0f)) 
                       * sutil::Matrix4x4::scale(make_float3(40.0f));
    auto teapot_ps = PrimitiveInstance(teapot_matrix);
    auto teapot = createTriangleMesh("model/teapot_normal_merged.obj");
    teapot_ps.addPrimitive(teapot, teapot_diffuse);
    scene.addPrimitiveInstance(teapot_ps);

    // Sphere 1
    auto earth_sphere_matrix = sutil::Matrix4x4::translate(make_float3(250.0f, -185.0f, 150.0f))
                             * sutil::Matrix4x4::rotate(M_PIf, make_float3(0.0f, 1.0f, 0.0f));
    auto earth_sphere_ps = PrimitiveInstance(earth_sphere_matrix);
    auto earth_sphere = std::make_shared<Sphere>(make_float3(0.0f), 90.0f);
    earth_sphere_ps.addPrimitive(earth_sphere, earth_diffuse);
    scene.addPrimitiveInstance(earth_sphere_ps);

    // Sphere 2
    auto glass_sphere_matrix = sutil::Matrix4x4::translate(make_float3(-250.0f, -195.0f, 150.0f)) 
                             * sutil::Matrix4x4::rotate(M_PIf, make_float3(1.0f, 0.0f, 0.0f));
    auto glass_sphere_ps = PrimitiveInstance(glass_sphere_matrix);
    auto glass_sphere = std::make_shared<Sphere>(make_float3(0.0f), 80.0f);
    glass_sphere_ps.addPrimitive(glass_sphere, glass);
    scene.addPrimitiveInstance(glass_sphere_ps);

    // Cylinder
    auto cylinder_matrix = sutil::Matrix4x4::translate(make_float3(0.0f, -220.0f, -300.0f));
    auto cylinder_ps = PrimitiveInstance(cylinder_matrix);
    auto cylinder = std::make_shared<Cylinder>(60.0f, 100.0f);
    cylinder_ps.addPrimitive(cylinder, floor_checker);
    scene.addPrimitiveInstance(cylinder_ps);

    return scene;
}