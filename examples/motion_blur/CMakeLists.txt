PRAYGROUND_add_executable(motion_blur target_name
    main.cpp 
    app.cpp 
    app.h
    params.h
    cuda/hitgroups.cu
    cuda/miss.cu
    cuda/raygen.cu
    cuda/textures.cu
    cuda/util.cuh
)

target_compile_definitions(
    ${target_name}
    PRIVATE
    APP_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    APP_NAME_DEFINE="${target_name}"
)

target_link_libraries(${target_name} ${CUDA_LIBRARIES})