PRAYGROUND_add_executalbe(thumbnail target_name
    main.cpp 
    app.cpp 
    app.h
    params.h
    cuda/hitgroups.cu
    cuda/surfaces.cu
    cuda/miss.cu
    cuda/raygen.cu
    cuda/textures.cu
    cuda/util.cuh
)

target_compile_definitions(
    ${target_name}
    PRIVATE
    APP_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(${target_name} ${CUDA_LIBRARIES})