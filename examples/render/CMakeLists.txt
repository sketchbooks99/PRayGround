PRAYGROUND_add_executalbe(empty_app target_name
    main.cpp 
    app.cpp 
    app.h
    params.h 
    cuda/surfaces.cu 
    cuda/raygen.cu 
    cuda/hitgroups.cu
    cuda/textures.cu
    cuda/utils.cu
)

target_compile_definitions(
    ${target_name}
    PRIVATE
    APP_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(${target_name} ${CUDA_LIBRARIES})