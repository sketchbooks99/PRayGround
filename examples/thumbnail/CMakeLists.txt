PRAYGROUND_add_executalbe(thumbnail target_name
    main.cpp 
    app.cpp 
    app.h
    params.h
    data/hitgroups.cu
    data/surfaces.cu
    data/miss.cu
    data/raygen.cu
    data/textures.cu
    data/util.cuh
)

target_compile_definitions(
    ${target_name}
    PRIVATE
    PRAYGROUND_APP_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    PRAYGROUND_ROOT_DIR="${PRAYGROUND_DIR}"
)

target_link_libraries(${target_name} ${CUDA_LIBRARIES})