# Set CUDA source properties BEFORE macro call
set(cuda_obj_sources envmap.cu textures.cu bloom.cu svgf.cu)
set_source_files_properties(${cuda_obj_sources} PROPERTIES 
    CUDA_SOURCE_PROPERTY_FORMAT OBJ
)

PRAYGROUND_add_executable(rtcamp11 target_name
    main.cpp 
    app.cpp 
    app.h
    kernels.cu
    params.h
    textures.h
    envmap.cu
    envmap.cuh
    textures.cuh
    textures.cu
    bloom.cuh
    bloom.cu
    svgf.cuh
    svgf.cu
    OPTIONS -rdc true
)

target_compile_definitions(
    ${target_name}
    PRIVATE
    APP_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(${target_name} ${CUDA_LIBRARIES})
