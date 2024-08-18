PRAYGROUND_add_executable(rtcamp10 target_name
    main.cpp 
    app.cpp 
    app.h
    kernels.cu
    params.h
)

target_compile_definitions(
    ${target_name}
    PRIVATE
    APP_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(${target_name} ${CUDA_LIBRARIES})