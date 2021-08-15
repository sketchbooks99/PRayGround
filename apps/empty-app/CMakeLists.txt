OPTIX_add_sample_executable( empty-app target_name 
    main.cpp 
    app.cpp 
    app.h
)

target_compile_definitions(
    ${target_name}
    PRIVATE
    OPRT_ROOT_DIR="${PROJECT_SOURCE_DIR}"
    OPRT_APP_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries( ${target_name}
    ${CUDA_LIBRARIES}
)