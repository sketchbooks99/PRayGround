PRAYGROUND_add_executable(rayTracingNextWeek target_name
    main.cpp 
    app.cpp 
    app.h
    box_medium.h
    box_medium.cpp
    sphere_medium.h
    sphere_medium.cpp
    params.h
    kernels.cu
)

target_compile_definitions(
    ${target_name}
    PRIVATE
    APP_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    APP_NAME_DEFINE="${target_name}"
)

target_link_libraries(${target_name} ${CUDA_LIBRARIES})