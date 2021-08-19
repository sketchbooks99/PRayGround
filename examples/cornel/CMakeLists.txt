OPRT_add_executalbe(cornel target_name
    main.cpp 
    app.cpp 
    app.h
    params.h
)

target_compile_definitions(
    cornel
    PRIVATE
    OPRT_APP_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    OPRT_ROOT_DIR="${OPRT_DIR}"
)

target_link_libraries(${target_name} ${CUDA_LIBRARIES})