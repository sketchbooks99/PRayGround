OPRT_add_executalbe(empty-app target_name
    main.cpp 
    app.cpp 
    app.h
)

target_compile_definitions(
    empty-app
    PRIVATE
    OPRT_APP_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    OPRT_ROOT_DIR="${OPRT_DIR}"
)

target_link_libraries(${target_name} ${CUDA_LIBRARIES})