PRAYGROUND_add_executalbe(load3d target_name
    # main.cpp
    load3d.cpp
)

target_compile_definitions(
    ${target_name}
    PRIVATE
    APP_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(${target_name} ${CUDA_LIBRARIES})