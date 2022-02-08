PRAYGROUND_add_executalbe(core target_name
    # main.cpp
    # load_and_write_hdr.cpp
    spectrum.cpp
)

target_compile_definitions(
    ${target_name}
    PRIVATE
    APP_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(${target_name} ${CUDA_LIBRARIES})