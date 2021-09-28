PRAYGROUND_add_executalbe(core target_name
    main.cpp
)

target_link_libraries(${target_name} ${CUDA_LIBRARIES})