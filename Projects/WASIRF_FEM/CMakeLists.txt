if (CUDA_FOUND)

add_cuda_executable(wasirf_fem)
target_sources(wasirf_fem
    PRIVATE     wasirf_fem.cu
)
target_link_libraries(wasirf_fem
    PRIVATE     mnio
                fmt
)

endif()