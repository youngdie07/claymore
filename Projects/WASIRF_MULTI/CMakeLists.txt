if (CUDA_FOUND)

add_cuda_executable(wasirf_multi)
target_sources(wasirf_multi
    PRIVATE     wasirf_multi.cu
)
target_link_libraries(wasirf_multi 
    PRIVATE     mnio
                fmt
)

endif()