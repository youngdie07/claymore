if (CUDA_FOUND)

add_cuda_executable(multi)
target_sources(multi
    PRIVATE     multi.cu
)
target_link_libraries(multi 
    PRIVATE     mnio
                fmt
)

endif()