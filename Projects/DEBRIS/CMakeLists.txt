if (CUDA_FOUND)

add_cuda_executable(debris)
target_sources(debris
    PRIVATE     debris.cu
)
target_link_libraries(debris 
    PRIVATE     mnio
                fmt
)

endif()