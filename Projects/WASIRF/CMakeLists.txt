if (CUDA_FOUND)

add_cuda_executable(wasirf)
target_sources(wasirf
    PRIVATE     wasirf.cu
)
target_link_libraries(wasirf
    PRIVATE     mnio
                cxxopts
                spdlog
                fmt
                rapidjson
                range-v3
                filesystem
                stdc++fs
)

endif()
