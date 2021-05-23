if (CUDA_FOUND)

add_cuda_executable(wasirf_old)
target_sources(wasirf_old
    PRIVATE     wasirf_old.cu
)
target_link_libraries(wasirf_old
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
