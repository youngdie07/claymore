if (CUDA_FOUND)

add_cuda_executable(flap)
target_sources(flap
    PRIVATE     flap.cu
)
target_link_libraries(flap
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