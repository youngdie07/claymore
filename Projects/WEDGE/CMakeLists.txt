if (CUDA_FOUND)

add_cuda_executable(wedge)
target_sources(wedge
    PRIVATE     wedge.cu
)
target_link_libraries(wedge
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