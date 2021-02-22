if (CUDA_FOUND)

add_cuda_executable(dam)
target_sources(dam
    PRIVATE     dam.cu
)
target_link_libraries(dam
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
