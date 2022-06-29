if (CUDA_FOUND)

add_cuda_executable(fbi)
target_sources(fbi
    PRIVATE     fbi.cu
)
target_link_libraries(fbi
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