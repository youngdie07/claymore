if (CUDA_FOUND)

add_cuda_executable(fsi)
target_sources(fsi
    PRIVATE     fsi.cu
)
target_link_libraries(fsi
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