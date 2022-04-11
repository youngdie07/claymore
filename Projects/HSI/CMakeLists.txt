if (CUDA_FOUND)

add_cuda_executable(hsi)
target_sources(hsi
    PRIVATE     hsi.cu
)
target_link_libraries(hsi
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