if (CUDA_FOUND)

add_cuda_executable(dsi)
target_sources(dsi
    PRIVATE     dsi.cu
)
target_link_libraries(dsi
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