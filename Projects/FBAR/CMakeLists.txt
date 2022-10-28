if (CUDA_FOUND)

add_cuda_executable(fbar)
target_sources(fbar
    PRIVATE     fbar.cu
)
target_link_libraries(fbar
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