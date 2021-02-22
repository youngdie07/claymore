if (CUDA_FOUND)

add_cuda_executable(osu)
target_sources(osu
    PRIVATE     osu.cu
)
target_link_libraries(osu
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
