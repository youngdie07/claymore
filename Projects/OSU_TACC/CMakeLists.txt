if (CUDA_FOUND)

add_cuda_executable(osu_tacc)
target_sources(osu_tacc
    PRIVATE     osu_tacc.cu
)
target_link_libraries(osu_tacc
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