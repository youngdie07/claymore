if (CUDA_FOUND)

add_cuda_executable(osu_new)
target_sources(osu_new
    PRIVATE     osu_new.cu
)
target_link_libraries(osu_new
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
