if (CUDA_FOUND)

add_cuda_executable(osu_mgsp)
target_sources(osu_mgsp 
    PRIVATE     osu_mgsp.cu
)
target_link_libraries(osu_mgsp 
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