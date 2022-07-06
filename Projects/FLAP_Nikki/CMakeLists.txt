if (CUDA_FOUND)

add_cuda_executable(flap_nikki)
target_sources(flap_nikki
    PRIVATE     flap_nikki.cu
)
target_link_libraries(flap_nikki
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