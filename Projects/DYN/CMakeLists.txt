if (CUDA_FOUND)

add_cuda_executable(dyn)
target_sources(dyn
    PRIVATE     dyn.cu
)
target_link_libraries(dyn
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
