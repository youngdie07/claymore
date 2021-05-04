if (CUDA_FOUND)

add_cuda_executable(imp)
target_sources(imp
    PRIVATE     imp.cu
)
target_link_libraries(imp
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
