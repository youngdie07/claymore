message("## Setup CUDA in setup_cuda.cmake")
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  message("-- cuda-compiler " ${CMAKE_CUDA_COMPILER})
else()
  message(STATUS "No CUDA support")
endif()
set(CUDA_FOUND ${CMAKE_CUDA_COMPILER})

#  --------- READ ME IMPORTANT --------- Justin Bonus
# Set GPU architecture(s) to compile code for. Very important!
# Use the TARGET_CUDA_ARCH appropiate for the Compute Capability of your GPU below
# To determine CC of your GPU, try 'nvidia-smi' in console to get GPU model, then Google search
# Recommended minimum of CC 6.0, some backward compatability available
# You can compile for one GPU architecture efficiently or a wide-range less efficiently
# Just-in-time (JIT) compilation can affect software performance but is convenient.
# Learn the difference between -arch, -gencode, and -code.
# Reference: http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

# --------- Mox Hyak HPC System - RTX 2080ti GPUs - Univ. of Wash. - Justin Bonus
#set(TARGET_CUDA_ARCH -gencode=arch=compute_75,code=compute_75)

# --------- Frontera TACC HPC System - RTX Quadro 4000 GPUs - Univ. of Texas - Justin Bonus
#set(TARGET_CUDA_ARCH -gencode=arch=compute_75,code=compute_75)

# --------- Dell G7 Laptop - GTX 1060 Max-Q GPU - Univ. of Wash. - Justin Bonus
#set(TARGET_CUDA_ARCH -gencode=arch=compute_61,code=sm_61)
set(TARGET_CUDA_ARCH -arch=sm_61)

# --------- Set NVCC compiler flags ---------
# NOTE: I removed compiler flags such as --use_fast_math to improve engineering accuracy
# Reference: https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
function(CUDA_CONVERT_FLAGS EXISTING_TARGET)
    get_property(old_flags TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS)
    if(NOT "${old_flags}" STREQUAL "")
        string(REPLACE ";" "," CUDA_flags "${old_flags}")
        set_property(TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS
            "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>"
            )
    endif()
endfunction()

function(add_cuda_executable binary)
  if(CUDA_FOUND)
    add_executable(${binary} ${ARGN})
    # seems not working
    target_compile_options(${binary} 
      PRIVATE     $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS} ${TARGET_CUDA_ARCH} --expt-extended-lambda --expt-relaxed-constexpr --default-stream=per-thread --fmad=true -lineinfo --ptxas-options=-allow-expensive-optimizations=true>
    )
    target_compile_features(${binary} PRIVATE cuda_std_14)
    set_target_properties(${binary}
      PROPERTIES  CUDA_EXTENSIONS ON
                  CUDA_SEPARABLE_COMPILATION ON
                  #LINKER_LANGUAGE CUDA
                  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(${binary}
        PRIVATE mncuda
    )
    message("-- [${binary}]\tcuda executable build config")
  endif()
endfunction(add_cuda_executable)

function(add_cuda_library library)
  if(CUDA_FOUND)
    add_library(${library} ${ARGN})
    # Seems to not be working
    target_compile_options(${library} 
      PUBLIC        $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS} ${TARGET_CUDA_ARCH} --expt-extended-lambda --expt-relaxed-constexpr --default-stream=per-thread -lineinfo --fmad=true --ptxas-options=-allow-expensive-optimizations=true>
    )
    #target_link_options(${library} 
    #  PRIVATE       $<$<LINKER_LANGUAGE:CUDA>:-arch=sm_75>
    #)
    target_compile_features(${library} PRIVATE cuda_std_14)
    set_target_properties(${library}
      PROPERTIES  CUDA_EXTENSIONS ON
                  CUDA_SEPARABLE_COMPILATION ON
                  CUDA_RESOLVE_DEVICE_SYMBOLS OFF
                  POSITION_INDEPENDENT_CODE ON
                  #LINKER_LANGUAGE CUDA
    )
    target_compile_definitions(${library} 
      PUBLIC        CMAKE_GENERATOR_PLATFORM=x64
    )
    message("-- [${library}]\tcuda executable build config")
  endif()
endfunction(add_cuda_library)