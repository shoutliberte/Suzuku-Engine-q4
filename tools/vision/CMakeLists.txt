# q4-vision: the CPU image encoder q4's server starts for pictures
# (llama.cpp mtmd + the model's mmproj).  scripts/build-vision.sh drives it:
#
#   cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release \
#         -DLLAMA_DIR=<llama.cpp checkout>
#   cmake --build build-vision --target q4-vision
#
# Native CPU build (this machine's AVX-512 / BF16), no GPU backend: the
# encoder never opens the card.
cmake_minimum_required(VERSION 3.21)
project(q4_vision C CXX)

set(LLAMA_DIR "" CACHE PATH "llama.cpp checkout (with tools/mtmd)")
if(NOT EXISTS "${LLAMA_DIR}/tools/mtmd/mtmd.h")
  message(FATAL_ERROR "LLAMA_DIR=${LLAMA_DIR} is not a llama.cpp checkout with tools/mtmd")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
foreach(_o LLAMA_BUILD_COMMON LLAMA_BUILD_TOOLS LLAMA_BUILD_TESTS LLAMA_BUILD_EXAMPLES
           LLAMA_BUILD_SERVER LLAMA_BUILD_APP LLAMA_OPENSSL LLAMA_CURL
           GGML_CUDA GGML_HIP GGML_VULKAN GGML_METAL GGML_SYCL GGML_BACKEND_DL)
  set(${_o} OFF CACHE BOOL "" FORCE)
endforeach()
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(GGML_NATIVE ON CACHE BOOL "" FORCE)
set(GGML_OPENMP ON CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_DIR} llama EXCLUDE_FROM_ALL)

add_executable(q4-vision q4_vision_helper.c)
target_link_libraries(q4-vision PRIVATE mtmd llama ggml)
target_include_directories(q4-vision PRIVATE ${LLAMA_DIR}/tools/mtmd)
set_target_properties(q4-vision PROPERTIES LINKER_LANGUAGE CXX
                      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
