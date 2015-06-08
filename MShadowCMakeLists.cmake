message("Including MShadow")

add_definitions(-DMSHADOW_USE_CBLAS)
add_definitions(-DMSHADOW_USE_MKL=0)

# packages
find_package(CUDA REQUIRED)

INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/third_party/mshadow)

SET(CUDA_EXTRA_FLAGS "-fno-strict-aliasing")

SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -L/usr/local/cuda/lib")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CUDA_EXTRA_FLAGS}")

LIST(APPEND CUDA_NVCC_FLAGS --compiler-options ${CUDA_EXTRA_FLAGS} -lineinfo -Xptxas -dlcm=cg  -use_fast_math -std=c++11)

# here we should essentially do all the cuda stuff
# then:
# target_link_libraries(mainCudaLib ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} cuda cudart)
