# cmake_minimum_required(VERSION 2.8)
# project(gpu_img_proc)
# find_package(CUDA REQUIRED)
# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS} -D WITH_CUDA=ON)
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; "-arch=sm_35; -ccbin /usr/bin/gcc-5 ; -std=c++11; -O2; -DVERBOSE;") 
# cuda_add_executable(main main.cu kernel.cu kernelcall.cu)
# target_link_libraries(main ${OpenCV_LIBS})

cmake_minimum_required(VERSION 3.10)
project(gpu_img_proc)

# --- Find CUDA ---
find_package(CUDA REQUIRED)

# --- Find OpenCV ---
# Use system-installed OpenCV 4 (from apt)
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
find_package(OpenCV REQUIRED)

# --- Include paths and defines ---
include_directories(${OpenCV_INCLUDE_DIRS})
add_definitions(-DWITH_CUDA=ON)

# --- CUDA compile flags ---
# RTX 3050 → Ampere architecture = sm_86
# g++ 11.4.0 is compatible, so no need to specify -ccbin
set(CUDA_NVCC_FLAGS 
    ${CUDA_NVCC_FLAGS};
    "-arch=sm_86; -std=c++14; -O2; -DVERBOSE;"
)

# --- Build target ---
cuda_add_executable(main main.cu kernel.cu kernelcall.cu)
target_link_libraries(main ${OpenCV_LIBS})

