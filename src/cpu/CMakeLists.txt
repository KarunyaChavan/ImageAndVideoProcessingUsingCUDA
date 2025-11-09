cmake_minimum_required(VERSION 3.10)
project(main)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Collect all .cpp files automatically
file(GLOB SRC "*.cpp")

# Use modern C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add executable
add_executable(main ${SRC})

# Link OpenCV libraries
target_link_libraries(main ${OpenCV_LIBS})

# Optional: show build info
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")
