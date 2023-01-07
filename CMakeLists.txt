cmake_minimum_required(VERSION 3.22)
project(HalideExample)

set(CMAKE_CXX_STANDARD 17)  # or newer
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_package(Halide REQUIRED)

add_executable(gen gen.cpp)
target_link_libraries(gen PRIVATE Halide::Halide)
