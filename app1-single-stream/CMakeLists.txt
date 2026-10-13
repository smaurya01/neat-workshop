cmake_minimum_required(VERSION 3.16)
project(app1_single_stream LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SimaNeat REQUIRED CONFIG)

add_executable(app1_single_stream main.cpp)
target_link_libraries(app1_single_stream PRIVATE SimaNeat::sima_neat)
