cmake_minimum_required(VERSION 3.16)
project(app2_dual_stream LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SimaNeat REQUIRED CONFIG)

add_executable(app2_dual_stream main.cpp)
target_link_libraries(app2_dual_stream PRIVATE SimaNeat::sima_neat)
