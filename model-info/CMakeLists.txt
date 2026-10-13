cmake_minimum_required(VERSION 3.16)
project(model_info LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SimaNeat REQUIRED CONFIG)

add_executable(model_info main.cpp)
target_link_libraries(model_info PRIVATE SimaNeat::sima_neat)
