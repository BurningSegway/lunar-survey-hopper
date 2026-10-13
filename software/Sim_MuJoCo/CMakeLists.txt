cmake_minimum_required(VERSION 3.10)
project(Sim2Real_Jumper)

set(CMAKE_CXX_STANDARD 17)

find_package(glfw3 REQUIRED)

set(MUJOCO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mujoco-3.2.3)
include_directories(${MUJOCO_DIR}/include)
link_directories(${MUJOCO_DIR}/lib)

add_executable(jumper_sim main.cpp rw_controller.cpp)
target_link_libraries(jumper_sim mujoco glfw GL m)