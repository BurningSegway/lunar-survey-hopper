cmake_minimum_required(VERSION 3.10)
project(Sim2Real_Jumper)

find_package(glfw3 REQUIRED)

set(MUJOCO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mujoco-3.2.3")
find_library(MUJOCO_LIBRARY
    NAMES mujoco
    PATHS "${MUJOCO_DIR}/lib"
    NO_DEFAULT_PATH
)
if(NOT MUJOCO_LIBRARY)
    message(FATAL_ERROR "Could not find the bundled MuJoCo library in ${MUJOCO_DIR}/lib")
endif()

add_library(controller_core STATIC
    rw_controller.cpp
    lqr_controller.cpp
)
target_include_directories(controller_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(controller_core PUBLIC cxx_std_17)

add_executable(jumper_sim main.cpp)
target_include_directories(jumper_sim PRIVATE "${MUJOCO_DIR}/include")
target_compile_features(jumper_sim PRIVATE cxx_std_17)
target_link_libraries(jumper_sim PRIVATE
    controller_core
    "${MUJOCO_LIBRARY}"
    glfw
    GL
    m
)

include(CTest)
if(BUILD_TESTING)
    add_executable(lqr_controller_test lqr_controller_test.cpp)
    target_link_libraries(lqr_controller_test PRIVATE controller_core)
    add_test(NAME lqr_controller_test COMMAND lqr_controller_test)
    add_executable(wheel_rest_controller_test wheel_rest_controller_test.cpp)
    target_link_libraries(wheel_rest_controller_test PRIVATE controller_core)
    add_test(NAME wheel_rest_controller_test COMMAND wheel_rest_controller_test)
endif()