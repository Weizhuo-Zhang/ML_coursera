cmake_minimum_required(VERSION 3.5)
project(test1)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(${PROJECT_NAME} opencv_demo.cpp)
target_link_libraries(${PROJECT_NAME} opencv_core opencv_highgui opencv_imgcodecs)
