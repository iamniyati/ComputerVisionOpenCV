cmake_minimum_required(VERSION 3.7)
project(Hw6)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
add_executable(Hw6 ${SOURCE_FILES})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package( OpenCV REQUIRED )
set(SOURCE_FILES main.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( Hw6 ${OpenCV_LIBS} )