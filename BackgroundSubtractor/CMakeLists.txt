cmake_minimum_required(VERSION 3.7)
project(HW03_Shah_Niyati)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES OverlapChangeImage.cpp)
add_executable(HW03_Shah_Niyati ${SOURCE_FILES})


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( HW03_Shah_Niyati ${OpenCV_LIBS} )