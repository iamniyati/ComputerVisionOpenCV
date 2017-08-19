cmake_minimum_required(VERSION 3.7)
project(HW05_Shah_Niyati_PartC)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
add_executable(HW05_Shah_Niyati_PartC ${SOURCE_FILES})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( HW05_Shah_Niyati_PartC ${OpenCV_LIBS} )