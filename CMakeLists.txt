#
# file : sep2dfilt/CMakeLists.txt
#

cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(sep2dfilt LANGUAGES CXX)

message("CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")

set(CMAKE_CXX_STANDARD  11)

set(OPENCV_INCLUDE_DIR "/home/golan/opencv4_1_1/modules"                    
                       "/home/golan/opencv4_1_1/modules/core/include"       
                       "/home/golan/opencv4_1_1/build"                      
                       "/home/golan/opencv4_1_1/modules/imgcodecs/include" 
                       "/home/golan/opencv4_1_1/modules/videoio/include" )  

set(OPENCV_LIB_DIR     "/home/golan/opencv4_1_1/build/lib")

include_directories("${OPENCV_INCLUDE_DIR}")

link_directories("${OPENCV_LIB_DIR}")
file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Source/*.cpp")
add_executable(sep2dfilt "${SOURCES}")
target_link_libraries(sep2dfilt LINK_PUBLIC opencv_core opencv_highgui opencv_imgcodecs opencv_imgproc)