cmake_minimum_required(VERSION 3.19)
project(test_paddle_lite)

set(CMAKE_CXX_STANDARD 17)



find_package(OpenCV REQUIRED core imgproc highgui)
include_directories(${OpenCV_INCLUDE_DIRS})

set(LITE_DIR "${CMAKE_CURRENT_LIST_DIR}/paddle_lite_2.11/inference_lite_lib.with_log/cxx")
set(MKLML_DIR "${CMAKE_CURRENT_LIST_DIR}/paddle_lite_2.11/inference_lite_lib.with_log/third_party/mklml")


#add_library(tflite SHARED IMPORTED)
#set_target_properties(tflite PROPERTIES IMPORTED_LOCATION
#        ${CMAKE_CURRENT_LIST_DIR}/libs/tflite-2.10.0/tflite2.10.0_lib/${ANDROID_ABI}/libtensorflowlite.so)


#link_directories(${LITE_DIR}/lib)
#link_directories(${MKLML_DIR}/lib)

include_directories(${LITE_DIR}/include)
include_directories(${MKLML_DIR}/include)

add_library(paddle_lite_lib SHARED IMPORTED)
set_target_properties(paddle_lite_lib PROPERTIES IMPORTED_LOCATION ${LITE_DIR}/lib/libpaddle_light_api_shared.dylib)
add_library(libiomp5 SHARED IMPORTED)
set_target_properties(libiomp5 PROPERTIES IMPORTED_LOCATION ${MKLML_DIR}/lib/libiomp5.dylib)
add_library(libmklml SHARED IMPORTED)
set_target_properties(libmklml PROPERTIES IMPORTED_LOCATION ${MKLML_DIR}/lib/libmklml.dylib)



add_executable(test_paddle_lite main.cpp)
target_link_libraries(test_paddle_lite paddle_lite_lib)
target_link_libraries(test_paddle_lite libiomp5)
target_link_libraries(test_paddle_lite libmklml)
target_link_libraries(test_paddle_lite ${OpenCV_LIBS})
