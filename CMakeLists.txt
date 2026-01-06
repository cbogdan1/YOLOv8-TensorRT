cmake_minimum_required(VERSION 3.10)
project(YoloTensorRT)

set(CMAKE_CXX_STANDARD 17)

# Gaseste pachetele necesare
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# Include directoare pentru CUDA si TensorRT
include_directories(/usr/local/cuda/include)
include_directories(/usr/include/aarch64-linux-gnu) 

# Link directoare
link_directories(/usr/local/cuda/lib64)
link_directories(/usr/lib/aarch64-linux-gnu)

add_executable(yolo_app main.cpp)

# Linkeaza bibliotecile: nvinfer este biblioteca principala TensorRT
target_link_libraries(yolo_app ${OpenCV_LIBS} nvinfer cudart)