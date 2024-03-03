# 项目名称，随便写
PROJECT(image_onnx)
# cmake版本，根据自己的写
cmake_minimum_required(VERSION 3.0)
 
# 编译好的可执行文件放置的位置
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${image_onnx_SOURCE_DIR}/bin)
 
# find required opencv
find_package(OpenCV REQUIRED)
# directory of opencv headers
include_directories(${OpenCV_INCLUDE_DIRS})
 
set(CMAKE_BUILD_TYPE Debug)

# 根据自己的onnxruntime存放路径编写
set(ONNXRUNTIME_ROOT_PATH /mnt/d/linux_project/onnxruntime-linux-x64-gpu-1.15.1/)
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include/)
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}lib/libonnxruntime.so)
 
# 需要编译的cpp文件所在路径，前面是编译好的可执行文件名
add_executable(image_onnx main.cpp
src/ort_infer.cpp)
# directory of opencv library
link_directories(${OpenCV_LIBRARY_DIRS})
# opencv libraries
target_link_libraries(image_onnx ${OpenCV_LIBS})
 
include_directories(${ONNXRUNTIME_INCLUDE_DIRS})
target_link_libraries(image_onnx ${ONNXRUNTIME_LIB})
 
# include
target_include_directories(image_onnx
    PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
)
