cmake_minimum_required(VERSION 3.10)
project(camsub)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
#find_package(TENSORRT REQUIRED)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${TENSORRT_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link directories
link_directories(${CUDA_LIBRARIES})
link_directories(${TENSORRT_LIB_DIR})

# Add source files
set(SOURCES
    src/my_main.cpp
    src/my_ImageProcessor.cpp
    src/my_TensorRTEngine.cpp
)

#build
add_executable(learn ${SOURCES})
ament_target_dependencies(learn rclcpp sensor_msgs cv_bridge OpenCV)

# Include directories for the target
target_include_directories(learn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link libraries
target_link_libraries(learn
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer
    nvinfer_plugin
    nvonnxparser
)

# Install
install(TARGETS
  learn
  DESTINATION lib/${PROJECT_NAME})

ament_package()
