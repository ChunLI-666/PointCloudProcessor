cmake_minimum_required(VERSION 3.5)
project(PointCloudColorization)

set(CMAKE_CXX_STANDARD 14)

find_package(PCL 1.8 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} include)

add_definitions(${PCL_DEFINITIONS})

add_executable(${PROJECT_NAME} src/main.cpp src/FrameData.cpp src/PointCloudProcessor.cpp src/RGBFrames.cpp)

target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} ${OpenCV_LIBS})
