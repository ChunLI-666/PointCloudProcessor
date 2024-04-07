cmake_minimum_required(VERSION 3.18)
project(PointCloudColorization LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")

# set(Open3D_DIR ${HOME}/open3d_install/lib/cmake/Open3D)
# message(STATUS "Open3D_DIR: ${Open3D_DIR}")

find_package(PCL 1.8 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Open3D REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  tf
  eigen_conversions
  cv_bridge
)


# list(APPEND Open3D_LIBRARIES dl)
if (Open3D_FOUND)
    message(STATUS "Found Open3D ${Open3D_VERSION}")
    link_directories(${Open3D_LIBRARY_DIRS})
endif()

include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS} include)

add_definitions(${PCL_DEFINITIONS})

add_executable(${PROJECT_NAME} src/main.cpp src/FrameData.cpp src/PointCloudProcessor.cpp src/RGBFrames.cpp)

target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES} Open3D::Open3D)
# target_link_libraries(${PROJECT_NAME} PRIVATE Open3D::Open3D)