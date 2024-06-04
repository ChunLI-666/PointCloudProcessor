cmake_minimum_required(VERSION 3.18)
project(PointCloudColorization LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")

set(Open3D_DIR ${HOME}/open3d_install/lib/cmake/Open3D)
# message(STATUS "Open3D_DIR: ${Open3D_DIR}")

find_package(PCL 1.8 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Open3D REQUIRED)
find_package(Boost 1.65 REQUIRED COMPONENTS filesystem program_options)

# For VLCAL
find_package(Ceres REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

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

### For VLCAL
add_library(direct_visual_lidar_calibration SHARED
  src/camera/create_camera.cpp
  src/vlcal/common/estimate_fov.cpp
  src/vlcal/calib/view_culling.cpp
  src/vlcal/calib/visual_camera_calibration.cpp
)
target_include_directories(direct_visual_lidar_calibration PUBLIC
  include
  thirdparty/Sophus
  ${Boost_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(direct_visual_lidar_calibration
  fmt
  ${Boost_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
  # ${GTSAM_LIBRARIES}
  ${OpenCV_LIBRARIES}
  # ${Iridescence_LIBRARIES}
)

# include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS} include)

# include_directories(${Boost_INCLUDE_DIRS})

# add_definitions(${PCL_DEFINITIONS})

# add_executable(${PROJECT_NAME} src/main.cpp src/FrameData.cpp src/PointCloudProcessor.cpp src/RGBFrames.cpp src/cloudSmooth.cpp)

# target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES} ${Boost_LIBRARIES} Open3D::Open3D)
# # target_link_libraries(${PROJECT_NAME} PRIVATE Open3D::Open3D)

add_executable(${PROJECT_NAME} 
  src/main.cpp 
  src/FrameData.cpp 
  src/PointCloudProcessor.cpp 
  src/RGBFrames.cpp 
  src/cloudSmooth.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
  ${catkin_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
  ${PCL_LIBRARIES} 
  ${OpenCV_LIBS} 
  ${catkin_LIBRARIES} 
  ${Boost_LIBRARIES} 
  Open3D::Open3D 
  direct_visual_lidar_calibration
)
