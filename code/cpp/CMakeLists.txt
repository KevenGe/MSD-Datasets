cmake_minimum_required(VERSION 3.10)

project(HelloWorld)

set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD 23)

set(GFLAGS_USE_TARGET_NAMESPACE ON)

find_package(fmt CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(gflags CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(jsoncpp CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(hdf5 CONFIG REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

#add_executable(main main.cpp)
add_executable(main main_tongyi.cpp)

target_link_libraries(main PRIVATE fmt::fmt)
target_link_libraries(main PRIVATE Eigen3::Eigen)
target_link_libraries(main PRIVATE gflags::gflags)
target_link_libraries(main PRIVATE glog::glog)
target_link_libraries(main PRIVATE JsonCpp::JsonCpp)
target_link_libraries(main PRIVATE ${OpenCV_LIBS})
target_link_libraries(main PRIVATE hdf5::hdf5-shared hdf5::hdf5_hl-shared hdf5::hdf5_cpp-shared hdf5::hdf5_hl_cpp-shared)