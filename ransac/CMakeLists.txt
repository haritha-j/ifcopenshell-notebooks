cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set (VTK_DIR "/home/haritha/vtk-inst/lib/cmake/vtk-9.2")

project(random_sample_consensus)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (random_sample_consensus ransac.cpp)
target_link_libraries (random_sample_consensus ${PCL_LIBRARIES})