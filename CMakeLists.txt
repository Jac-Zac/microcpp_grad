cmake_minimum_required(VERSION 3.15.0)
project(pymicrograd)

set(CMAKE_CROSSCOMPILING "True")

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "-O0")

file(GLOB_RECURSE SOURCES
      # examples/training.cpp
      python/**.cpp
)

add_subdirectory(pybind11)
pybind11_add_module(pymicrograd ${SOURCES})

target_include_directories(pymicrograd PRIVATE include)
set_target_properties(pymicrograd PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python")

# Add the include directory to the micrograd library as well
add_library(micrograd INTERFACE)
target_include_directories(micrograd INTERFACE include)

# add_executable(test_executable ${SOURCES})
# target_link_libraries(test_executable micrograd)
