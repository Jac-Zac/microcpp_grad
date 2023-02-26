cmake_minimum_required(VERSION 3.10)
project(test CXX)

set(CMAKE_CROSSCOMPILING "True")

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "-O0")

file(GLOB_RECURSE SOURCES
      micrograd/*.cpp
      micrograd/**/*.cpp
      test/**.cpp
      # examples/**.cpp
)
file(GLOB_RECURSE HEADERS
      micrograd/*hpp
      micrograd/**/*.hpp
)

add_executable(test_executable ${SOURCES})
target_precompile_headers(test_executable PUBLIC ${HEADERS})
