cmake_minimum_required(VERSION 3.12)
project(KMatrixMCMC)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-O3")

find_package(Armadillo REQUIRED)
find_package(ROOT REQUIRED)

include_directories(${ARMADILLO_INCLUDE_DIRS})
include_directories(${ROOT_INCLUDE_DIRS})

set(SOURCES
    file1.cpp
    file2.cpp
)

set(HEADERS
    file1.h
    file2.h
)

add_executable(run_mcmc ${SOURCES})

target_link_libraries(run_mcmc ${ARMADILLO_LIBRARIES})
target_link_libraries(run_mcmc ${ROOT_LIBRARIES})
