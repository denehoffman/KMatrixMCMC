# tested with 3.22 and 3.26
cmake_minimum_required(VERSION 3.22)
project(
  KMatrixMCMC
  VERSION 1.0
  DESCRIPTION "MCMC on the KMatrix Amplitude"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set default build type to "Default" if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Default)
endif()

# Add the desired build types and flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Set additional flags for the "Debug" build type
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Wextra -O3")

message("C++ Compiler: ${CMAKE_CXX_COMPILER}")

# Find required packages
find_package(OpenMP)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(Armadillo REQUIRED)
find_package(ROOT REQUIRED)

include(FetchContent)
# Fetch Tyche library
FetchContent_Declare(
  Tyche
  GIT_REPOSITORY https://github.com/denehoffman/Tyche.git
  GIT_TAG main
)
FetchContent_MakeAvailable(Tyche)

# Include directories
include_directories(${HDF5_INCLUDE_DIRS})
include_directories(${ARMADILLO_INCLUDE_DIRS})
include_directories(${ROOT_INCLUDE_DIRS})
include_directories(${TYCHE_INCLUDE_DIRS})

# Add subdirectories
add_subdirectory(src)
add_subdirectory(apps)

# Add a custom target for generating Doxygen documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(docs
      COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating API documentation with Doxygen"
      VERBATIM
    )
else()
    message(WARNING "Doxygen not found. Documentation will not be generated.")
endif()

set(INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/bin")
install(
  TARGETS kmatrix_mcmc
  RUNTIME DESTINATION ${INSTALL_DIR}
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)


enable_testing()
add_subdirectory(tests)
