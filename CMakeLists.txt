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
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -Wl,--trace")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Wextra -O3")

# Find required packages
find_package(Armadillo REQUIRED)
find_package(ROOT REQUIRED)

find_library(MCMC_LIBRARIES NAMES mcmc)

if(NOT MCMC_LIBRARIES)
  message(FATAL_ERROR "mcmc library not found")
else()
  message(STATUS "mcmc library found at: ${MCMC_LIBRARIES}")
endif()


# Include directories
include_directories(${ARMADILLO_INCLUDE_DIRS})
include_directories(${ROOT_INCLUDE_DIRS})
get_filename_component(MCMC_LIBRARIES_DIR ${MCMC_LIBRARIES} DIRECTORY)
include_directories(${MCMC_LIBRARIES_DIR}/../include/mcmc)

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
