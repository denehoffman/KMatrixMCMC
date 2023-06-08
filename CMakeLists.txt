# tested with 3.26
cmake_minimum_required(VERSION 3.26)
project(
  KMatrixMCMC
  VERSION 1.0
  DESCRIPTION "MCMC on the KMatrix Amplitude"
  LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find required packages
find_package(Armadillo REQUIRED)
find_package(ROOT REQUIRED)

# Include directories
include_directories(${ARMADILLO_INCLUDE_DIRS})
include_directories(${ROOT_INCLUDE_DIRS})

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
