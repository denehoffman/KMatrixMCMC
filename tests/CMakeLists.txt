Include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.3.2
)

FetchContent_MakeAvailable(Catch2)

add_executable(tests)

target_sources(tests PRIVATE test_kmatrix.cpp)
target_link_libraries(tests PRIVATE kmatrixmcmc_library ${ARMADILLO_LIBRARIES} ${ROOT_LIBRARIES} Catch2::Catch2WithMain)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)
include(CTest)
include(Catch)
catch_discover_tests(tests)
