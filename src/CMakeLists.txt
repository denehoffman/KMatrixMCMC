set(SOURCES
  Amplitude.cpp
  DataReader.cpp
  KMatrix.cpp
  Likelihood.cpp)

add_library(kmatrixmcmc_library ${SOURCES})
target_include_directories(kmatrixmcmc_library PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

target_link_libraries(kmatrixmcmc_library PRIVATE ${ARMADILLO_LIBRARIES})
target_link_libraries(kmatrixmcmc_library PRIVATE ${ROOT_LIBRARIES})
target_link_libraries(kmatrixmcmc_library PRIVATE ${HDF5_CXX_LIBRARIES} hdf5)
