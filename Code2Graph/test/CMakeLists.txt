INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})
add_executable(testAccuracy testAccuracy.cpp)
target_link_libraries(testAccuracy ${DEPENDENCIES} ChladniCode)
