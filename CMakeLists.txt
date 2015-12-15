cmake_minimum_required(VERSION 2.6)
project(ghowl)

option(test "Build tests." OFF)

add_library(${CMAKE_PROJECT_NAME}
  src/ghowl.cpp
)

if (test)
  include_directories(${CMAKE_SOURCE_DIR}/src)
  add_executable(ghowlTest
    test/bmpcolor.cpp
    test/bmpsobel.cpp
    test/heat.cpp
    test/sum.cpp
    test/bmprotate.cpp
    test/gaussian.cpp
    test/main.cpp  
  )
  
  target_link_libraries(ghowlTest ${CMAKE_PROJECT_NAME})
endif()
