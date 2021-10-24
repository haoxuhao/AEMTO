cmake_minimum_required(VERSION 3.2)
project(AEMTO)

OPTION(DEBUG "Option Debug" OFF)
IF(DEBUG)
    ADD_DEFINITIONS(-D DEBUG)
ENDIF(DEBUG)

OPTION(DINSIGHT "Option DINSIGHT" OFF)
IF(DINSIGHT)
    ADD_DEFINITIONS(-D DEEPINSIGHT)
ENDIF(DINSIGHT)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

if(CMAKE_COMPILER_IS_GNUCXX)
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}") # c++11 required.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "-O2 ${CMAKE_CXX_FLAGS}") # compiler optimization.
endif(CMAKE_COMPILER_IS_GNUCXX)

find_package(Threads)
INCLUDE_DIRECTORIES(oss)
INCLUDE_DIRECTORIES(common)
AUX_SOURCE_DIRECTORY(common DIR_SRCS)
AUX_SOURCE_DIRECTORY(oss DIR_SRCS)

SET(CMAKE_CXX_COMPILER g++)

set(exec_name AEMTO)
set(main_file "main/aemto.cc")
message(STATUS "build ${exec_name}")
ADD_EXECUTABLE(${exec_name} ${main_file} ${DIR_SRCS})
TARGET_LINK_LIBRARIES(${exec_name} ${CMAKE_THREAD_LIBS_INIT})

set(exec_name SBO)
set(main_file "main/sbo.cc")
message(STATUS "build ${exec_name}")
ADD_EXECUTABLE(${exec_name} ${main_file} ${DIR_SRCS})
TARGET_LINK_LIBRARIES(${exec_name} ${CMAKE_THREAD_LIBS_INIT})

set(exec_name MATDE)
set(main_file "main/matde.cc")
message(STATUS "build ${exec_name}")
ADD_EXECUTABLE(${exec_name} ${main_file} ${DIR_SRCS})
TARGET_LINK_LIBRARIES(${exec_name} ${CMAKE_THREAD_LIBS_INIT})

set(exec_name MFEA2)
set(main_file "main/mfea.cc")
message(STATUS "build ${exec_name}")
ADD_EXECUTABLE(${exec_name} ${main_file} ${DIR_SRCS})
TARGET_LINK_LIBRARIES(${exec_name} ${CMAKE_THREAD_LIBS_INIT})

# set(exec_name unittest)
# set(main_file "main/unittest.cc")
# message(STATUS "build ${exec_name}")
# ADD_EXECUTABLE(${exec_name} ${main_file} ${DIR_SRCS})
# TARGET_LINK_LIBRARIES(${exec_name} ${CMAKE_THREAD_LIBS_INIT})
