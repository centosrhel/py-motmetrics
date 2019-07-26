cmake_minimum_required(VERSION 2.8)
project(Code2Graph)

#add_definitions(-DDEBUG)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11")
#set(OpenCV_DIR /home/sixd-ailabs/Develop/Libs/opencv-3.3.1/build_no_cuda/install/share/OpenCV)
find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
set(DEPENDENCIES m pthread ${OpenCV_LIBRARIES})

add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries(${PROJECT_NAME} ${DEPENDENCIES})


add_executable(RenameTool renametools.cpp)
target_link_libraries(RenameTool ${DEPENDENCIES})
set(CHLADNI_CODE_SRC_FILE interface.h IQRCode.h QRCode.h interface.cpp image_coder.cpp IQRCode.cpp QRCode.cpp)
add_library(ChladniCode SHARED ${CHLADNI_CODE_SRC_FILE})
target_link_libraries(ChladniCode ${DEPENDENCIES})

add_executable(Code2 main2.cpp)
target_link_libraries(Code2 ${DEPENDENCIES} ChladniCode)

set(SOURCE_FILES testCppInterfaces.cpp)
add_executable(RS_CRC_Code ${SOURCE_FILES})
target_link_libraries(RS_CRC_Code ${DEPENDENCIES} ChladniCode)

ADD_SUBDIRECTORY(test)
