cmake_minimum_required(VERSION 2.8)
project( main )
find_package( OpenCV REQUIRED )

set(CMAKE_CXX_FLAGS "-std=c++17 -O3 -flto -fopenmp -march=native")
set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=lld")

include_directories(../lbp-library)

file(GLOB_RECURSE lbp ../lbp-library/package_lbp/*.cpp ../lbp-library/package_lbp/*.c)

add_executable( main main.cpp ${lbp} )
target_link_libraries( main ${OpenCV_LIBS} )
