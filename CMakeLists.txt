cmake_minimum_required(VERSION 3.20)
project(supercomps C)

set(CMAKE_C_STANDARD 99)

set(MPI_INCLUDE_DIR "C:/Program Files (x86)/Microsoft SDKs/MPI/Include")
set(MPI_LIB_DIR "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64")

add_executable(example src/9/example.c)
target_include_directories(example PRIVATE ${MPI_INCLUDE_DIR})
target_link_directories(example PRIVATE ${MPI_LIB_DIR})
target_link_libraries(example PRIVATE msmpi)

add_executable(9_3 src/9/9_3.c)
target_include_directories(9_3 PRIVATE ${MPI_INCLUDE_DIR})
target_link_directories(9_3 PRIVATE ${MPI_LIB_DIR})
target_link_libraries(9_3 PRIVATE msmpi)

add_executable(9_3_benchmark src/9/9_3_benchmark.c)
target_include_directories(9_3_benchmark PRIVATE ${MPI_INCLUDE_DIR})
target_link_directories(9_3_benchmark PRIVATE ${MPI_LIB_DIR})
target_link_libraries(9_3_benchmark PRIVATE msmpi)

add_executable(9_3_packed src/9/9_3_packed.c)
target_include_directories(9_3_packed PRIVATE ${MPI_INCLUDE_DIR})
target_link_directories(9_3_packed PRIVATE ${MPI_LIB_DIR})
target_link_libraries(9_3_packed PRIVATE msmpi)