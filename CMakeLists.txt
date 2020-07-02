cmake_minimum_required(VERSION 3.12)
project(SML)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DMY_DEBUG")

add_executable(metropolis MCMC/metropolis.cpp)
add_executable(metropolis2 MCMC/metropolis2.cpp)