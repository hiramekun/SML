cmake_minimum_required(VERSION 3.12)
project(SML)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DMY_DEBUG")

add_executable(metropolis MCMC/metropolis.cpp)
add_executable(metropolis2 MCMC/metropolis2.cpp)
add_executable(gaussian_hmc MCMC/gaussian_hmc.cpp)
add_executable(gaussian_hmc_multiple_val MCMC/gaussian_hmc_multiple_val.cpp)
add_executable(gibbs_sampling MCMC/gibbs_sampling_three.cpp)