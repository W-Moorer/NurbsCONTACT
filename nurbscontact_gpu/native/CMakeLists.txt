cmake_minimum_required(VERSION 3.18)
project(nc_gpu_exchange LANGUAGES CXX)
add_library(nc_hinge_exchange SHARED hinge_exchange.cpp)
target_compile_features(nc_hinge_exchange PRIVATE cxx_std_20)
if(SKBUILD)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../../src/cuda_resident_step.cuh"
        DESTINATION nurbscontact_gpu/kernels)
    install(TARGETS nc_hinge_exchange
        RUNTIME DESTINATION nurbscontact_gpu
        LIBRARY DESTINATION nurbscontact_gpu)
endif()
