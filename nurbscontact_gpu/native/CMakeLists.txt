cmake_minimum_required(VERSION 3.18)
project(nc_gpu_exchange LANGUAGES CXX)
add_library(nc_hinge_exchange SHARED hinge_exchange.cpp)
target_compile_features(nc_hinge_exchange PRIVATE cxx_std_20)
if(SKBUILD)
    install(TARGETS nc_hinge_exchange
        RUNTIME DESTINATION nurbscontact_gpu
        LIBRARY DESTINATION nurbscontact_gpu)
endif()
