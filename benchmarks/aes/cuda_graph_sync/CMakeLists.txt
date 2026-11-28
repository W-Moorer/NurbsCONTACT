cmake_minimum_required(VERSION 3.23)
project(nc_graph_sync_probe LANGUAGES CXX CUDA)
add_executable(nc_graph_sync_probe main.cu)
set_target_properties(nc_graph_sync_probe PROPERTIES CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(nc_graph_sync_probe PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
