cmake_minimum_required(VERSION 3.23)
project(batch_sync_probe LANGUAGES CXX CUDA)
set(CMAKE_VS_GLOBALS "VcpkgEnabled=false")
add_executable(batch_sync_probe main.cu)
set_target_properties(batch_sync_probe PROPERTIES CUDA_STANDARD 17 CUDA_ARCHITECTURES 86
    CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
