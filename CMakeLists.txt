cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dcgan)

set(CMAKE_PREFIX_PATH "C:\\Users\\User\\libtorch\\debug")
cmake_policy(SET CMP0054 NEW)
find_package(Torch REQUIRED)

add_executable(dcgan dcgan.cpp)
target_link_libraries(dcgan "${TORCH_LIBRARIES}")
set_property(TARGET dcgan PROPERTY CXX_STANDARD 14)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET dcgan
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:dcgan>)
endif (MSVC)