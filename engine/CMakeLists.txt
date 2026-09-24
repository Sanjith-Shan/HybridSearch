cmake_minimum_required(VERSION 3.20)
project(hybridsearch_engine VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(HS_BUILD_TESTS "Build the test suite" ON)
option(HS_BUILD_BENCH "Build microbenchmarks" ON)
option(HS_BUILD_SERVER "Build the gRPC shard server (needs gRPC + protobuf)" ON)
option(HS_NATIVE "Compile for the host architecture" ON)
option(HS_SANITIZE "Build with ASan + UBSan" OFF)

# -O3 plus native tuning. Every latency number records which flags built it.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
if(HS_NATIVE)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mcpu=native" HAVE_MCPU_NATIVE)
  check_cxx_compiler_flag("-march=native" HAVE_MARCH_NATIVE)
  if(HAVE_MCPU_NATIVE)
    add_compile_options($<$<CONFIG:Release>:-mcpu=native>)
  elseif(HAVE_MARCH_NATIVE)
    add_compile_options($<$<CONFIG:Release>:-march=native>)
  endif()
endif()
add_compile_options(-Wall -Wextra)
if(HS_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

find_package(Threads REQUIRED)

# One library per module. Sources are globbed so module owners never need to
# edit this file to add a translation unit.
file(GLOB_RECURSE HS_COMMON_SOURCES CONFIGURE_DEPENDS src/common/*.cc)
file(GLOB_RECURSE HS_LEXICAL_SOURCES CONFIGURE_DEPENDS src/lexical/*.cc)
file(GLOB_RECURSE HS_VECTOR_SOURCES CONFIGURE_DEPENDS src/vector/*.cc)

add_library(hs_common ${HS_COMMON_SOURCES})
target_include_directories(hs_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(hs_common PUBLIC Threads::Threads)

add_library(hs_lexical ${HS_LEXICAL_SOURCES} src/lexical/_anchor.cc)
target_link_libraries(hs_lexical PUBLIC hs_common)

add_library(hs_vector ${HS_VECTOR_SOURCES} src/vector/_anchor.cc)
target_link_libraries(hs_vector PUBLIC hs_common)

# Command-line tools: every tools/*.cc becomes an executable of the same name.
file(GLOB HS_TOOL_SOURCES CONFIGURE_DEPENDS tools/*.cc)
foreach(tool_src ${HS_TOOL_SOURCES})
  get_filename_component(tool_name ${tool_src} NAME_WE)
  add_executable(${tool_name} ${tool_src})
  target_link_libraries(${tool_name} PRIVATE hs_lexical hs_vector)
endforeach()

if(HS_BUILD_TESTS)
  include(FetchContent)
  # Always vendored: a system GoogleTest built against a different libc++ aborts
  # during test discovery on this machine.
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  include(GoogleTest)
  foreach(module common lexical vector)
    file(GLOB_RECURSE test_sources CONFIGURE_DEPENDS tests/${module}/*.cc)
    if(test_sources)
      add_executable(test_${module} ${test_sources})
      target_link_libraries(test_${module} PRIVATE hs_lexical hs_vector GTest::gtest_main)
      target_compile_definitions(test_${module} PRIVATE
        HS_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
      gtest_discover_tests(test_${module} DISCOVERY_TIMEOUT 60)
    endif()
  endforeach()
endif()

if(HS_BUILD_BENCH)
  file(GLOB HS_BENCH_SOURCES CONFIGURE_DEPENDS bench/*.cc)
  foreach(bench_src ${HS_BENCH_SOURCES})
    get_filename_component(bench_name ${bench_src} NAME_WE)
    add_executable(${bench_name} ${bench_src})
    target_link_libraries(${bench_name} PRIVATE hs_lexical hs_vector)
  endforeach()
endif()

# The shard server lives in its own file so the core builds without gRPC.
if(HS_BUILD_SERVER)
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/server.cmake OPTIONAL)
endif()
