# gRPC shard server. Kept out of the core build so the engine and its tests
# compile on a machine without gRPC.
#
# On macOS, Homebrew's gRPC must win over any Anaconda copy on PATH: pass
# -DCMAKE_PREFIX_PATH=/opt/homebrew (scripts/build_engine.sh does).

find_package(Protobuf CONFIG QUIET)
find_package(gRPC CONFIG QUIET)
if(NOT gRPC_FOUND OR NOT Protobuf_FOUND)
  message(STATUS "hs: gRPC/protobuf not found, shard server not built")
  return()
endif()
message(STATUS "hs: gRPC ${gRPC_VERSION}, protobuf ${Protobuf_VERSION}")

set(HS_PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../proto)
set(HS_PROTO ${HS_PROTO_DIR}/hybridsearch/v1/shard.proto)
set(HS_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/gen)
file(MAKE_DIRECTORY ${HS_GEN_DIR})

set(HS_PROTO_SRCS ${HS_GEN_DIR}/hybridsearch/v1/shard.pb.cc ${HS_GEN_DIR}/hybridsearch/v1/shard.grpc.pb.cc)
set(HS_PROTO_HDRS ${HS_GEN_DIR}/hybridsearch/v1/shard.pb.h ${HS_GEN_DIR}/hybridsearch/v1/shard.grpc.pb.h)
add_custom_command(
  OUTPUT ${HS_PROTO_SRCS} ${HS_PROTO_HDRS}
  COMMAND $<TARGET_FILE:protobuf::protoc>
    --proto_path=${HS_PROTO_DIR}
    --cpp_out=${HS_GEN_DIR}
    --grpc_out=${HS_GEN_DIR}
    --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
    ${HS_PROTO}
  DEPENDS ${HS_PROTO}
  COMMENT "protoc shard.proto")

add_library(hs_proto ${HS_PROTO_SRCS})
target_include_directories(hs_proto PUBLIC ${HS_GEN_DIR})
target_link_libraries(hs_proto PUBLIC gRPC::grpc++ protobuf::libprotobuf)
target_compile_options(hs_proto PRIVATE -w)

file(GLOB_RECURSE HS_SERVER_SOURCES CONFIGURE_DEPENDS src/server/*.cc)
if(HS_SERVER_SOURCES)
  add_library(hs_server ${HS_SERVER_SOURCES})
  target_link_libraries(hs_server PUBLIC hs_lexical hs_vector hs_proto)
  file(GLOB HS_SERVER_TOOLS CONFIGURE_DEPENDS tools/server/*.cc)
  foreach(tool_src ${HS_SERVER_TOOLS})
    get_filename_component(tool_name ${tool_src} NAME_WE)
    add_executable(${tool_name} ${tool_src})
    target_link_libraries(${tool_name} PRIVATE hs_server gRPC::grpc++_reflection)
  endforeach()
  if(HS_BUILD_TESTS)
    file(GLOB_RECURSE HS_SERVER_TESTS CONFIGURE_DEPENDS tests/server/*.cc)
    if(HS_SERVER_TESTS)
      add_executable(test_server ${HS_SERVER_TESTS})
      target_link_libraries(test_server PRIVATE hs_server GTest::gtest_main)
      gtest_discover_tests(test_server DISCOVERY_TIMEOUT 60)
    endif()
  endif()
endif()
