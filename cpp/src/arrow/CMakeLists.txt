# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

add_custom_target(arrow-all)
add_custom_target(arrow)
add_custom_target(arrow-benchmarks)
add_custom_target(arrow-tests)
add_custom_target(arrow-integration)
add_dependencies(arrow-all
                 arrow
                 arrow-tests
                 arrow-benchmarks
                 arrow-integration)

# Adding unit tests part of the "arrow" portion of the test suite
function(ADD_ARROW_TEST REL_TEST_NAME)
  set(options)
  set(one_value_args PREFIX)
  set(multi_value_args LABELS)
  cmake_parse_arguments(ARG
                        "${options}"
                        "${one_value_args}"
                        "${multi_value_args}"
                        ${ARGN})

  if(ARG_PREFIX)
    set(PREFIX ${ARG_PREFIX})
  else()
    set(PREFIX "arrow")
  endif()

  if(ARG_LABELS)
    set(LABELS ${ARG_LABELS})
  else()
    set(LABELS "arrow-tests")
  endif()

  add_test_case(${REL_TEST_NAME}
                PREFIX
                ${PREFIX}
                LABELS
                ${LABELS}
                ${ARG_UNPARSED_ARGUMENTS})
endfunction()

function(ADD_ARROW_BENCHMARK REL_TEST_NAME)
  set(options)
  set(one_value_args PREFIX)
  set(multi_value_args)
  cmake_parse_arguments(ARG
                        "${options}"
                        "${one_value_args}"
                        "${multi_value_args}"
                        ${ARGN})
  if(ARG_PREFIX)
    set(PREFIX ${ARG_PREFIX})
  else()
    set(PREFIX "arrow")
  endif()
  add_benchmark(${REL_TEST_NAME}
                PREFIX
                ${PREFIX}
                LABELS
                "arrow-benchmarks"
                ${ARG_UNPARSED_ARGUMENTS})
endfunction()

set(ARROW_SRCS
    array.cc
    builder.cc
    array/builder_adaptive.cc
    array/builder_base.cc
    array/builder_binary.cc
    array/builder_decimal.cc
    array/builder_dict.cc
    array/builder_nested.cc
    array/builder_primitive.cc
    array/builder_union.cc
    array/concatenate.cc
    array/diff.cc
    buffer.cc
    compare.cc
    extension_type.cc
    memory_pool.cc
    pretty_print.cc
    record_batch.cc
    result.cc
    scalar.cc
    sparse_tensor.cc
    status.cc
    table.cc
    table_builder.cc
    tensor.cc
    type.cc
    visitor.cc
    csv/converter.cc
    csv/chunker.cc
    csv/column_builder.cc
    csv/options.cc
    csv/parser.cc
    csv/reader.cc
    filesystem/filesystem.cc
    filesystem/localfs.cc
    filesystem/mockfs.cc
    filesystem/path_util.cc
    filesystem/util_internal.cc
    io/buffered.cc
    io/compressed.cc
    io/file.cc
    io/hdfs.cc
    io/hdfs_internal.cc
    io/interfaces.cc
    io/memory.cc
    io/readahead.cc
    testing/util.cc
    util/basic_decimal.cc
    util/bit_util.cc
    util/compression.cc
    util/cpu_info.cc
    util/decimal.cc
    util/int_util.cc
    util/io_util.cc
    util/logging.cc
    util/key_value_metadata.cc
    util/memory.cc
    util/string.cc
    util/string_builder.cc
    util/task_group.cc
    util/thread_pool.cc
    util/trie.cc
    util/utf8.cc
    vendored/datetime/tz.cpp)

if(ARROW_JEMALLOC)
  if("${CMAKE_GENERATOR}" STREQUAL "Ninja")
    set_source_files_properties(memory_pool.cc PROPERTIES OBJECT_DEPENDS jemalloc_ep)
  else()
    add_dependencies(arrow_dependencies jemalloc_ep)
  endif()
endif()

if(ARROW_JSON)
  add_subdirectory(json)
  set(ARROW_SRCS
      ${ARROW_SRCS}
      json/options.cc
      json/chunked_builder.cc
      json/chunker.cc
      json/converter.cc
      json/parser.cc
      json/reader.cc)
endif()

if(ARROW_S3)
  set(ARROW_SRCS ${ARROW_SRCS} filesystem/s3fs.cc)
endif()

if(ARROW_WITH_URIPARSER)
  set(ARROW_SRCS ${ARROW_SRCS} util/uri.cc)
endif()

if("${COMPILER_FAMILY}" STREQUAL "clang")
  set_property(SOURCE util/io_util.cc
               APPEND_STRING
               PROPERTY COMPILE_FLAGS " -Wno-unused-macros ")
endif()

# Unconditionally install testing headers that are also useful for Arrow consumers.
add_subdirectory(testing)

if(ARROW_COMPUTE)
  add_subdirectory(compute)
  set(ARROW_SRCS
      ${ARROW_SRCS}
      compute/context.cc
      compute/expression.cc
      compute/logical_type.cc
      compute/operation.cc
      compute/kernels/aggregate.cc
      compute/kernels/boolean.cc
      compute/kernels/cast.cc
      compute/kernels/compare.cc
      compute/kernels/count.cc
      compute/kernels/hash.cc
      compute/kernels/filter.cc
      compute/kernels/mean.cc
      compute/kernels/sort_to_indices.cc
      compute/kernels/sum.cc
      compute/kernels/take.cc
      compute/kernels/isin.cc
      compute/kernels/util_internal.cc
      compute/operations/cast.cc
      compute/operations/literal.cc)
endif()

if(ARROW_CUDA)
  # IPC extensions required to build the CUDA library
  set(ARROW_IPC ON)
  add_subdirectory(gpu)
endif()

if(ARROW_WITH_BROTLI)
  add_definitions(-DARROW_WITH_BROTLI)
  set(ARROW_SRCS util/compression_brotli.cc ${ARROW_SRCS})
endif()

if(ARROW_WITH_BZ2)
  add_definitions(-DARROW_WITH_BZ2)
  set(ARROW_SRCS util/compression_bz2.cc ${ARROW_SRCS})
endif()

if(ARROW_WITH_LZ4)
  add_definitions(-DARROW_WITH_LZ4)
  set(ARROW_SRCS util/compression_lz4.cc ${ARROW_SRCS})
endif()

if(ARROW_WITH_SNAPPY)
  add_definitions(-DARROW_WITH_SNAPPY)
  set(ARROW_SRCS util/compression_snappy.cc ${ARROW_SRCS})
endif()

if(ARROW_WITH_ZLIB)
  add_definitions(-DARROW_WITH_ZLIB)
  set(ARROW_SRCS util/compression_zlib.cc ${ARROW_SRCS})
endif()

if(ARROW_WITH_ZSTD)
  add_definitions(-DARROW_WITH_ZSTD)
  set(ARROW_SRCS util/compression_zstd.cc ${ARROW_SRCS})
endif()

if(ARROW_ORC)
  add_subdirectory(adapters/orc)
  set(ARROW_SRCS adapters/orc/adapter.cc adapters/orc/adapter_util.cc ${ARROW_SRCS})
endif()

if(ARROW_TENSORFLOW)
  add_subdirectory(adapters/tensorflow)
endif()

if(ARROW_IPC)
  add_subdirectory(ipc)

  set(ARROW_IPC_SRCS
      ipc/dictionary.cc
      ipc/feather.cc
      ipc/json_integration.cc
      ipc/json_internal.cc
      ipc/json_simple.cc
      ipc/message.cc
      ipc/metadata_internal.cc
      ipc/options.cc
      ipc/reader.cc
      ipc/writer.cc)
  set(ARROW_SRCS ${ARROW_SRCS} ${ARROW_IPC_SRCS})

  add_dependencies(arrow_dependencies metadata_fbs)
endif()

if(NOT APPLE AND NOT MSVC)
  # Localize thirdparty symbols using a linker version script. This hides them
  # from the client application. The OS X linker does not support the
  # version-script option.
  set(ARROW_VERSION_SCRIPT_FLAGS
      "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/symbols.map")
  set(ARROW_SHARED_LINK_FLAGS ${ARROW_VERSION_SCRIPT_FLAGS})
endif()

set(ARROW_ALL_SRCS ${ARROW_SRCS})

add_arrow_lib(arrow
              SOURCES
              ${ARROW_ALL_SRCS}
              OUTPUTS
              ARROW_LIBRARIES
              DEPENDENCIES
              arrow_dependencies
              SHARED_LINK_FLAGS
              ${ARROW_SHARED_LINK_FLAGS}
              SHARED_LINK_LIBS
              ${ARROW_LINK_LIBS}
              SHARED_PRIVATE_LINK_LIBS
              ${ARROW_SHARED_PRIVATE_LINK_LIBS}
              STATIC_LINK_LIBS
              ${ARROW_STATIC_LINK_LIBS}
              SHARED_INSTALL_INTERFACE_LIBS
              ${ARROW_SHARED_INSTALL_INTERFACE_LIBS}
              STATIC_INSTALL_INTERFACE_LIBS
              ${ARROW_STATIC_INSTALL_INTERFACE_LIBS})

add_dependencies(arrow ${ARROW_LIBRARIES})

if(ARROW_BUILD_STATIC AND WIN32)
  target_compile_definitions(arrow_static PUBLIC ARROW_STATIC)
endif()

if(ARROW_BUILD_TESTS OR ARROW_BUILD_BENCHMARKS OR ARROW_BUILD_INTEGRATION)
  # that depend on gtest
  add_arrow_lib(arrow_testing
                SOURCES
                io/test_common.cc
                ipc/test_common.cc
                filesystem/test_util.cc
                testing/gtest_util.cc
                testing/random.cc
                testing/generator.cc
                OUTPUTS
                ARROW_TESTING_LIBRARIES
                DEPENDENCIES
                arrow_test_dependencies
                SHARED_LINK_LIBS
                arrow_shared
                GTest::GTest
                STATIC_LINK_LIBS
                arrow_static)

  add_custom_target(arrow_testing)
  add_dependencies(arrow_testing ${ARROW_TESTING_LIBRARIES})

  if(ARROW_BUILD_STATIC AND WIN32)
    target_compile_definitions(arrow_testing_static PUBLIC ARROW_STATIC)
  endif()

  set(ARROW_LIBRARIES ${ARROW_LIBRARIES} ${ARROW_TESTING_LIBRARIES})
endif()

if(ARROW_WITH_BACKTRACE)
  find_package(Backtrace)

  foreach(LIB_TARGET ${ARROW_LIBRARIES})
    target_compile_definitions(${LIB_TARGET} PRIVATE ARROW_EXPORTING)
    if(Backtrace_FOUND AND ARROW_WITH_BACKTRACE)
      target_compile_definitions(${LIB_TARGET} PRIVATE ARROW_WITH_BACKTRACE)
    endif()
  endforeach()
endif()

# Headers: top level
arrow_install_all_headers("arrow")

# pkg-config support
arrow_add_pkg_config("arrow")

#
# Unit tests
#

add_arrow_test(allocator_test)
add_arrow_test(array_test
               SOURCES
               array_test.cc
               array_binary_test.cc
               array_dict_test.cc
               array_list_test.cc
               array_struct_test.cc
               array_union_test.cc
               array_view_test.cc)
add_arrow_test(buffer_test)

if(ARROW_IPC)
  # The extension type unit tests require IPC / Flatbuffers support
  add_arrow_test(extension_type_test)
endif()

add_arrow_test(memory_pool_test)
add_arrow_test(pretty_print_test)
add_arrow_test(public_api_test)
add_arrow_test(result_test)
add_arrow_test(scalar_test)
add_arrow_test(status_test)
add_arrow_test(stl_test)
add_arrow_test(type_test)
add_arrow_test(table_test)
add_arrow_test(table_builder_test)
add_arrow_test(tensor_test)
add_arrow_test(sparse_tensor_test)

add_arrow_benchmark(builder_benchmark)
add_arrow_benchmark(type_benchmark)

add_subdirectory(array)
add_subdirectory(csv)
add_subdirectory(filesystem)
add_subdirectory(io)
add_subdirectory(util)
add_subdirectory(vendored)

if(ARROW_DATASET)
  add_subdirectory(dataset)
endif()

if(ARROW_FLIGHT)
  add_subdirectory(flight)
endif()

if(ARROW_PYTHON)
  add_subdirectory(python)
endif()

if(ARROW_HIVESERVER2)
  add_subdirectory(dbi/hiveserver2)
endif()
