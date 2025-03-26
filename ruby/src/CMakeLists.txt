# Find Ruby development files
find_package(Ruby REQUIRED)
include_directories(${RUBY_INCLUDE_DIRS})

# Add the Ruby extension module
add_library(core SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/mlx.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/array.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/stream.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/metal.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/memory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mlx_func.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/transforms.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/random.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linalg.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/constants.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fft.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/trees.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/indexing.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/convert.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fast.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/distributed.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/export.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/load.cpp
)

# Set the output name to match Ruby's naming convention for extensions
set_target_properties(
  core
  PROPERTIES 
  PREFIX ""
  OUTPUT_NAME "core"
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../mlx/ext
)

# Link against MLX and Ruby
target_link_libraries(core PRIVATE mlx ${RUBY_LIBRARY})

# Set version definition
target_compile_definitions(core PRIVATE _VERSION_=${MLX_VERSION})

# If MLX is built as shared libraries, set the rpath appropriately
if(BUILD_SHARED_LIBS)
  if(APPLE)
    target_link_options(core PRIVATE -Wl,-rpath,@loader_path/lib)
  else()
    target_link_options(core PRIVATE -Wl,-rpath,$ORIGIN/lib)
  endif()
endif() 