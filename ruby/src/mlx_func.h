#pragma once

#include <ruby.h>
#include <functional>
#include <vector>
#include "mlx/array.h"

namespace mx = mlx::core;

// A function wrapper that properly handles Ruby GC
// and dependency tracking

// Typedefs for function types
typedef std::function<mx::array(const std::vector<mx::array>&)> RubyMlxFunction;
typedef std::function<std::pair<mx::array, std::vector<mx::array>>(const std::vector<mx::array>&)> RubyMlxFunctionWithAux;

// Create a GC-aware function wrapper
VALUE mlx_func_create(VALUE func, std::vector<VALUE> deps);

// Version with function object directly
template <typename F>
VALUE mlx_func_create(F func, std::vector<VALUE> deps = {}) {
  // Create a wrapper around the C++ function
  // This will be implemented in the .cpp file
  return mlx_func_create(func, deps);
}

// Initialize the mlx function module
void init_mlx_func(VALUE module); 