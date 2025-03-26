#pragma once

#include <ruby.h>
#include "mlx/array.h"

// Convert Ruby objects to MLX arrays
mlx::core::array ruby_to_array(VALUE obj);

// Convert Ruby numeric types to appropriate C++ types
template <typename T>
T ruby_to_scalar(VALUE obj);

// Initialize the convert module
void init_convert(VALUE module); 