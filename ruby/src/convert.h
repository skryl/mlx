#pragma once

#include <ruby.h>
#include <optional>
#include "mlx/array.h"

// Convert Ruby objects to MLX arrays
mlx::core::array ruby_to_array(VALUE obj);
mlx::core::array ruby_to_array(VALUE obj, std::optional<mlx::core::Dtype> dtype);

// Convert MLX arrays to Ruby objects
VALUE to_ruby_scalar(const mlx::core::array& arr);
VALUE to_ruby_array(const mlx::core::array& arr);
VALUE array_from_nested_array(VALUE rb_array, std::optional<mlx::core::Dtype> dtype);

// Convert nested Ruby arrays to MLX arrays with shape validation
mlx::core::array nested_array_to_mlx(VALUE rb_array, std::optional<mlx::core::Dtype> dtype);

// Handle external array libraries and DLPack protocol
bool responds_to_to_mlx_array(VALUE obj);
mlx::core::array convert_using_to_mlx_array(VALUE obj);

// Convert Ruby numeric types to appropriate C++ types
template <typename T>
T ruby_to_scalar(VALUE obj);

// Initialize the convert module
void init_convert(VALUE module); 