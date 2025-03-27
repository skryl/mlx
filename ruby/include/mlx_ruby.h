#ifndef MLX_RUBY_H
#define MLX_RUBY_H

#include <ruby.h>
#include <mlx/array.h>
#include <mlx/ops.h>

// Convert Ruby object to MLX array
mlx::core::array get_array(VALUE obj);

// Check if object responds to to_mlx_array
bool responds_to_to_mlx_array(VALUE obj);

// Convert using to_mlx_array
mlx::core::array convert_using_to_mlx_array(VALUE obj);

// Wrap MLX array in Ruby object
VALUE wrap_array(const mlx::core::array& arr);

#endif // MLX_RUBY_H 