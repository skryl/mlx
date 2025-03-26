#pragma once

#include <ruby.h>
#include "mlx/array.h"

// Function to extract Ruby value or MLX array from a parameter
struct ScalarOrArray {
  enum class Type { Scalar, Array };
  
  Type type;
  union {
    double scalar;
    mlx::core::array array;
  };
  
  ~ScalarOrArray() {
    if (type == Type::Array) {
      array.~array();
    }
  }
  
  ScalarOrArray(double scalar_val) : type(Type::Scalar), scalar(scalar_val) {}
  ScalarOrArray(const mlx::core::array& array_val) : type(Type::Array), array(array_val) {}
  
  bool is_scalar() const { return type == Type::Scalar; }
  bool is_array() const { return type == Type::Array; }
};

// Convert Ruby value to ScalarOrArray
ScalarOrArray value_to_scalar_or_array(VALUE value);

// Initialize the mlx function module
void init_mlx_func(VALUE module); 