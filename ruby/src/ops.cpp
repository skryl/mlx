#include <ruby.h>
#include <vector>
#include "mlx/ops.h"
#include "mlx_func.h"

namespace mx = mlx::core;

// Helper function to extract mx::array from Ruby VALUE
static mx::array& get_array(VALUE obj) {
  mx::array* arr_ptr;
  Data_Get_Struct(obj, mx::array, arr_ptr);
  return *arr_ptr;
}

// Helper function to wrap mx::array into Ruby VALUE
static VALUE wrap_array(const mx::array& arr) {
  return Data_Wrap_Struct(rb_path2class("MLX::Core::Array"), 0, nullptr, new mx::array(arr));
}

// Array creation operations
static VALUE ops_zeros(VALUE self, VALUE shape, VALUE dtype) {
  Check_Type(shape, T_ARRAY);
  
  // Convert Ruby array to std::vector<int>
  std::vector<int> cpp_shape;
  for (long i = 0; i < RARRAY_LEN(shape); i++) {
    VALUE item = rb_ary_entry(shape, i);
    cpp_shape.push_back(NUM2INT(item));
  }
  
  // Create zeros array
  mx::Dtype d = static_cast<mx::Dtype::Val>(NUM2INT(dtype));
  mx::array result = mx::zeros(cpp_shape, d);
  
  return wrap_array(result);
}

static VALUE ops_ones(VALUE self, VALUE shape, VALUE dtype) {
  Check_Type(shape, T_ARRAY);
  
  // Convert Ruby array to std::vector<int>
  std::vector<int> cpp_shape;
  for (long i = 0; i < RARRAY_LEN(shape); i++) {
    VALUE item = rb_ary_entry(shape, i);
    cpp_shape.push_back(NUM2INT(item));
  }
  
  // Create ones array
  mx::Dtype d = static_cast<mx::Dtype::Val>(NUM2INT(dtype));
  mx::array result = mx::ones(cpp_shape, d);
  
  return wrap_array(result);
}

static VALUE ops_full(VALUE self, VALUE shape, VALUE fill_value, VALUE dtype) {
  Check_Type(shape, T_ARRAY);
  
  // Convert Ruby array to std::vector<int>
  std::vector<int> cpp_shape;
  for (long i = 0; i < RARRAY_LEN(shape); i++) {
    VALUE item = rb_ary_entry(shape, i);
    cpp_shape.push_back(NUM2INT(item));
  }
  
  // Get the fill value
  ScalarOrArray fill = value_to_scalar_or_array(fill_value);
  
  // Create full array
  mx::Dtype d = static_cast<mx::Dtype::Val>(NUM2INT(dtype));
  mx::array result;
  
  if (fill.is_scalar()) {
    result = mx::full(cpp_shape, fill.scalar, d);
  } else {
    result = mx::full(cpp_shape, fill.array, d);
  }
  
  return wrap_array(result);
}

// Basic operations
static VALUE ops_add(VALUE self, VALUE a, VALUE b) {
  ScalarOrArray arg_a = value_to_scalar_or_array(a);
  ScalarOrArray arg_b = value_to_scalar_or_array(b);
  
  mx::array result;
  
  if (arg_a.is_scalar() && arg_b.is_scalar()) {
    result = mx::array(arg_a.scalar + arg_b.scalar);
  } else if (arg_a.is_scalar() && arg_b.is_array()) {
    result = arg_a.scalar + arg_b.array;
  } else if (arg_a.is_array() && arg_b.is_scalar()) {
    result = arg_a.array + arg_b.scalar;
  } else {
    result = arg_a.array + arg_b.array;
  }
  
  return wrap_array(result);
}

static VALUE ops_subtract(VALUE self, VALUE a, VALUE b) {
  ScalarOrArray arg_a = value_to_scalar_or_array(a);
  ScalarOrArray arg_b = value_to_scalar_or_array(b);
  
  mx::array result;
  
  if (arg_a.is_scalar() && arg_b.is_scalar()) {
    result = mx::array(arg_a.scalar - arg_b.scalar);
  } else if (arg_a.is_scalar() && arg_b.is_array()) {
    result = arg_a.scalar - arg_b.array;
  } else if (arg_a.is_array() && arg_b.is_scalar()) {
    result = arg_a.array - arg_b.scalar;
  } else {
    result = arg_a.array - arg_b.array;
  }
  
  return wrap_array(result);
}

static VALUE ops_multiply(VALUE self, VALUE a, VALUE b) {
  ScalarOrArray arg_a = value_to_scalar_or_array(a);
  ScalarOrArray arg_b = value_to_scalar_or_array(b);
  
  mx::array result;
  
  if (arg_a.is_scalar() && arg_b.is_scalar()) {
    result = mx::array(arg_a.scalar * arg_b.scalar);
  } else if (arg_a.is_scalar() && arg_b.is_array()) {
    result = arg_a.scalar * arg_b.array;
  } else if (arg_a.is_array() && arg_b.is_scalar()) {
    result = arg_a.array * arg_b.scalar;
  } else {
    result = arg_a.array * arg_b.array;
  }
  
  return wrap_array(result);
}

static VALUE ops_divide(VALUE self, VALUE a, VALUE b) {
  ScalarOrArray arg_a = value_to_scalar_or_array(a);
  ScalarOrArray arg_b = value_to_scalar_or_array(b);
  
  mx::array result;
  
  if (arg_a.is_scalar() && arg_b.is_scalar()) {
    result = mx::array(arg_a.scalar / arg_b.scalar);
  } else if (arg_a.is_scalar() && arg_b.is_array()) {
    result = arg_a.scalar / arg_b.array;
  } else if (arg_a.is_array() && arg_b.is_scalar()) {
    result = arg_a.array / arg_b.scalar;
  } else {
    result = arg_a.array / arg_b.array;
  }
  
  return wrap_array(result);
}

// Initialize ops module
void init_ops(VALUE module) {
  // Array creation
  rb_define_module_function(module, "zeros", RUBY_METHOD_FUNC(ops_zeros), 2);
  rb_define_module_function(module, "ones", RUBY_METHOD_FUNC(ops_ones), 2);
  rb_define_module_function(module, "full", RUBY_METHOD_FUNC(ops_full), 3);
  
  // Basic operations
  rb_define_module_function(module, "add", RUBY_METHOD_FUNC(ops_add), 2);
  rb_define_module_function(module, "subtract", RUBY_METHOD_FUNC(ops_subtract), 2);
  rb_define_module_function(module, "multiply", RUBY_METHOD_FUNC(ops_multiply), 2);
  rb_define_module_function(module, "divide", RUBY_METHOD_FUNC(ops_divide), 2);
  
  // Add many other ops as needed - this is just a starter set
} 