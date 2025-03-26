#include <ruby.h>
#include "mlx/transforms.h"
#include "mlx/ops.h"

namespace mx = mlx::core;

// Helper function to wrap mx::array into Ruby VALUE
static VALUE wrap_array(const mx::array& arr) {
  return Data_Wrap_Struct(rb_path2class("MLX::Core::Array"), 0, nullptr, new mx::array(arr));
}

// Helper to extract mx::array from Ruby VALUE
static mx::array& get_array(VALUE obj) {
  mx::array* arr_ptr;
  Data_Get_Struct(obj, mx::array, arr_ptr);
  return *arr_ptr;
}

// Helper to extract Ruby array into C++ vector
static std::vector<int> ruby_array_to_vector(VALUE arr) {
  Check_Type(arr, T_ARRAY);
  
  std::vector<int> cpp_arr;
  for (long i = 0; i < RARRAY_LEN(arr); i++) {
    VALUE item = rb_ary_entry(arr, i);
    cpp_arr.push_back(NUM2INT(item));
  }
  
  return cpp_arr;
}

// Transforms module methods
static VALUE transforms_reshape(VALUE self, VALUE arr, VALUE shape) {
  mx::array& a = get_array(arr);
  std::vector<int> new_shape = ruby_array_to_vector(shape);
  
  mx::array result = mx::reshape(a, new_shape);
  return wrap_array(result);
}

static VALUE transforms_transpose(VALUE self, VALUE arr, VALUE axes) {
  mx::array& a = get_array(arr);
  
  if (NIL_P(axes)) {
    mx::array result = mx::transpose(a);
    return wrap_array(result);
  } else {
    std::vector<int> perm = ruby_array_to_vector(axes);
    mx::array result = mx::transpose(a, perm);
    return wrap_array(result);
  }
}

static VALUE transforms_squeeze(VALUE self, VALUE arr, VALUE axes) {
  mx::array& a = get_array(arr);
  
  if (NIL_P(axes)) {
    mx::array result = mx::squeeze(a);
    return wrap_array(result);
  } else if (RB_TYPE_P(axes, T_FIXNUM)) {
    int axis = NUM2INT(axes);
    mx::array result = mx::squeeze(a, axis);
    return wrap_array(result);
  } else {
    std::vector<int> axes_vec = ruby_array_to_vector(axes);
    mx::array result = mx::squeeze(a, axes_vec);
    return wrap_array(result);
  }
}

static VALUE transforms_expand_dims(VALUE self, VALUE arr, VALUE axis) {
  mx::array& a = get_array(arr);
  int ax = NUM2INT(axis);
  
  mx::array result = mx::expand_dims(a, ax);
  return wrap_array(result);
}

static VALUE transforms_split(VALUE self, VALUE arr, VALUE num_or_indices, VALUE axis) {
  mx::array& a = get_array(arr);
  int ax = NIL_P(axis) ? 0 : NUM2INT(axis);
  
  std::vector<mx::array> result;
  
  if (RB_TYPE_P(num_or_indices, T_FIXNUM)) {
    int num = NUM2INT(num_or_indices);
    result = mx::split(a, num, ax);
  } else {
    std::vector<int> indices = ruby_array_to_vector(num_or_indices);
    result = mx::split(a, indices, ax);
  }
  
  VALUE rb_result = rb_ary_new();
  for (const auto& res : result) {
    rb_ary_push(rb_result, wrap_array(res));
  }
  
  return rb_result;
}

static VALUE transforms_concatenate(VALUE self, VALUE arrays, VALUE axis) {
  Check_Type(arrays, T_ARRAY);
  
  std::vector<mx::array> cpp_arrays;
  for (long i = 0; i < RARRAY_LEN(arrays); i++) {
    VALUE arr = rb_ary_entry(arrays, i);
    cpp_arrays.push_back(get_array(arr));
  }
  
  int ax = NIL_P(axis) ? 0 : NUM2INT(axis);
  
  mx::array result = mx::concatenate(cpp_arrays, ax);
  return wrap_array(result);
}

static VALUE transforms_stack(VALUE self, VALUE arrays, VALUE axis) {
  Check_Type(arrays, T_ARRAY);
  
  std::vector<mx::array> cpp_arrays;
  for (long i = 0; i < RARRAY_LEN(arrays); i++) {
    VALUE arr = rb_ary_entry(arrays, i);
    cpp_arrays.push_back(get_array(arr));
  }
  
  int ax = NIL_P(axis) ? 0 : NUM2INT(axis);
  
  mx::array result = mx::stack(cpp_arrays, ax);
  return wrap_array(result);
}

// Initialize transforms module
void init_transforms(VALUE module) {
  // Define module functions
  rb_define_module_function(module, "reshape", RUBY_METHOD_FUNC(transforms_reshape), 2);
  rb_define_module_function(module, "transpose", RUBY_METHOD_FUNC(transforms_transpose), 2);
  rb_define_module_function(module, "squeeze", RUBY_METHOD_FUNC(transforms_squeeze), 2);
  rb_define_module_function(module, "expand_dims", RUBY_METHOD_FUNC(transforms_expand_dims), 2);
  rb_define_module_function(module, "split", RUBY_METHOD_FUNC(transforms_split), 3);
  rb_define_module_function(module, "concatenate", RUBY_METHOD_FUNC(transforms_concatenate), 2);
  rb_define_module_function(module, "stack", RUBY_METHOD_FUNC(transforms_stack), 2);
} 