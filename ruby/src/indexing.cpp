#include <ruby.h>
#include "mlx/ops.h"

namespace mx = mlx::core;

// Helper to extract mx::array from Ruby VALUE
static mx::array& get_array(VALUE obj) {
  mx::array* arr_ptr;
  Data_Get_Struct(obj, mx::array, arr_ptr);
  return *arr_ptr;
}

// Helper function to wrap mx::array into Ruby VALUE
static VALUE wrap_array(const mx::array& arr) {
  return Data_Wrap_Struct(rb_path2class("MLX::Core::Array"), 0, nullptr, new mx::array(arr));
}

// Indexing module methods
static VALUE indexing_take(VALUE self, VALUE arr, VALUE indices, VALUE axis) {
  mx::array& a = get_array(arr);
  mx::array& idx = get_array(indices);
  int ax = NIL_P(axis) ? 0 : NUM2INT(axis);
  
  mx::array result = mx::take(a, idx, ax);
  return wrap_array(result);
}

static VALUE indexing_take_along_axis(VALUE self, VALUE arr, VALUE indices, VALUE axis) {
  mx::array& a = get_array(arr);
  mx::array& idx = get_array(indices);
  int ax = NIL_P(axis) ? 0 : NUM2INT(axis);
  
  mx::array result = mx::take_along_axis(a, idx, ax);
  return wrap_array(result);
}

static VALUE indexing_slice(VALUE self, VALUE arr, VALUE start, VALUE stop, VALUE step) {
  mx::array& a = get_array(arr);
  
  mx::array result;
  if (NIL_P(start)) {
    // Use None for start
    if (NIL_P(stop)) {
      // Both start and stop are None
      if (NIL_P(step)) {
        // All are None - full slice
        result = mx::slice(a, std::nullopt, std::nullopt, std::nullopt);
      } else {
        // Only step is provided
        result = mx::slice(a, std::nullopt, std::nullopt, NUM2INT(step));
      }
    } else {
      // Start is None, stop is provided
      if (NIL_P(step)) {
        // Step is None
        result = mx::slice(a, std::nullopt, NUM2INT(stop), std::nullopt);
      } else {
        // Step is provided
        result = mx::slice(a, std::nullopt, NUM2INT(stop), NUM2INT(step));
      }
    }
  } else {
    // Start is provided
    if (NIL_P(stop)) {
      // Stop is None
      if (NIL_P(step)) {
        // Step is None
        result = mx::slice(a, NUM2INT(start), std::nullopt, std::nullopt);
      } else {
        // Step is provided
        result = mx::slice(a, NUM2INT(start), std::nullopt, NUM2INT(step));
      }
    } else {
      // Both start and stop are provided
      if (NIL_P(step)) {
        // Step is None
        result = mx::slice(a, NUM2INT(start), NUM2INT(stop), std::nullopt);
      } else {
        // All are provided
        result = mx::slice(a, NUM2INT(start), NUM2INT(stop), NUM2INT(step));
      }
    }
  }
  
  return wrap_array(result);
}

static VALUE indexing_index(VALUE self, VALUE arr, VALUE indices) {
  mx::array& a = get_array(arr);
  
  // Check if indices is an array of arrays
  if (RB_TYPE_P(indices, T_ARRAY)) {
    std::vector<mx::array> idx_arrays;
    for (long i = 0; i < RARRAY_LEN(indices); i++) {
      VALUE item = rb_ary_entry(indices, i);
      mx::array& idx = get_array(item);
      idx_arrays.push_back(idx);
    }
    
    mx::array result = mx::index(a, idx_arrays);
    return wrap_array(result);
  } else {
    // Single index array
    mx::array& idx = get_array(indices);
    std::vector<mx::array> idx_arrays = {idx};
    
    mx::array result = mx::index(a, idx_arrays);
    return wrap_array(result);
  }
}

static VALUE indexing_dynamic_slice(VALUE self, VALUE arr, VALUE start_indices, VALUE slice_sizes) {
  mx::array& a = get_array(arr);
  
  // Check if start_indices is an array of arrays
  if (RB_TYPE_P(start_indices, T_ARRAY)) {
    std::vector<mx::array> start_idx_arrays;
    for (long i = 0; i < RARRAY_LEN(start_indices); i++) {
      VALUE item = rb_ary_entry(start_indices, i);
      mx::array& idx = get_array(item);
      start_idx_arrays.push_back(idx);
    }
    
    // Check slice_sizes
    if (RB_TYPE_P(slice_sizes, T_ARRAY)) {
      std::vector<int> slice_sizes_vec;
      for (long i = 0; i < RARRAY_LEN(slice_sizes); i++) {
        VALUE item = rb_ary_entry(slice_sizes, i);
        slice_sizes_vec.push_back(NUM2INT(item));
      }
      
      mx::array result = mx::dynamic_slice(a, start_idx_arrays, slice_sizes_vec);
      return wrap_array(result);
    } else {
      rb_raise(rb_eArgError, "slice_sizes must be an array");
      return Qnil;
    }
  } else {
    rb_raise(rb_eArgError, "start_indices must be an array");
    return Qnil;
  }
}

// Initialize indexing module
void init_indexing(VALUE module) {
  // Define module functions
  rb_define_module_function(module, "take", RUBY_METHOD_FUNC(indexing_take), 3);
  rb_define_module_function(module, "take_along_axis", RUBY_METHOD_FUNC(indexing_take_along_axis), 3);
  rb_define_module_function(module, "slice", RUBY_METHOD_FUNC(indexing_slice), 4);
  rb_define_module_function(module, "index", RUBY_METHOD_FUNC(indexing_index), 2);
  rb_define_module_function(module, "dynamic_slice", RUBY_METHOD_FUNC(indexing_dynamic_slice), 3);
} 