#include <ruby.h>
#include <vector>
#include "mlx/fft.h"

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

// FFT module methods
static VALUE fft_fft(VALUE self, VALUE arr, VALUE n, VALUE axis) {
  mx::array& a = get_array(arr);
  int ax = NIL_P(axis) ? -1 : NUM2INT(axis);
  
  mx::array result;
  if (NIL_P(n)) {
    result = mx::fft::fft(a, std::nullopt, ax);
  } else {
    result = mx::fft::fft(a, NUM2INT(n), ax);
  }
  
  return wrap_array(result);
}

static VALUE fft_ifft(VALUE self, VALUE arr, VALUE n, VALUE axis) {
  mx::array& a = get_array(arr);
  int ax = NIL_P(axis) ? -1 : NUM2INT(axis);
  
  mx::array result;
  if (NIL_P(n)) {
    result = mx::fft::ifft(a, std::nullopt, ax);
  } else {
    result = mx::fft::ifft(a, NUM2INT(n), ax);
  }
  
  return wrap_array(result);
}

static VALUE fft_fft2(VALUE self, VALUE arr, VALUE s, VALUE axes) {
  mx::array& a = get_array(arr);
  
  mx::array result;
  if (NIL_P(s) && NIL_P(axes)) {
    result = mx::fft::fft2(a);
  } else if (!NIL_P(s) && NIL_P(axes)) {
    std::vector<int> shape = ruby_array_to_vector(s);
    result = mx::fft::fft2(a, shape);
  } else if (NIL_P(s) && !NIL_P(axes)) {
    std::vector<int> ax = ruby_array_to_vector(axes);
    result = mx::fft::fft2(a, std::nullopt, ax);
  } else {
    std::vector<int> shape = ruby_array_to_vector(s);
    std::vector<int> ax = ruby_array_to_vector(axes);
    result = mx::fft::fft2(a, shape, ax);
  }
  
  return wrap_array(result);
}

static VALUE fft_ifft2(VALUE self, VALUE arr, VALUE s, VALUE axes) {
  mx::array& a = get_array(arr);
  
  mx::array result;
  if (NIL_P(s) && NIL_P(axes)) {
    result = mx::fft::ifft2(a);
  } else if (!NIL_P(s) && NIL_P(axes)) {
    std::vector<int> shape = ruby_array_to_vector(s);
    result = mx::fft::ifft2(a, shape);
  } else if (NIL_P(s) && !NIL_P(axes)) {
    std::vector<int> ax = ruby_array_to_vector(axes);
    result = mx::fft::ifft2(a, std::nullopt, ax);
  } else {
    std::vector<int> shape = ruby_array_to_vector(s);
    std::vector<int> ax = ruby_array_to_vector(axes);
    result = mx::fft::ifft2(a, shape, ax);
  }
  
  return wrap_array(result);
}

static VALUE fft_fftn(VALUE self, VALUE arr, VALUE s, VALUE axes) {
  mx::array& a = get_array(arr);
  
  mx::array result;
  if (NIL_P(s) && NIL_P(axes)) {
    result = mx::fft::fftn(a);
  } else if (!NIL_P(s) && NIL_P(axes)) {
    std::vector<int> shape = ruby_array_to_vector(s);
    result = mx::fft::fftn(a, shape);
  } else if (NIL_P(s) && !NIL_P(axes)) {
    std::vector<int> ax = ruby_array_to_vector(axes);
    result = mx::fft::fftn(a, std::nullopt, ax);
  } else {
    std::vector<int> shape = ruby_array_to_vector(s);
    std::vector<int> ax = ruby_array_to_vector(axes);
    result = mx::fft::fftn(a, shape, ax);
  }
  
  return wrap_array(result);
}

static VALUE fft_ifftn(VALUE self, VALUE arr, VALUE s, VALUE axes) {
  mx::array& a = get_array(arr);
  
  mx::array result;
  if (NIL_P(s) && NIL_P(axes)) {
    result = mx::fft::ifftn(a);
  } else if (!NIL_P(s) && NIL_P(axes)) {
    std::vector<int> shape = ruby_array_to_vector(s);
    result = mx::fft::ifftn(a, shape);
  } else if (NIL_P(s) && !NIL_P(axes)) {
    std::vector<int> ax = ruby_array_to_vector(axes);
    result = mx::fft::ifftn(a, std::nullopt, ax);
  } else {
    std::vector<int> shape = ruby_array_to_vector(s);
    std::vector<int> ax = ruby_array_to_vector(axes);
    result = mx::fft::ifftn(a, shape, ax);
  }
  
  return wrap_array(result);
}

// Initialize FFT module
void init_fft(VALUE module) {
  // Define module functions
  rb_define_module_function(module, "fft", RUBY_METHOD_FUNC(fft_fft), 3);
  rb_define_module_function(module, "ifft", RUBY_METHOD_FUNC(fft_ifft), 3);
  rb_define_module_function(module, "fft2", RUBY_METHOD_FUNC(fft_fft2), 3);
  rb_define_module_function(module, "ifft2", RUBY_METHOD_FUNC(fft_ifft2), 3);
  rb_define_module_function(module, "fftn", RUBY_METHOD_FUNC(fft_fftn), 3);
  rb_define_module_function(module, "ifftn", RUBY_METHOD_FUNC(fft_ifftn), 3);
} 