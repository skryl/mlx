#include <ruby.h>
#include <vector>
#include <complex>
#include "mlx/ops.h"
#include "convert.h"

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

// Template specializations for converting Ruby types to C++ scalars
template<>
bool ruby_to_scalar<bool>(VALUE obj) {
  return RTEST(obj);
}

template<>
uint8_t ruby_to_scalar<uint8_t>(VALUE obj) {
  return NUM2UINT(obj);
}

template<>
uint16_t ruby_to_scalar<uint16_t>(VALUE obj) {
  return NUM2UINT(obj);
}

template<>
uint32_t ruby_to_scalar<uint32_t>(VALUE obj) {
  return NUM2UINT(obj);
}

template<>
uint64_t ruby_to_scalar<uint64_t>(VALUE obj) {
  return NUM2ULL(obj);
}

template<>
int8_t ruby_to_scalar<int8_t>(VALUE obj) {
  return NUM2INT(obj);
}

template<>
int16_t ruby_to_scalar<int16_t>(VALUE obj) {
  return NUM2INT(obj);
}

template<>
int32_t ruby_to_scalar<int32_t>(VALUE obj) {
  return NUM2INT(obj);
}

template<>
int64_t ruby_to_scalar<int64_t>(VALUE obj) {
  return NUM2LL(obj);
}

template<>
float ruby_to_scalar<float>(VALUE obj) {
  return (float)NUM2DBL(obj);
}

template<>
double ruby_to_scalar<double>(VALUE obj) {
  return NUM2DBL(obj);
}

template<>
std::complex<float> ruby_to_scalar<std::complex<float>>(VALUE obj) {
  if (rb_respond_to(obj, rb_intern("real")) && rb_respond_to(obj, rb_intern("imag"))) {
    float real = (float)NUM2DBL(rb_funcall(obj, rb_intern("real"), 0));
    float imag = (float)NUM2DBL(rb_funcall(obj, rb_intern("imag"), 0));
    return std::complex<float>(real, imag);
  } else {
    return std::complex<float>((float)NUM2DBL(obj), 0.0f);
  }
}

// Convert Ruby array to std::vector
template<typename T>
std::vector<T> ruby_array_to_vector(VALUE rb_array) {
  Check_Type(rb_array, T_ARRAY);
  
  std::vector<T> result;
  long size = RARRAY_LEN(rb_array);
  result.reserve(size);
  
  for (long i = 0; i < size; i++) {
    VALUE item = rb_ary_entry(rb_array, i);
    result.push_back(ruby_to_scalar<T>(item));
  }
  
  return result;
}

// Function to convert Ruby objects to MLX arrays
mx::array ruby_to_array(VALUE obj) {
  if (rb_obj_is_kind_of(obj, rb_path2class("MLX::Core::Array"))) {
    // Object is already an MLX array
    return get_array(obj);
  } else if (RB_TYPE_P(obj, T_ARRAY)) {
    // Convert Ruby array to MLX array
    std::vector<double> data;
    for (long i = 0; i < RARRAY_LEN(obj); i++) {
      VALUE item = rb_ary_entry(obj, i);
      data.push_back(NUM2DBL(item));
    }
    return mx::array(data);
  } else if (rb_obj_is_kind_of(obj, rb_cNumeric)) {
    // Convert Ruby numeric to MLX scalar array
    return mx::array(NUM2DBL(obj));
  } else {
    rb_raise(rb_eTypeError, "Cannot convert Ruby object to MLX array");
    return mx::array(); // Dummy return to satisfy compiler
  }
}

// Convert module methods
static VALUE convert_to_float16(VALUE self, VALUE arr) {
  mx::array& a = get_array(arr);
  mx::array result = mx::astype(a, mx::float16);
  return wrap_array(result);
}

static VALUE convert_to_float32(VALUE self, VALUE arr) {
  mx::array& a = get_array(arr);
  mx::array result = mx::astype(a, mx::float32);
  return wrap_array(result);
}

static VALUE convert_to_int32(VALUE self, VALUE arr) {
  mx::array& a = get_array(arr);
  mx::array result = mx::astype(a, mx::int32);
  return wrap_array(result);
}

static VALUE convert_to_bool(VALUE self, VALUE arr) {
  mx::array& a = get_array(arr);
  mx::array result = mx::astype(a, mx::bool_);
  return wrap_array(result);
}

static VALUE convert_to_type(VALUE self, VALUE arr, VALUE dtype) {
  mx::array& a = get_array(arr);
  mx::Dtype d = static_cast<mx::Dtype::Val>(NUM2INT(dtype));
  mx::array result = mx::astype(a, d);
  return wrap_array(result);
}

// Initialize convert module
void init_convert(VALUE module) {
  // Define module functions
  rb_define_module_function(module, "to_float16", RUBY_METHOD_FUNC(convert_to_float16), 1);
  rb_define_module_function(module, "to_float32", RUBY_METHOD_FUNC(convert_to_float32), 1);
  rb_define_module_function(module, "to_int32", RUBY_METHOD_FUNC(convert_to_int32), 1);
  rb_define_module_function(module, "to_bool", RUBY_METHOD_FUNC(convert_to_bool), 1);
  rb_define_module_function(module, "to_type", RUBY_METHOD_FUNC(convert_to_type), 2);
} 