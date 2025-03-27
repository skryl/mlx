#include <ruby.h>
#include <cmath>
#include <limits>
#include "mlx/ops.h"

namespace mx = mlx::core;

// Helper function to wrap mx::array into Ruby VALUE
static VALUE wrap_array(const mx::array& arr) {
  return Data_Wrap_Struct(rb_path2class("MLX::Core::Array"), 0, nullptr, new mx::array(arr));
}

// Constants module methods
static VALUE constants_pi(VALUE self) {
  mx::array result = mx::array(3.1415926535897932384626433);
  return wrap_array(result);
}

static VALUE constants_e(VALUE self) {
  mx::array result = mx::array(2.71828182845904523536028747135266249775724709369995);
  return wrap_array(result);
}

static VALUE constants_euler_gamma(VALUE self) {
  mx::array result = mx::array(0.5772156649015328606065120900824024310421);
  return wrap_array(result);
}

static VALUE constants_inf(VALUE self) {
  mx::array result = mx::array(std::numeric_limits<double>::infinity());
  return wrap_array(result);
}

static VALUE constants_nan(VALUE self) {
  mx::array result = mx::array(NAN);
  return wrap_array(result);
}

static VALUE constants_newaxis(VALUE self) {
  return Qnil;
}

// Initialize constants module
void init_constants(VALUE module) {
  // Define module functions
  rb_define_module_function(module, "pi", RUBY_METHOD_FUNC(constants_pi), 0);
  rb_define_module_function(module, "e", RUBY_METHOD_FUNC(constants_e), 0);
  rb_define_module_function(module, "euler_gamma", RUBY_METHOD_FUNC(constants_euler_gamma), 0);
  rb_define_module_function(module, "inf", RUBY_METHOD_FUNC(constants_inf), 0);
  rb_define_module_function(module, "nan", RUBY_METHOD_FUNC(constants_nan), 0);
  rb_define_module_function(module, "newaxis", RUBY_METHOD_FUNC(constants_newaxis), 0);
} 