#include <ruby.h>
#include <cmath>
#include "mlx/ops.h"

namespace mx = mlx::core;

// Helper function to wrap mx::array into Ruby VALUE
static VALUE wrap_array(const mx::array& arr) {
  return Data_Wrap_Struct(rb_path2class("MLX::Core::Array"), 0, nullptr, new mx::array(arr));
}

// Constants module methods
static VALUE constants_pi(VALUE self) {
  mx::array result = mx::array(M_PI);
  return wrap_array(result);
}

static VALUE constants_e(VALUE self) {
  mx::array result = mx::array(M_E);
  return wrap_array(result);
}

static VALUE constants_inf(VALUE self) {
  mx::array result = mx::array(INFINITY);
  return wrap_array(result);
}

static VALUE constants_nan(VALUE self) {
  mx::array result = mx::array(NAN);
  return wrap_array(result);
}

// Initialize constants module
void init_constants(VALUE module) {
  // Define module functions
  rb_define_module_function(module, "pi", RUBY_METHOD_FUNC(constants_pi), 0);
  rb_define_module_function(module, "e", RUBY_METHOD_FUNC(constants_e), 0);
  rb_define_module_function(module, "inf", RUBY_METHOD_FUNC(constants_inf), 0);
  rb_define_module_function(module, "nan", RUBY_METHOD_FUNC(constants_nan), 0);
} 