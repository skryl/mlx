#include <ruby.h>
#include "mlx/linalg.h"

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

// Linear algebra module methods
static VALUE linalg_norm(VALUE self, VALUE arr, VALUE ord, VALUE axis) {
  mx::array& a = get_array(arr);
  
  if (NIL_P(ord) && NIL_P(axis)) {
    // Default norm
    mx::array result = mx::linalg::norm(a);
    return wrap_array(result);
  } else if (!NIL_P(ord) && NIL_P(axis)) {
    // Norm with order
    double order = NUM2DBL(ord);
    mx::array result = mx::linalg::norm(a, order);
    return wrap_array(result);
  } else if (!NIL_P(ord) && !NIL_P(axis)) {
    // Norm with order and axis
    double order = NUM2DBL(ord);
    
    if (RB_TYPE_P(axis, T_ARRAY)) {
      // Multiple axes
      std::vector<int> axes;
      for (long i = 0; i < RARRAY_LEN(axis); i++) {
        VALUE item = rb_ary_entry(axis, i);
        axes.push_back(NUM2INT(item));
      }
      mx::array result = mx::linalg::norm(a, order, axes);
      return wrap_array(result);
    } else {
      // Single axis
      int ax = NUM2INT(axis);
      mx::array result = mx::linalg::norm(a, order, ax);
      return wrap_array(result);
    }
  } else {
    // Axis only
    if (RB_TYPE_P(axis, T_ARRAY)) {
      // Multiple axes
      std::vector<int> axes;
      for (long i = 0; i < RARRAY_LEN(axis); i++) {
        VALUE item = rb_ary_entry(axis, i);
        axes.push_back(NUM2INT(item));
      }
      mx::array result = mx::linalg::norm(a, 2.0, axes);
      return wrap_array(result);
    } else {
      // Single axis
      int ax = NUM2INT(axis);
      mx::array result = mx::linalg::norm(a, 2.0, ax);
      return wrap_array(result);
    }
  }
}

static VALUE linalg_svd(VALUE self, VALUE arr, VALUE full_matrices) {
  mx::array& a = get_array(arr);
  bool full = RTEST(full_matrices);
  
  std::tuple<mx::array, mx::array, mx::array> result = mx::linalg::svd(a, full);
  
  VALUE rb_result = rb_ary_new();
  rb_ary_push(rb_result, wrap_array(std::get<0>(result)));
  rb_ary_push(rb_result, wrap_array(std::get<1>(result)));
  rb_ary_push(rb_result, wrap_array(std::get<2>(result)));
  
  return rb_result;
}

static VALUE linalg_qr(VALUE self, VALUE arr, VALUE mode) {
  mx::array& a = get_array(arr);
  
  std::string mode_str = StringValueCStr(mode);
  std::tuple<mx::array, mx::array> result = mx::linalg::qr(a, mode_str);
  
  VALUE rb_result = rb_ary_new();
  rb_ary_push(rb_result, wrap_array(std::get<0>(result)));
  rb_ary_push(rb_result, wrap_array(std::get<1>(result)));
  
  return rb_result;
}

static VALUE linalg_inv(VALUE self, VALUE arr) {
  mx::array& a = get_array(arr);
  
  mx::array result = mx::linalg::inv(a);
  return wrap_array(result);
}

static VALUE linalg_matmul(VALUE self, VALUE a, VALUE b) {
  mx::array& arr_a = get_array(a);
  mx::array& arr_b = get_array(b);
  
  mx::array result = mx::matmul(arr_a, arr_b);
  return wrap_array(result);
}

static VALUE linalg_det(VALUE self, VALUE arr) {
  mx::array& a = get_array(arr);
  
  mx::array result = mx::linalg::det(a);
  return wrap_array(result);
}

// Initialize linear algebra module
void init_linalg(VALUE module) {
  // Define module functions
  rb_define_module_function(module, "norm", RUBY_METHOD_FUNC(linalg_norm), 3);
  rb_define_module_function(module, "svd", RUBY_METHOD_FUNC(linalg_svd), 2);
  rb_define_module_function(module, "qr", RUBY_METHOD_FUNC(linalg_qr), 2);
  rb_define_module_function(module, "inv", RUBY_METHOD_FUNC(linalg_inv), 1);
  rb_define_module_function(module, "matmul", RUBY_METHOD_FUNC(linalg_matmul), 2);
  rb_define_module_function(module, "det", RUBY_METHOD_FUNC(linalg_det), 1);
} 