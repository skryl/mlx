#include <ruby.h>
#include "mlx/fast.h"
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

// Fast module methods
static VALUE fast_gemm(VALUE self, VALUE a, VALUE b, VALUE c, VALUE transpose_a, VALUE transpose_b) {
  mx::array& arr_a = get_array(a);
  mx::array& arr_b = get_array(b);
  mx::array& arr_c = get_array(c);
  
  bool trans_a = RTEST(transpose_a);
  bool trans_b = RTEST(transpose_b);
  
  mx::array result = mx::fast::gemm(arr_a, arr_b, arr_c, trans_a, trans_b);
  return wrap_array(result);
}

static VALUE fast_scaled_dot_product_attention(VALUE self, VALUE queries, VALUE keys, VALUE values, VALUE scale, VALUE mask) {
  mx::array& q = get_array(queries);
  mx::array& k = get_array(keys);
  mx::array& v = get_array(values);
  
  double scale_val = NUM2DBL(scale);
  
  if (NIL_P(mask)) {
    mx::array result = mx::fast::scaled_dot_product_attention(q, k, v, scale_val);
    return wrap_array(result);
  } else {
    mx::array& m = get_array(mask);
    mx::array result = mx::fast::scaled_dot_product_attention(q, k, v, scale_val, m);
    return wrap_array(result);
  }
}

static VALUE fast_multi_head_attention(VALUE self, VALUE queries, VALUE keys, VALUE values, VALUE num_heads) {
  mx::array& q = get_array(queries);
  mx::array& k = get_array(keys);
  mx::array& v = get_array(values);
  
  int heads = NUM2INT(num_heads);
  
  mx::array result = mx::fast::multi_head_attention(q, k, v, heads);
  return wrap_array(result);
}

static VALUE fast_rms_norm(VALUE self, VALUE x, VALUE weight, VALUE eps) {
  mx::array& arr_x = get_array(x);
  mx::array& arr_w = get_array(weight);
  
  double eps_val = NUM2DBL(eps);
  
  mx::array result = mx::fast::rms_norm(arr_x, arr_w, eps_val);
  return wrap_array(result);
}

static VALUE fast_layer_norm(VALUE self, VALUE x, VALUE weight, VALUE bias, VALUE eps) {
  mx::array& arr_x = get_array(x);
  mx::array& arr_w = get_array(weight);
  
  if (NIL_P(bias)) {
    double eps_val = NUM2DBL(eps);
    mx::array result = mx::fast::layer_norm(arr_x, arr_w, std::nullopt, eps_val);
    return wrap_array(result);
  } else {
    mx::array& arr_b = get_array(bias);
    double eps_val = NUM2DBL(eps);
    mx::array result = mx::fast::layer_norm(arr_x, arr_w, arr_b, eps_val);
    return wrap_array(result);
  }
}

static VALUE fast_rope(VALUE self, VALUE x, VALUE dims, VALUE traditional, VALUE base, VALUE scale) {
  mx::array& arr_x = get_array(x);
  
  int dims_val = NUM2INT(dims);
  bool traditional_val = RTEST(traditional);
  double base_val = NUM2DBL(base);
  double scale_val = NUM2DBL(scale);
  
  mx::array result = mx::fast::rope(arr_x, dims_val, traditional_val, base_val, scale_val);
  return wrap_array(result);
}

static VALUE fast_rope_inplace(VALUE self, VALUE x, VALUE dims, VALUE traditional, VALUE base, VALUE scale) {
  mx::array& arr_x = get_array(x);
  
  int dims_val = NUM2INT(dims);
  bool traditional_val = RTEST(traditional);
  double base_val = NUM2DBL(base);
  double scale_val = NUM2DBL(scale);
  
  mx::fast::rope_inplace(arr_x, dims_val, traditional_val, base_val, scale_val);
  return x;
}

// Initialize fast module
void init_fast(VALUE module) {
  // Define module functions
  rb_define_module_function(module, "gemm", RUBY_METHOD_FUNC(fast_gemm), 5);
  rb_define_module_function(module, "scaled_dot_product_attention", RUBY_METHOD_FUNC(fast_scaled_dot_product_attention), 5);
  rb_define_module_function(module, "multi_head_attention", RUBY_METHOD_FUNC(fast_multi_head_attention), 4);
  rb_define_module_function(module, "rms_norm", RUBY_METHOD_FUNC(fast_rms_norm), 3);
  rb_define_module_function(module, "layer_norm", RUBY_METHOD_FUNC(fast_layer_norm), 4);
  rb_define_module_function(module, "rope", RUBY_METHOD_FUNC(fast_rope), 5);
  rb_define_module_function(module, "rope_inplace", RUBY_METHOD_FUNC(fast_rope_inplace), 5);
} 