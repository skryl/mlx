#include <ruby.h>
#include "mlx/random.h"
#include "mlx/ops.h"

namespace mx = mlx::core;

// Helper function to wrap mx::array into Ruby VALUE
static VALUE wrap_array(const mx::array& arr) {
  return Data_Wrap_Struct(rb_path2class("MLX::Core::Array"), 0, nullptr, new mx::array(arr));
}

// Helper to extract Ruby array into C++ vector
static std::vector<int> ruby_array_to_vector(VALUE shape) {
  Check_Type(shape, T_ARRAY);
  
  std::vector<int> cpp_shape;
  for (long i = 0; i < RARRAY_LEN(shape); i++) {
    VALUE item = rb_ary_entry(shape, i);
    cpp_shape.push_back(NUM2INT(item));
  }
  
  return cpp_shape;
}

// Random module methods
static VALUE random_key(VALUE self, VALUE seed) {
  uint64_t seed_val = NUM2ULL(seed);
  mx::array key = mx::random::key(seed_val);
  return wrap_array(key);
}

static VALUE random_split(VALUE self, VALUE key, VALUE num) {
  mx::array* key_ptr;
  Data_Get_Struct(key, mx::array, key_ptr);
  
  int num_val = NUM2INT(num);
  std::vector<mx::array> keys = mx::random::split(*key_ptr, num_val);
  
  VALUE result = rb_ary_new();
  for (const auto& k : keys) {
    rb_ary_push(result, wrap_array(k));
  }
  
  return result;
}

static VALUE random_uniform(VALUE self, VALUE key, VALUE shape, VALUE dtype) {
  mx::array* key_ptr;
  Data_Get_Struct(key, mx::array, key_ptr);
  
  std::vector<int> cpp_shape = ruby_array_to_vector(shape);
  mx::Dtype d = static_cast<mx::Dtype::Val>(NUM2INT(dtype));
  
  mx::array result = mx::random::uniform(*key_ptr, cpp_shape, d);
  return wrap_array(result);
}

static VALUE random_normal(VALUE self, VALUE key, VALUE shape, VALUE dtype) {
  mx::array* key_ptr;
  Data_Get_Struct(key, mx::array, key_ptr);
  
  std::vector<int> cpp_shape = ruby_array_to_vector(shape);
  mx::Dtype d = static_cast<mx::Dtype::Val>(NUM2INT(dtype));
  
  mx::array result = mx::random::normal(*key_ptr, cpp_shape, d);
  return wrap_array(result);
}

static VALUE random_randint(VALUE self, VALUE key, VALUE low, VALUE high, VALUE shape, VALUE dtype) {
  mx::array* key_ptr;
  Data_Get_Struct(key, mx::array, key_ptr);
  
  int low_val = NUM2INT(low);
  int high_val = NUM2INT(high);
  std::vector<int> cpp_shape = ruby_array_to_vector(shape);
  mx::Dtype d = static_cast<mx::Dtype::Val>(NUM2INT(dtype));
  
  mx::array result = mx::random::randint(*key_ptr, low_val, high_val, cpp_shape, d);
  return wrap_array(result);
}

static VALUE random_bernoulli(VALUE self, VALUE key, VALUE p, VALUE shape) {
  mx::array* key_ptr;
  Data_Get_Struct(key, mx::array, key_ptr);
  
  double p_val = NUM2DBL(p);
  std::vector<int> cpp_shape = ruby_array_to_vector(shape);
  
  mx::array result = mx::random::bernoulli(*key_ptr, p_val, cpp_shape);
  return wrap_array(result);
}

// Initialize random module
void init_random(VALUE module) {
  // Define module functions
  rb_define_module_function(module, "key", RUBY_METHOD_FUNC(random_key), 1);
  rb_define_module_function(module, "split", RUBY_METHOD_FUNC(random_split), 2);
  rb_define_module_function(module, "uniform", RUBY_METHOD_FUNC(random_uniform), 3);
  rb_define_module_function(module, "normal", RUBY_METHOD_FUNC(random_normal), 3);
  rb_define_module_function(module, "randint", RUBY_METHOD_FUNC(random_randint), 5);
  rb_define_module_function(module, "bernoulli", RUBY_METHOD_FUNC(random_bernoulli), 3);
} 