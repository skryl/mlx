#include <ruby.h>
#include <vector>
#include "mlx/utils.h"

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

// Utils module methods
static VALUE utils_tree_flatten(VALUE self, VALUE arr) {
  mx::array& a = get_array(arr);
  
  auto flattened = mx::tree_flatten(a);
  
  VALUE result = rb_ary_new();
  for (const auto& item : flattened) {
    rb_ary_push(result, wrap_array(item));
  }
  
  return result;
}

static VALUE utils_is_array_like(VALUE self, VALUE obj) {
  bool is_like = false;
  
  if (rb_obj_is_kind_of(obj, rb_path2class("MLX::Core::Array"))) {
    is_like = true;
  }
  
  return is_like ? Qtrue : Qfalse;
}

static VALUE utils_is_pytree_leaf(VALUE self, VALUE obj) {
  return utils_is_array_like(self, obj);
}

static VALUE utils_tree_map(VALUE self, VALUE func, VALUE arr) {
  // This requires handling Ruby blocks/procs
  rb_raise(rb_eNotImpError, "tree_map not fully implemented yet");
  return Qnil;
}

static VALUE utils_dtype_to_string(VALUE self, VALUE dtype) {
  int dtype_val = NUM2INT(dtype);
  mx::Dtype d = static_cast<mx::Dtype::Val>(dtype_val);
  
  std::string dtype_str;
  switch (d.val()) {
    case mx::bool_.val():
      dtype_str = "bool";
      break;
    case mx::uint8.val():
      dtype_str = "uint8";
      break;
    case mx::uint16.val():
      dtype_str = "uint16";
      break;
    case mx::uint32.val():
      dtype_str = "uint32";
      break;
    case mx::uint64.val():
      dtype_str = "uint64";
      break;
    case mx::int8.val():
      dtype_str = "int8";
      break;
    case mx::int16.val():
      dtype_str = "int16";
      break;
    case mx::int32.val():
      dtype_str = "int32";
      break;
    case mx::int64.val():
      dtype_str = "int64";
      break;
    case mx::float16.val():
      dtype_str = "float16";
      break;
    case mx::float32.val():
      dtype_str = "float32";
      break;
    case mx::bfloat16.val():
      dtype_str = "bfloat16";
      break;
    case mx::complex64.val():
      dtype_str = "complex64";
      break;
    default:
      dtype_str = "unknown";
  }
  
  return rb_str_new_cstr(dtype_str.c_str());
}

// Initialize utils module
void init_utils(VALUE module) {
  // Define module functions
  rb_define_module_function(module, "tree_flatten", RUBY_METHOD_FUNC(utils_tree_flatten), 1);
  rb_define_module_function(module, "is_array_like", RUBY_METHOD_FUNC(utils_is_array_like), 1);
  rb_define_module_function(module, "is_pytree_leaf", RUBY_METHOD_FUNC(utils_is_pytree_leaf), 1);
  rb_define_module_function(module, "tree_map", RUBY_METHOD_FUNC(utils_tree_map), 2);
  rb_define_module_function(module, "dtype_to_string", RUBY_METHOD_FUNC(utils_dtype_to_string), 1);
}