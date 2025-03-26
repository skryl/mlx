#include <ruby.h>
#include <string>
#include <unordered_map>
#include "mlx/export.h"

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

// Helper to convert Ruby Hash to C++ unordered_map<string, array>
static std::unordered_map<std::string, mx::array> ruby_hash_to_map(VALUE hash) {
  Check_Type(hash, T_HASH);
  
  std::unordered_map<std::string, mx::array> result;
  
  VALUE keys = rb_funcall(hash, rb_intern("keys"), 0);
  for (long i = 0; i < RARRAY_LEN(keys); i++) {
    VALUE key = rb_ary_entry(keys, i);
    VALUE value = rb_hash_aref(hash, key);
    
    // Convert key to string
    std::string key_str = StringValueCStr(key);
    
    // Extract array from value
    mx::array* value_arr;
    Data_Get_Struct(value, mx::array, value_arr);
    
    result[key_str] = *value_arr;
  }
  
  return result;
}

// Export module methods
static VALUE export_to_safetensors(VALUE self, VALUE weights, VALUE path) {
  Check_Type(path, T_STRING);
  std::string path_str = StringValueCStr(path);
  
  auto weights_map = ruby_hash_to_map(weights);
  
  mx::save_safetensors(weights_map, path_str);
  return Qnil;
}

static VALUE export_to_gguf(VALUE self, VALUE weights, VALUE path, VALUE metadata) {
  Check_Type(path, T_STRING);
  std::string path_str = StringValueCStr(path);
  
  auto weights_map = ruby_hash_to_map(weights);
  
  // Process metadata
  std::unordered_map<std::string, mx::export_gguf::GGUFMetadataValue> meta_map;
  
  if (!NIL_P(metadata)) {
    Check_Type(metadata, T_HASH);
    
    VALUE meta_keys = rb_funcall(metadata, rb_intern("keys"), 0);
    for (long i = 0; i < RARRAY_LEN(meta_keys); i++) {
      VALUE key = rb_ary_entry(meta_keys, i);
      VALUE value = rb_hash_aref(metadata, key);
      
      // Convert key to string
      std::string key_str = StringValueCStr(key);
      
      // Process value based on type
      if (rb_obj_is_kind_of(value, rb_cString)) {
        std::string str_val = StringValueCStr(value);
        meta_map[key_str] = str_val;
      } else if (rb_obj_is_kind_of(value, rb_cNumeric)) {
        if (rb_obj_is_kind_of(value, rb_cInteger)) {
          int64_t int_val = NUM2LL(value);
          meta_map[key_str] = int_val;
        } else {
          double float_val = NUM2DBL(value);
          meta_map[key_str] = float_val;
        }
      } else if (RB_TYPE_P(value, T_ARRAY)) {
        if (RARRAY_LEN(value) > 0) {
          VALUE first_item = rb_ary_entry(value, 0);
          
          if (rb_obj_is_kind_of(first_item, rb_cString)) {
            // Array of strings
            std::vector<std::string> str_vec;
            for (long j = 0; j < RARRAY_LEN(value); j++) {
              VALUE item = rb_ary_entry(value, j);
              str_vec.push_back(StringValueCStr(item));
            }
            meta_map[key_str] = str_vec;
          } else if (rb_obj_is_kind_of(first_item, rb_cNumeric)) {
            if (rb_obj_is_kind_of(first_item, rb_cInteger)) {
              // Array of integers
              std::vector<int64_t> int_vec;
              for (long j = 0; j < RARRAY_LEN(value); j++) {
                VALUE item = rb_ary_entry(value, j);
                int_vec.push_back(NUM2LL(item));
              }
              meta_map[key_str] = int_vec;
            } else {
              // Array of floats
              std::vector<double> float_vec;
              for (long j = 0; j < RARRAY_LEN(value); j++) {
                VALUE item = rb_ary_entry(value, j);
                float_vec.push_back(NUM2DBL(item));
              }
              meta_map[key_str] = float_vec;
            }
          }
        }
      } else if (value == Qtrue || value == Qfalse) {
        bool bool_val = (value == Qtrue);
        meta_map[key_str] = bool_val;
      }
    }
  }
  
  mx::save_gguf(weights_map, path_str, meta_map);
  return Qnil;
}

// Initialize export module
void init_export(VALUE module) {
  // Define module functions
  rb_define_module_function(module, "to_safetensors", RUBY_METHOD_FUNC(export_to_safetensors), 2);
  rb_define_module_function(module, "to_gguf", RUBY_METHOD_FUNC(export_to_gguf), 3);
} 