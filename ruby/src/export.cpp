#include <ruby.h>
#include <string>
#include <unordered_map>
#include <map>
#include <fstream>
#include <functional>
#include "mlx/export.h"
#include "mlx/graph_utils.h"
#include "mlx/io.h"

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

// Helper to make a copy of mx::array to avoid reference issues
static mx::array copy_array(const mx::array& arr) {
  return mx::array(arr); // Use copy constructor
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
    
    // Make a copy to avoid reference issues
    result.emplace(key_str, copy_array(*value_arr));
  }
  
  return result;
}

// Helper to convert Ruby Hash to C++ std::map<string, array>
static std::map<std::string, mx::array> ruby_hash_to_std_map(VALUE hash) {
  Check_Type(hash, T_HASH);
  
  std::map<std::string, mx::array> result;
  
  VALUE keys = rb_funcall(hash, rb_intern("keys"), 0);
  for (long i = 0; i < RARRAY_LEN(keys); i++) {
    VALUE key = rb_ary_entry(keys, i);
    VALUE value = rb_hash_aref(hash, key);
    
    // Convert key to string
    std::string key_str = StringValueCStr(key);
    
    // Extract array from value
    mx::array* value_arr;
    Data_Get_Struct(value, mx::array, value_arr);
    
    // Make a copy to avoid reference issues
    result.emplace(key_str, copy_array(*value_arr));
  }
  
  return result;
}

// Helper to extract arrays from Ruby array
static std::vector<mx::array> ruby_array_to_vector(VALUE arr) {
  Check_Type(arr, T_ARRAY);
  
  std::vector<mx::array> result;
  for (long i = 0; i < RARRAY_LEN(arr); i++) {
    VALUE item = rb_ary_entry(arr, i);
    mx::array* arr_ptr;
    Data_Get_Struct(item, mx::array, arr_ptr);
    result.push_back(copy_array(*arr_ptr));
  }
  
  return result;
}

// Export module methods
static VALUE export_to_safetensors(VALUE self, VALUE weights, VALUE path, VALUE metadata = Qnil) {
  Check_Type(path, T_STRING);
  std::string path_str = StringValueCStr(path);
  
  auto weights_map = ruby_hash_to_map(weights);
  
  // Process metadata if provided
  if (metadata != Qnil) {
    // Convert Ruby hash to std::unordered_map<std::string, std::string>
    std::unordered_map<std::string, std::string> metadata_map;
    Check_Type(metadata, T_HASH);
    
    VALUE meta_keys = rb_funcall(metadata, rb_intern("keys"), 0);
    for (long i = 0; i < RARRAY_LEN(meta_keys); i++) {
      VALUE key = rb_ary_entry(meta_keys, i);
      VALUE value = rb_hash_aref(metadata, key);
      
      Check_Type(key, T_STRING);
      Check_Type(value, T_STRING);
      
      std::string key_str = StringValueCStr(key);
      std::string value_str = StringValueCStr(value);
      
      metadata_map[key_str] = value_str;
    }
    
    // Save with metadata
    mx::save_safetensors(path_str, weights_map, metadata_map);
  } else {
    // Save without metadata
    mx::save_safetensors(path_str, weights_map);
  }
  
  return Qnil;
}

static VALUE export_to_gguf(VALUE self, VALUE weights, VALUE path, VALUE metadata = Qnil) {
  Check_Type(path, T_STRING);
  std::string path_str = StringValueCStr(path);
  
  auto weights_map = ruby_hash_to_map(weights);
  
  if (metadata != Qnil) {
    // Process metadata
    rb_raise(rb_eNotImpError, "GGUF metadata support not yet implemented for Ruby bindings");
    return Qnil;
  }
  
  // Save without metadata
  mx::save_gguf(path_str, weights_map);
  return Qnil;
}

static VALUE export_to_dot(int argc, VALUE* argv, VALUE self) {
  if (argc < 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected at least 2)", argc);
  }
  
  // Extract file parameter
  VALUE file = argv[0];
  Check_Type(file, T_STRING);
  std::string file_path = StringValueCStr(file);
  
  // Gather arrays from arguments
  std::vector<mx::array> arrays;
  mx::NodeNamer namer;
  
  // Process positional array arguments (starting from index 1)
  for (int i = 1; i < argc; i++) {
    VALUE arg = argv[i];
    
    // If it's a hash, treat as keyword arguments with names
    if (RB_TYPE_P(arg, T_HASH)) {
      VALUE keys = rb_funcall(arg, rb_intern("keys"), 0);
      for (long j = 0; j < RARRAY_LEN(keys); j++) {
        VALUE key = rb_ary_entry(keys, j);
        VALUE value = rb_hash_aref(arg, key);
        
        if (rb_obj_is_kind_of(value, rb_path2class("MLX::Core::Array"))) {
          mx::array& arr = get_array(value);
          arrays.push_back(copy_array(arr));
          
          // Set name in namer
          std::string name = StringValueCStr(key);
          namer.set_name(arrays.back(), name);
        }
      }
    } else if (rb_obj_is_kind_of(arg, rb_path2class("MLX::Core::Array"))) {
      // Regular array argument
      mx::array& arr = get_array(arg);
      arrays.push_back(copy_array(arr));
    } else if (RB_TYPE_P(arg, T_ARRAY)) {
      // Array of arrays
      for (long j = 0; j < RARRAY_LEN(arg); j++) {
        VALUE item = rb_ary_entry(arg, j);
        if (rb_obj_is_kind_of(item, rb_path2class("MLX::Core::Array"))) {
          mx::array& arr = get_array(item);
          arrays.push_back(copy_array(arr));
        }
      }
    }
  }
  
  // Check if we have any arrays to export
  if (arrays.empty()) {
    rb_raise(rb_eArgError, "No arrays provided for export_to_dot");
  }
  
  // Export to DOT format
  std::ofstream out(file_path);
  if (!out.is_open()) {
    rb_raise(rb_eRuntimeError, "Failed to open file for writing: %s", file_path.c_str());
  }
  
  mx::export_to_dot(out, std::move(namer), arrays);
  
  return Qnil;
}

// Initialize export module
void init_export(VALUE module) {
  // Define module functions
  rb_define_module_function(module, "to_safetensors", RUBY_METHOD_FUNC(export_to_safetensors), -1);
  rb_define_module_function(module, "to_gguf", RUBY_METHOD_FUNC(export_to_gguf), -1);
  rb_define_module_function(module, "to_dot", RUBY_METHOD_FUNC(export_to_dot), -1);
} 