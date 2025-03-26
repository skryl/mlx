#include <ruby.h>
#include <unordered_map>
#include <string>
#include "mlx/load.h"
#include "load.h"

namespace mx = mlx::core;

// Helper function to wrap mx::array into Ruby VALUE
static VALUE wrap_array(const mx::array& arr) {
  return Data_Wrap_Struct(rb_path2class("MLX::Core::Array"), 0, nullptr, new mx::array(arr));
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

// Helper to convert C++ unordered_map<string, array> to Ruby Hash
static VALUE map_to_ruby_hash(const std::unordered_map<std::string, mx::array>& map) {
  VALUE result = rb_hash_new();
  
  for (const auto& [key, value] : map) {
    VALUE rb_key = rb_str_new_cstr(key.c_str());
    VALUE rb_value = wrap_array(value);
    
    rb_hash_aset(result, rb_key, rb_value);
  }
  
  return result;
}

// Load module methods
static VALUE load_load(VALUE self, VALUE path) {
  Check_Type(path, T_STRING);
  std::string path_str = StringValueCStr(path);
  
  auto result = mx::load(path_str);
  return map_to_ruby_hash(result);
}

static VALUE load_load_shard(VALUE self, VALUE path, VALUE shard_id, VALUE num_shards) {
  Check_Type(path, T_STRING);
  std::string path_str = StringValueCStr(path);
  
  int id = NUM2INT(shard_id);
  int shards = NUM2INT(num_shards);
  
  auto result = mx::load_shard(path_str, id, shards);
  return map_to_ruby_hash(result);
}

static VALUE load_save(VALUE self, VALUE weights, VALUE path) {
  Check_Type(path, T_STRING);
  std::string path_str = StringValueCStr(path);
  
  auto weights_map = ruby_hash_to_map(weights);
  
  mx::save(weights_map, path_str);
  return Qnil;
}

static VALUE load_save_shard(VALUE self, VALUE weights, VALUE path, VALUE shard_id, VALUE num_shards) {
  Check_Type(path, T_STRING);
  std::string path_str = StringValueCStr(path);
  
  int id = NUM2INT(shard_id);
  int shards = NUM2INT(num_shards);
  
  auto weights_map = ruby_hash_to_map(weights);
  
  mx::save_shard(weights_map, path_str, id, shards);
  return Qnil;
}

// Initialize load module
void init_load(VALUE module) {
  // Define module functions
  rb_define_module_function(module, "load", RUBY_METHOD_FUNC(load_load), 1);
  rb_define_module_function(module, "load_shard", RUBY_METHOD_FUNC(load_load_shard), 3);
  rb_define_module_function(module, "save", RUBY_METHOD_FUNC(load_save), 2);
  rb_define_module_function(module, "save_shard", RUBY_METHOD_FUNC(load_save_shard), 4);
} 