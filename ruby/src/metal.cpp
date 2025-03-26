#include <ruby.h>
#include "mlx/backend/metal/metal.h"

namespace mx = mlx::core;

// Metal module methods
static VALUE metal_is_available(VALUE self) {
    return mlx::core::metal::is_available() ? Qtrue : Qfalse;
}

static VALUE metal_set_compile_options(VALUE self, VALUE options) {
    Check_Type(options, T_HASH);
    
    std::unordered_map<std::string, std::string> compile_options;
    
    // Convert Ruby hash to C++ map
    VALUE keys = rb_funcall(options, rb_intern("keys"), 0);
    for (int i = 0; i < RARRAY_LEN(keys); i++) {
        VALUE key = rb_ary_entry(keys, i);
        VALUE value = rb_hash_aref(options, key);
        
        // Convert Ruby objects to C++ strings
        std::string key_str = StringValueCStr(key);
        std::string value_str = StringValueCStr(value);
        
        compile_options[key_str] = value_str;
    }
    
    mlx::core::metal::set_compile_options(compile_options);
    return options;
}

static VALUE metal_get_compile_options(VALUE self) {
    auto options = mlx::core::metal::get_compile_options();
    VALUE result = rb_hash_new();
    
    // Convert C++ map to Ruby hash
    for (const auto& pair : options) {
        rb_hash_aset(
            result,
            rb_str_new_cstr(pair.first.c_str()),
            rb_str_new_cstr(pair.second.c_str())
        );
    }
    
    return result;
}

static VALUE metal_get_state(VALUE self) {
    auto state = mlx::core::metal::get_state();
    VALUE result = rb_hash_new();
    
    // Convert C++ map to Ruby hash
    for (const auto& pair : state) {
        // Skip key-value pairs with empty values
        if (pair.second.empty()) {
            continue;
        }
        
        rb_hash_aset(
            result,
            rb_str_new_cstr(pair.first.c_str()),
            rb_str_new_cstr(pair.second.c_str())
        );
    }
    
    return result;
}

// Initialize metal module
void init_metal(VALUE module) {
    // Define module methods
    rb_define_module_function(module, "metal_is_available", RUBY_METHOD_FUNC(metal_is_available), 0);
    rb_define_module_function(module, "set_metal_compile_options", RUBY_METHOD_FUNC(metal_set_compile_options), 1);
    rb_define_module_function(module, "get_metal_compile_options", RUBY_METHOD_FUNC(metal_get_compile_options), 0);
    rb_define_module_function(module, "get_metal_state", RUBY_METHOD_FUNC(metal_get_state), 0);
} 