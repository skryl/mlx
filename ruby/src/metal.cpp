#include <ruby.h>
#include "mlx/backend/metal/metal.h"

namespace mx = mlx::core;

// Helper function to show a deprecation warning (to mirror Python's DEPRECATE macro)
static void rb_deprecate(const char* old_fn, const char* new_fn) {
    rb_warn("%s is deprecated and will be removed in a future version. Use %s instead.", old_fn, new_fn);
}

// Metal module methods
static VALUE metal_is_available(VALUE self) {
    // Check if the Metal back-end is available
    return mlx::core::metal::is_available() ? Qtrue : Qfalse;
}

static VALUE metal_get_active_memory(VALUE self) {
    // Mirror Python: mx.metal.get_active_memory() -> returns size_t
    rb_deprecate("mx.metal.get_active_memory", "mx.get_active_memory");
    size_t mem = mx::get_active_memory();
    return ULL2NUM(mem);
}

static VALUE metal_get_peak_memory(VALUE self) {
    // Mirror Python: mx.metal.get_peak_memory() -> returns size_t
    rb_deprecate("mx.metal.get_peak_memory", "mx.get_peak_memory");
    size_t mem = mx::get_peak_memory();
    return ULL2NUM(mem);
}

static VALUE metal_reset_peak_memory(VALUE self) {
    // Mirror Python: mx.metal.reset_peak_memory() -> returns void (nil)
    rb_deprecate("mx.metal.reset_peak_memory", "mx.reset_peak_memory");
    mx::reset_peak_memory();
    return Qnil;
}

static VALUE metal_get_cache_memory(VALUE self) {
    // Mirror Python: mx.metal.get_cache_memory() -> returns size_t
    rb_deprecate("mx.metal.get_cache_memory", "mx.get_cache_memory");
    size_t mem = mx::get_cache_memory();
    return ULL2NUM(mem);
}

static VALUE metal_set_memory_limit(VALUE self, VALUE limit_val) {
    // Mirror Python: mx.metal.set_memory_limit(limit)
    // returns size_t (old limit or new limit, depending on internal implementation).
    // This example assumes mx::set_memory_limit returns a size_t
    rb_deprecate("mx.metal.set_memory_limit", "mx.set_memory_limit");

    Check_Type(limit_val, T_FIXNUM); // or T_BIGNUM if extremely large
    size_t limit = NUM2ULL(limit_val);
    size_t result = mx::set_memory_limit(limit);
    return ULL2NUM(result);
}

static VALUE metal_set_cache_limit(VALUE self, VALUE limit_val) {
    // Mirror Python: mx.metal.set_cache_limit(limit)
    rb_deprecate("mx.metal.set_cache_limit", "mx.set_cache_limit");

    Check_Type(limit_val, T_FIXNUM);
    size_t limit = NUM2ULL(limit_val);
    size_t result = mx::set_cache_limit(limit);
    return ULL2NUM(result);
}

static VALUE metal_set_wired_limit(VALUE self, VALUE limit_val) {
    // Mirror Python: mx.metal.set_wired_limit(limit)
    rb_deprecate("mx.metal.set_wired_limit", "mx.set_wired_limit");

    Check_Type(limit_val, T_FIXNUM);
    size_t limit = NUM2ULL(limit_val);
    size_t result = mx::set_wired_limit(limit);
    return ULL2NUM(result);
}

static VALUE metal_clear_cache(VALUE self) {
    // Mirror Python: mx.metal.clear_cache() -> returns void
    rb_deprecate("mx.metal.clear_cache", "mx.clear_cache");
    mx::clear_cache();
    return Qnil;
}

static VALUE metal_start_capture(VALUE self, VALUE path) {
    // Start a Metal capture
    //
    // This begins capturing Metal GPU activity for debugging and performance analysis.
    // The capture can be viewed in Xcode or other Metal debugging tools.
    //
    // @param path [String] The path to save the capture which should have the extension ".gputrace"
    Check_Type(path, T_STRING);
    std::string path_str = StringValueCStr(path);
    
    mlx::core::metal::start_capture(path_str);
    return Qnil;
}

static VALUE metal_stop_capture(VALUE self) {
    // Stop a Metal capture
    //
    // This ends the Metal GPU activity capture started with start_metal_capture.
    mlx::core::metal::stop_capture();
    return Qnil;
}

static VALUE metal_device_info(VALUE self) {
    // Get information about the GPU device and system settings
    //
    // Currently returns:
    // * "architecture"
    // * "max_buffer_size"
    // * "max_recommended_working_set_size"
    // * "memory_size"
    // * "resource_limit"
    //
    // @return [Hash] A hash with string keys and string or integer values
    auto info = mlx::core::metal::device_info();
    VALUE result = rb_hash_new();
    
    // Convert C++ map to Ruby hash
    for (const auto& pair : info) {
        VALUE rb_key = rb_str_new_cstr(pair.first.c_str());
        VALUE rb_value;
        
        // Check if the value is a number (std::variant<std::string, size_t>)
        std::visit([&rb_value](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::string>) {
                rb_value = rb_str_new_cstr(arg.c_str());
            } else if constexpr (std::is_same_v<T, size_t>) {
                rb_value = ULL2NUM(arg);
            }
        }, pair.second);
        
        rb_hash_aset(result, rb_key, rb_value);
    }
    
    return result;
}

// Initialize metal module
void init_metal(VALUE module) {
    // Define module methods
    rb_define_module_function(module, "metal_is_available", RUBY_METHOD_FUNC(metal_is_available), 0);
    rb_define_module_function(module, "metal_get_active_memory", RUBY_METHOD_FUNC(metal_get_active_memory), 0);
    rb_define_module_function(module, "metal_get_peak_memory", RUBY_METHOD_FUNC(metal_get_peak_memory), 0);
    rb_define_module_function(module, "metal_reset_peak_memory", RUBY_METHOD_FUNC(metal_reset_peak_memory), 0);
    rb_define_module_function(module, "metal_get_cache_memory", RUBY_METHOD_FUNC(metal_get_cache_memory), 0);
    rb_define_module_function(module, "metal_set_memory_limit", RUBY_METHOD_FUNC(metal_set_memory_limit), 1);
    rb_define_module_function(module, "metal_set_cache_limit", RUBY_METHOD_FUNC(metal_set_cache_limit), 1);
    rb_define_module_function(module, "metal_set_wired_limit", RUBY_METHOD_FUNC(metal_set_wired_limit), 1);
    rb_define_module_function(module, "metal_clear_cache", RUBY_METHOD_FUNC(metal_clear_cache), 0);
    rb_define_module_function(module, "start_metal_capture", RUBY_METHOD_FUNC(metal_start_capture), 1);
    rb_define_module_function(module, "stop_metal_capture", RUBY_METHOD_FUNC(metal_stop_capture), 0);
    rb_define_module_function(module, "metal_device_info", RUBY_METHOD_FUNC(metal_device_info), 0);
} 