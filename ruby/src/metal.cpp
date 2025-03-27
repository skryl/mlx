#include <ruby.h>
#include "mlx/backend/metal/metal.h"

namespace mx = mlx::core;

// Metal module methods
static VALUE metal_is_available(VALUE self) {
    // Check if the Metal back-end is available
    return mlx::core::metal::is_available() ? Qtrue : Qfalse;
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
    rb_define_module_function(module, "start_metal_capture", RUBY_METHOD_FUNC(metal_start_capture), 1);
    rb_define_module_function(module, "stop_metal_capture", RUBY_METHOD_FUNC(metal_stop_capture), 0);
    rb_define_module_function(module, "metal_device_info", RUBY_METHOD_FUNC(metal_device_info), 0);
} 