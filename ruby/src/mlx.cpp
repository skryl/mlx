#include <ruby.h>
#include <iostream>
#include <string>
#include "mlx/backend/metal/metal.h"

// Version information
#ifndef MLX_VERSION
#define MLX_VERSION "0.2.0"
#endif

// Ruby equivalent of Python's module system
extern "C" void Init_core(void);

// Function declarations for individual modules
void init_mlx_func(VALUE);
void init_device(VALUE);
void init_stream(VALUE);
void init_array(VALUE);
void init_metal(VALUE);
void init_memory(VALUE);
void init_ops(VALUE);
void init_transforms(VALUE);
void init_random(VALUE);
void init_fft(VALUE);
void init_linalg(VALUE);
void init_constants(VALUE);
void init_fast(VALUE);
void init_distributed(VALUE);
void init_export(VALUE);
void init_utils(VALUE);        // Ruby-specific
void init_trees(VALUE);        // Ruby-specific
void init_indexing(VALUE);     // Ruby-specific
void init_convert(VALUE);      // Ruby-specific
void init_load(VALUE);         // Ruby-specific

// Get MLX version information
static VALUE mlx_version(VALUE self) {
    return rb_str_new_cstr(MLX_VERSION);
}

// Get the MLX platform (CPU/Metal)
static VALUE mlx_platform(VALUE self) {
    return mlx::core::metal::is_available() ? rb_str_new_cstr("Metal") : rb_str_new_cstr("CPU");
}

extern "C" void Init_core(void) {
    // Create the root module MLX
    VALUE mlx_module = rb_define_module("MLX");
    
    // Define the core module
    VALUE core_module = rb_define_module_under(mlx_module, "Core");
    
    // Add module information (before initializing modules)
    rb_define_const(core_module, "VERSION", rb_str_new_cstr(MLX_VERSION));
    rb_define_const(core_module, "DESCRIPTION", rb_str_new_cstr("MLX framework for machine learning on Apple silicon"));
    
    // Initialize all submodules in the same order as the Python version
    init_mlx_func(core_module);  // 1
    init_device(core_module);    // 2
    init_stream(core_module);    // 3
    init_array(core_module);     // 4
    init_metal(core_module);     // 5
    init_memory(core_module);    // 6
    init_ops(core_module);       // 7
    init_transforms(core_module); // 8
    init_random(core_module);     // 9
    init_fft(core_module);        // 10
    init_linalg(core_module);     // 11
    init_constants(core_module);  // 12
    init_fast(core_module);       // 13
    init_distributed(core_module); // 14
    init_export(core_module);     // 15
    
    // Ruby-specific modules (not in Python)
    init_utils(core_module);
    init_trees(core_module);
    init_indexing(core_module);
    init_convert(core_module);
    init_load(core_module);
    
    // Add module functions
    rb_define_module_function(core_module, "version", RUBY_METHOD_FUNC(mlx_version), 0);
    rb_define_module_function(core_module, "platform", RUBY_METHOD_FUNC(mlx_platform), 0);
} 