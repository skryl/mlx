#include <ruby.h>
#include <iostream>

// Ruby equivalent of Python's module system
extern "C" void Init_core(void);

// Function declarations for individual modules
void init_mlx_func(VALUE);
void init_array(VALUE);
void init_device(VALUE);
void init_stream(VALUE);
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
void init_utils(VALUE);
void init_trees(VALUE);
void init_indexing(VALUE);
void init_convert(VALUE);
void init_load(VALUE);

extern "C" void Init_core(void) {
    // Create the root module MLX
    VALUE mlx_module = rb_define_module("MLX");
    
    // Define the core module
    VALUE core_module = rb_define_module_under(mlx_module, "Core");
    
    // Initialize all submodules
    init_mlx_func(core_module);
    init_device(core_module);
    init_stream(core_module);
    init_array(core_module);
    init_metal(core_module);
    init_memory(core_module);
    init_ops(core_module);
    init_transforms(core_module);
    init_random(core_module);
    init_fft(core_module);
    init_linalg(core_module);
    init_constants(core_module);
    init_utils(core_module);
    init_trees(core_module);
    init_indexing(core_module);
    init_convert(core_module);
    init_fast(core_module);
    init_distributed(core_module);
    init_export(core_module);
    init_load(core_module);
    
    // Add version information
    rb_define_const(core_module, "VERSION", rb_str_new_cstr("0.1.0"));
} 