#include <ruby.h>
#include "mlx/memory.h"

namespace mx = mlx::core;

// Memory module methods
static VALUE memory_capture_allocations(VALUE self) {
    mx::enable_allocation_tracking();
    return Qnil;
}

static VALUE memory_stop_allocations(VALUE self) {
    mx::disable_allocation_tracking();
    return Qnil;
}

static VALUE memory_get_allocations(VALUE self) {
    auto allocations = mx::allocations();
    VALUE result = rb_ary_new();
    
    for (const auto& alloc : allocations) {
        VALUE alloc_info = rb_hash_new();
        rb_hash_aset(alloc_info, ID2SYM(rb_intern("id")), ULL2NUM(alloc.id));
        rb_hash_aset(alloc_info, ID2SYM(rb_intern("size")), ULL2NUM(alloc.size));
        rb_hash_aset(alloc_info, ID2SYM(rb_intern("device")), INT2NUM(static_cast<int>(alloc.device.type)));
        rb_ary_push(result, alloc_info);
    }
    
    return result;
}

static VALUE memory_clear_allocations(VALUE self) {
    mx::clear_allocations();
    return Qnil;
}

static VALUE memory_retain(VALUE self, VALUE array_obj) {
    // Get mx::array from Ruby object
    mx::array* arr_ptr;
    Data_Get_Struct(array_obj, mx::array, arr_ptr);
    
    // Retain the array
    mx::retain(*arr_ptr);
    return array_obj;
}

static VALUE memory_release(VALUE self, VALUE array_obj) {
    // Get mx::array from Ruby object
    mx::array* arr_ptr;
    Data_Get_Struct(array_obj, mx::array, arr_ptr);
    
    // Release the array
    mx::release(*arr_ptr);
    return array_obj;
}

// Initialize memory module
void init_memory(VALUE module) {
    // Define module methods
    rb_define_module_function(module, "capture_allocations", RUBY_METHOD_FUNC(memory_capture_allocations), 0);
    rb_define_module_function(module, "stop_allocations", RUBY_METHOD_FUNC(memory_stop_allocations), 0);
    rb_define_module_function(module, "get_allocations", RUBY_METHOD_FUNC(memory_get_allocations), 0);
    rb_define_module_function(module, "clear_allocations", RUBY_METHOD_FUNC(memory_clear_allocations), 0);
    rb_define_module_function(module, "retain", RUBY_METHOD_FUNC(memory_retain), 1);
    rb_define_module_function(module, "release", RUBY_METHOD_FUNC(memory_release), 1);
} 