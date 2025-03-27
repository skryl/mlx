#include <ruby.h>
#include "mlx/memory.h"

namespace mx = mlx::core;

// Memory module methods
static VALUE memory_get_active_memory(VALUE self) {
    // Get the actively used memory in bytes.
    //
    // Note, this will not always match memory use reported by the system because
    // it does not include cached memory buffers.
    size_t memory = mx::get_active_memory();
    return ULL2NUM(memory);
}

static VALUE memory_get_peak_memory(VALUE self) {
    // Get the peak amount of used memory in bytes.
    //
    // The maximum memory used recorded from the beginning of the program
    // execution or since the last call to reset_peak_memory.
    size_t memory = mx::get_peak_memory();
    return ULL2NUM(memory);
}

static VALUE memory_reset_peak_memory(VALUE self) {
    // Reset the peak memory to zero.
    mx::reset_peak_memory();
    return Qnil;
}

static VALUE memory_get_cache_memory(VALUE self) {
    // Get the cache size in bytes.
    //
    // The cache includes memory not currently used that has not been returned
    // to the system allocator.
    size_t memory = mx::get_cache_memory();
    return ULL2NUM(memory);
}

static VALUE memory_set_memory_limit(VALUE self, VALUE limit) {
    // Set the memory limit.
    //
    // The memory limit is a guideline for the maximum amount of memory to use
    // during graph evaluation. If the memory limit is exceeded and there is no
    // more RAM (including swap when available) allocations will result in an
    // exception.
    //
    // When metal is available the memory limit defaults to 1.5 times the
    // maximum recommended working set size reported by the device.
    //
    // @param limit [Integer] Memory limit in bytes.
    // @return [Integer] The previous memory limit in bytes.
    size_t limit_val = NUM2ULL(limit);
    size_t previous = mx::set_memory_limit(limit_val);
    return ULL2NUM(previous);
}

static VALUE memory_set_cache_limit(VALUE self, VALUE limit) {
    // Set the free cache limit.
    //
    // If using more than the given limit, free memory will be reclaimed
    // from the cache on the next allocation. To disable the cache, set
    // the limit to 0.
    //
    // The cache limit defaults to the memory limit. See
    // set_memory_limit for more details.
    //
    // @param limit [Integer] The cache limit in bytes.
    // @return [Integer] The previous cache limit in bytes.
    size_t limit_val = NUM2ULL(limit);
    size_t previous = mx::set_cache_limit(limit_val);
    return ULL2NUM(previous);
}

static VALUE memory_set_wired_limit(VALUE self, VALUE limit) {
    // Set the wired size limit.
    //
    // Note:
    // * This function is only useful on macOS 15.0 or higher.
    // * The wired limit should remain strictly less than the total
    //   memory size.
    //
    // The wired limit is the total size in bytes of memory that will be kept
    // resident. The default value is 0.
    //
    // Setting a wired limit larger than system wired limit is an error. You can
    // increase the system wired limit with:
    //
    //   sudo sysctl iogpu.wired_limit_mb=<size_in_megabytes>
    //
    // Use device_info to query the system wired limit
    // ("max_recommended_working_set_size") and the total memory size
    // ("memory_size").
    //
    // @param limit [Integer] The wired limit in bytes.
    // @return [Integer] The previous wired limit in bytes.
    size_t limit_val = NUM2ULL(limit);
    size_t previous = mx::set_wired_limit(limit_val);
    return ULL2NUM(previous);
}

static VALUE memory_clear_cache(VALUE self) {
    // Clear the memory cache.
    //
    // After calling this, get_cache_memory should return 0.
    mx::clear_cache();
    return Qnil;
}

// Initialize memory module
void init_memory(VALUE module) {
    // Define module methods
    rb_define_module_function(module, "get_active_memory", RUBY_METHOD_FUNC(memory_get_active_memory), 0);
    rb_define_module_function(module, "get_peak_memory", RUBY_METHOD_FUNC(memory_get_peak_memory), 0);
    rb_define_module_function(module, "reset_peak_memory", RUBY_METHOD_FUNC(memory_reset_peak_memory), 0);
    rb_define_module_function(module, "get_cache_memory", RUBY_METHOD_FUNC(memory_get_cache_memory), 0);
    rb_define_module_function(module, "set_memory_limit", RUBY_METHOD_FUNC(memory_set_memory_limit), 1);
    rb_define_module_function(module, "set_cache_limit", RUBY_METHOD_FUNC(memory_set_cache_limit), 1);
    rb_define_module_function(module, "set_wired_limit", RUBY_METHOD_FUNC(memory_set_wired_limit), 1);
    rb_define_module_function(module, "clear_cache", RUBY_METHOD_FUNC(memory_clear_cache), 0);
} 