#include <ruby.h>
#include "mlx/array.h"
#include "mlx_func.h"

namespace mx = mlx::core;

// Structure for the GC-aware function wrapper
struct gc_func {
  VALUE func;          // The Ruby function itself
  std::vector<VALUE> deps; // Dependencies that should not be GC'd while func is alive
};

// Ruby object lifecycle functions for gc_func
static void gc_func_mark(void* ptr) {
  gc_func* wrapper = static_cast<gc_func*>(ptr);
  // Mark the function to prevent GC
  rb_gc_mark(wrapper->func);
  
  // Mark all dependencies
  for (auto& dep : wrapper->deps) {
    rb_gc_mark(dep);
  }
}

static void gc_func_free(void* ptr) {
  gc_func* wrapper = static_cast<gc_func*>(ptr);
  delete wrapper;
}

// Structure for Ruby Data object type
static const rb_data_type_t gc_func_type = {
  "mlx_gc_func",
  {gc_func_mark, gc_func_free, nullptr, nullptr},
  nullptr, nullptr, RUBY_TYPED_FREE_IMMEDIATELY
};

// Function to call the wrapped Ruby function
static VALUE gc_func_call(int argc, VALUE* argv, VALUE self) {
  gc_func* wrapper;
  TypedData_Get_Struct(self, gc_func, &gc_func_type, wrapper);
  
  // Forward the call to the actual Ruby function
  return rb_funcall2(wrapper->func, rb_intern("call"), argc, argv);
}

// Create a GC-aware function wrapper
VALUE mlx_func_create(VALUE func, std::vector<VALUE> deps) {
  // Create wrapper object
  gc_func* wrapper = new gc_func();
  wrapper->func = func;
  wrapper->deps = std::move(deps);
  
  // Create Ruby object with proper GC handling
  VALUE result = TypedData_Wrap_Struct(rb_cObject, &gc_func_type, wrapper);
  
  // Define call method
  rb_define_method(rb_singleton_class(result), "call", RUBY_METHOD_FUNC(gc_func_call), -1);
  
  return result;
}

// Initialize the MLX function module
void init_mlx_func(VALUE module) {
  // Nothing to do here - just ensure the module is initialized
}
 