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

// --- Additional methods to mirror Python interface ---

// Return the __doc__ from the underlying function (if any)
static VALUE gc_func_get_doc(VALUE self) {
  gc_func* wrapper;
  TypedData_Get_Struct(self, gc_func, &gc_func_type, wrapper);

  // If we have a doc string stored somewhere or if the underlying
  // Ruby object implements `__doc__`, we can call that. In pure Ruby,
  // there's no standard doc for a proc/method, so either return nil
  // or call an existing method/ivar:

  // For example, if the underlying object has an instance variable @doc:
  // return rb_iv_get(wrapper->func, "@doc");

  // Or we can just return Qnil if no doc is available:
  return Qnil;
}

// Return the __nb_signature__ from the underlying function (if any)
static VALUE gc_func_get_nb_signature(VALUE self) {
  gc_func* wrapper;
  TypedData_Get_Struct(self, gc_func, &gc_func_type, wrapper);

  // Similarly, no built-in concept in Ruby, so we can do:
  return Qnil;
}

// Return the __vectorcalloffset__ (Python internal offset),
// for completeness we just return nil or a constant
static VALUE gc_func_get_vectorcalloffset(VALUE self) {
  // Not meaningful in Ruby, but we replicate the interface:
  return Qnil;
}

// method_missing to forward unknown calls to the underlying function
static VALUE gc_func_method_missing(int argc, VALUE* argv, VALUE self) {
  gc_func* wrapper;
  TypedData_Get_Struct(self, gc_func, &gc_func_type, wrapper);

  // The first argument to method_missing is the symbol of the method name
  VALUE method_sym = argv[0];

  // If the underlying object can respond, forward it:
  // Here we do a simple forward with rb_funcall, passing all args except argv[0].
  // We skip the method_sym in forward because it's the name, not a real argument.
  if (rb_respond_to(wrapper->func, RB_SYM2ID(method_sym))) {
    // Forward everything after the method symbol
    int fwd_argc = argc - 1;
    VALUE* fwd_argv = nullptr;
    if (fwd_argc > 0) {
      fwd_argv = &argv[1];
    }
    return rb_funcall2(wrapper->func, RB_SYM2ID(method_sym), fwd_argc, fwd_argv);
  }

  // Otherwise call super
  return rb_call_super(argc, argv);
}

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
  
  // --- Add the Python-equivalent methods ---
  // __doc__
  rb_define_method(rb_singleton_class(result), "__doc__",
                   RUBY_METHOD_FUNC(gc_func_get_doc), 0);

  // __nb_signature__
  rb_define_method(rb_singleton_class(result), "__nb_signature__",
                   RUBY_METHOD_FUNC(gc_func_get_nb_signature), 0);

  // __vectorcalloffset__
  rb_define_method(rb_singleton_class(result), "__vectorcalloffset__",
                   RUBY_METHOD_FUNC(gc_func_get_vectorcalloffset), 0);

  // method_missing for attribute/method delegation
  rb_define_method(rb_singleton_class(result), "method_missing",
                   RUBY_METHOD_FUNC(gc_func_method_missing), -1);

  return result;
}

// Initialize the MLX function module
void init_mlx_func(VALUE module) {
  // Nothing to do here - just ensure the module is initialized
}
 