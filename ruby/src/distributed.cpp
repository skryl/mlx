#include <ruby.h>
#include <vector>
#include "mlx/distributed.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

// Helper function to wrap mx::array into Ruby VALUE
static VALUE wrap_array(const mx::array& arr) {
  return Data_Wrap_Struct(rb_path2class("MLX::Core::Array"), 0, nullptr, new mx::array(arr));
}

// Helper to extract mx::array from Ruby VALUE
static mx::array& get_array(VALUE obj) {
  mx::array* arr_ptr;
  Data_Get_Struct(obj, mx::array, arr_ptr);
  return *arr_ptr;
}

// Distributed module methods
static VALUE distributed_initialize(VALUE self, VALUE communication_key, VALUE world_size, VALUE rank) {
  Check_Type(communication_key, T_STRING);
  std::string key_str = StringValueCStr(communication_key);
  
  int size = NUM2INT(world_size);
  int r = NUM2INT(rank);
  
  mx::distributed::initialize(key_str, size, r);
  return Qnil;
}

static VALUE distributed_is_initialized(VALUE self) {
  bool result = mx::distributed::is_initialized();
  return result ? Qtrue : Qfalse;
}

static VALUE distributed_world_size(VALUE self) {
  int result = mx::distributed::world_size();
  return INT2NUM(result);
}

static VALUE distributed_rank(VALUE self) {
  int result = mx::distributed::rank();
  return INT2NUM(result);
}

static VALUE distributed_local_world_size(VALUE self) {
  int result = mx::distributed::local_world_size();
  return INT2NUM(result);
}

static VALUE distributed_local_rank(VALUE self) {
  int result = mx::distributed::local_rank();
  return INT2NUM(result);
}

static VALUE distributed_local_group_size(VALUE self) {
  int result = mx::distributed::local_group_size();
  return INT2NUM(result);
}

static VALUE distributed_local_group_rank(VALUE self) {
  int result = mx::distributed::local_group_rank();
  return INT2NUM(result);
}

static VALUE distributed_shutdown(VALUE self) {
  mx::distributed::shutdown();
  return Qnil;
}

static VALUE distributed_barrier(VALUE self) {
  mx::distributed::barrier();
  return Qnil;
}

static VALUE distributed_all_reduce(VALUE self, VALUE input, VALUE reduction) {
  mx::array& arr = get_array(input);
  
  Check_Type(reduction, T_SYMBOL);
  ID reduction_id = SYM2ID(reduction);
  
  mx::distributed::ReduceOp op;
  if (reduction_id == rb_intern("sum")) {
    op = mx::distributed::ReduceOp::SUM;
  } else if (reduction_id == rb_intern("prod")) {
    op = mx::distributed::ReduceOp::PRODUCT;
  } else if (reduction_id == rb_intern("min")) {
    op = mx::distributed::ReduceOp::MIN;
  } else if (reduction_id == rb_intern("max")) {
    op = mx::distributed::ReduceOp::MAX;
  } else {
    rb_raise(rb_eArgError, "Invalid reduction operation. Must be one of: :sum, :prod, :min, :max");
    return Qnil;
  }
  
  mx::array result = mx::distributed::all_reduce(arr, op);
  return wrap_array(result);
}

static VALUE distributed_all_gather(VALUE self, VALUE input) {
  mx::array& arr = get_array(input);
  
  mx::array result = mx::distributed::all_gather(arr);
  return wrap_array(result);
}

static VALUE distributed_reduce_scatter(VALUE self, VALUE input, VALUE reduction) {
  mx::array& arr = get_array(input);
  
  Check_Type(reduction, T_SYMBOL);
  ID reduction_id = SYM2ID(reduction);
  
  mx::distributed::ReduceOp op;
  if (reduction_id == rb_intern("sum")) {
    op = mx::distributed::ReduceOp::SUM;
  } else if (reduction_id == rb_intern("prod")) {
    op = mx::distributed::ReduceOp::PRODUCT;
  } else if (reduction_id == rb_intern("min")) {
    op = mx::distributed::ReduceOp::MIN;
  } else if (reduction_id == rb_intern("max")) {
    op = mx::distributed::ReduceOp::MAX;
  } else {
    rb_raise(rb_eArgError, "Invalid reduction operation. Must be one of: :sum, :prod, :min, :max");
    return Qnil;
  }
  
  mx::array result = mx::distributed::reduce_scatter(arr, op);
  return wrap_array(result);
}

static VALUE distributed_broadcast(VALUE self, VALUE input, VALUE root) {
  mx::array& arr = get_array(input);
  int r = NUM2INT(root);
  
  mx::array result = mx::distributed::broadcast(arr, r);
  return wrap_array(result);
}

// Initialize distributed module
void init_distributed(VALUE module) {
  // Define module functions
  rb_define_module_function(module, "initialize", RUBY_METHOD_FUNC(distributed_initialize), 3);
  rb_define_module_function(module, "is_initialized", RUBY_METHOD_FUNC(distributed_is_initialized), 0);
  rb_define_module_function(module, "world_size", RUBY_METHOD_FUNC(distributed_world_size), 0);
  rb_define_module_function(module, "rank", RUBY_METHOD_FUNC(distributed_rank), 0);
  rb_define_module_function(module, "local_world_size", RUBY_METHOD_FUNC(distributed_local_world_size), 0);
  rb_define_module_function(module, "local_rank", RUBY_METHOD_FUNC(distributed_local_rank), 0);
  rb_define_module_function(module, "local_group_size", RUBY_METHOD_FUNC(distributed_local_group_size), 0);
  rb_define_module_function(module, "local_group_rank", RUBY_METHOD_FUNC(distributed_local_group_rank), 0);
  rb_define_module_function(module, "shutdown", RUBY_METHOD_FUNC(distributed_shutdown), 0);
  rb_define_module_function(module, "barrier", RUBY_METHOD_FUNC(distributed_barrier), 0);
  rb_define_module_function(module, "all_reduce", RUBY_METHOD_FUNC(distributed_all_reduce), 2);
  rb_define_module_function(module, "all_gather", RUBY_METHOD_FUNC(distributed_all_gather), 1);
  rb_define_module_function(module, "reduce_scatter", RUBY_METHOD_FUNC(distributed_reduce_scatter), 2);
  rb_define_module_function(module, "broadcast", RUBY_METHOD_FUNC(distributed_broadcast), 2);
} 