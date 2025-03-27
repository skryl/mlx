#include <ruby.h>
#include <vector>
#include <optional>
#include "mlx/distributed/distributed.h"
#include "mlx/distributed/ops.h"
#include "mlx/utils.h"
#include "mlx/dtype.h"

namespace mx = mlx::core;

// Define Group class wrapper
typedef struct {
  std::shared_ptr<mx::distributed::Group> group;
} GroupWrapper;

// GC callback for Group
static void group_free(void* ptr) {
  GroupWrapper* wrapper = static_cast<GroupWrapper*>(ptr);
  delete wrapper;
}

// Allocate a new Group
static VALUE group_alloc(VALUE klass) {
  GroupWrapper* wrapper = new GroupWrapper();
  return Data_Wrap_Struct(klass, 0, group_free, wrapper);
}

// Get Group from Ruby VALUE
static std::shared_ptr<mx::distributed::Group>& get_group(VALUE obj) {
  GroupWrapper* wrapper;
  Data_Get_Struct(obj, GroupWrapper, wrapper);
  return wrapper->group;
}

// Initialize a Group
static VALUE group_initialize(VALUE self, VALUE size, VALUE rank) {
  rb_raise(rb_eRuntimeError, "Cannot directly initialize Group objects. Use MLX::Distributed.init instead.");
  return self;
}

// Group methods
static VALUE group_rank(VALUE self) {
  auto& group = get_group(self);
  return INT2NUM(group->rank());
}

static VALUE group_size(VALUE self) {
  auto& group = get_group(self);
  return INT2NUM(group->size());
}

static VALUE group_split(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..2)", argc);
  }

  VALUE color = argv[0];
  auto& group = get_group(self);
  int c = NUM2INT(color);

  int k = -1;
  if (argc == 2) {
    k = NIL_P(argv[1]) ? -1 : NUM2INT(argv[1]);
  }

  auto new_group = group->split(c, k);

  // Create a new Group Ruby object
  VALUE group_class = rb_obj_class(self);
  GroupWrapper* wrapper = new GroupWrapper();
  wrapper->group = std::make_shared<mx::distributed::Group>(new_group);

  return Data_Wrap_Struct(group_class, 0, group_free, wrapper);
}

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

// Try to handle scalar or array inputs the same way Python does
// by automatically converting numeric scalars to 0D arrays.
static mx::array get_array_or_scalar(VALUE obj) {
  // If it's numeric (Fixnum or Float in Ruby), wrap it
  if (RB_TYPE_P(obj, T_FIXNUM) || RB_TYPE_P(obj, T_FLOAT)) {
    double val = NUM2DBL(obj);
    // Create a scalar array directly
    return mx::array(val, mx::float64);
  } else {
    // Otherwise, expect an MLX::Core::Array
    return get_array(obj);
  }
}

// Helper to extract StreamOrDevice from Ruby VALUE
static mx::StreamOrDevice get_stream_or_device(VALUE stream_obj) {
  if (NIL_P(stream_obj)) {
    return mx::StreamOrDevice{};
  }
  
  // Check if it's a Stream object
  if (rb_obj_is_kind_of(stream_obj, rb_path2class("MLX::Stream"))) {
    mx::Stream* stream_ptr;
    Data_Get_Struct(stream_obj, mx::Stream, stream_ptr);
    return mx::StreamOrDevice(*stream_ptr);
  }
  
  // Check if it's a Device object
  if (rb_obj_is_kind_of(stream_obj, rb_path2class("MLX::Device"))) {
    mx::Device* device_ptr;
    Data_Get_Struct(stream_obj, mx::Device, device_ptr);
    return mx::StreamOrDevice(*device_ptr);
  }
  
  rb_raise(rb_eTypeError, "Expected Stream or Device object");
  return mx::StreamOrDevice{};
}

// Convert Ruby group to optional<Group>
static std::optional<mx::distributed::Group> ruby_to_group(VALUE group_obj) {
  if (NIL_P(group_obj)) {
    return std::nullopt;
  }
  
  auto& group = get_group(group_obj);
  return *group;
}

// Distributed module methods
static VALUE distributed_is_available(VALUE self) {
  bool result = mx::distributed::is_available();
  return result ? Qtrue : Qfalse;
}

static VALUE distributed_init(int argc, VALUE* argv, VALUE self) {
  bool strict = false;
  std::string backend = "any";
  
  if (argc > 0) {
    strict = RTEST(argv[0]);
  }
  
  if (argc > 1) {
    Check_Type(argv[1], T_STRING);
    backend = StringValueCStr(argv[1]);
  }
  
  try {
    auto group = mx::distributed::init(strict, backend);
    
    // Create a new Group Ruby object
    VALUE group_class = rb_path2class("MLX::Core::Distributed::Group");
    GroupWrapper* wrapper = new GroupWrapper();
    wrapper->group = std::make_shared<mx::distributed::Group>(group);
    
    return Data_Wrap_Struct(group_class, 0, group_free, wrapper);
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "Error initializing distributed: %s", e.what());
    return Qnil;
  }
}

static VALUE distributed_all_sum(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..3)", argc);
  }
  
  mx::array arr = get_array_or_scalar(argv[0]);
  
  VALUE group_obj = Qnil;
  VALUE stream_obj = Qnil;
  
  if (argc > 1) {
    // Check if argc == 2 with named params
    if (argc == 2 && rb_obj_is_kind_of(argv[1], rb_cHash)) {
      VALUE hash = argv[1];
      group_obj = rb_hash_aref(hash, ID2SYM(rb_intern("group")));
      stream_obj = rb_hash_aref(hash, ID2SYM(rb_intern("stream")));
    } else {
      // Positional params
      group_obj = argv[1];
      if (argc > 2) {
        stream_obj = argv[2];
      }
    }
  }
  
  auto group = ruby_to_group(group_obj);
  auto stream_or_device = get_stream_or_device(stream_obj);
  
  mx::array result = mx::distributed::all_sum(arr, group, stream_or_device);
  return wrap_array(result);
}

static VALUE distributed_all_gather(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..3)", argc);
  }
  
  mx::array arr = get_array_or_scalar(argv[0]);
  
  VALUE group_obj = Qnil;
  VALUE stream_obj = Qnil;
  
  if (argc > 1) {
    // Check if argc == 2 with named params
    if (argc == 2 && rb_obj_is_kind_of(argv[1], rb_cHash)) {
      VALUE hash = argv[1];
      group_obj = rb_hash_aref(hash, ID2SYM(rb_intern("group")));
      stream_obj = rb_hash_aref(hash, ID2SYM(rb_intern("stream")));
    } else {
      // Positional params
      group_obj = argv[1];
      if (argc > 2) {
        stream_obj = argv[2];
      }
    }
  }
  
  auto group = ruby_to_group(group_obj);
  auto stream_or_device = get_stream_or_device(stream_obj);
  
  mx::array result = mx::distributed::all_gather(arr, group, stream_or_device);
  return wrap_array(result);
}

static VALUE distributed_send(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..4)", argc);
  }
  
  mx::array arr = get_array_or_scalar(argv[0]);
  int dst = NUM2INT(argv[1]);
  
  VALUE group_obj = Qnil;
  VALUE stream_obj = Qnil;
  
  if (argc > 2) {
    // Check if argc == 3 with named params
    if (argc == 3 && rb_obj_is_kind_of(argv[2], rb_cHash)) {
      VALUE hash = argv[2];
      group_obj = rb_hash_aref(hash, ID2SYM(rb_intern("group")));
      stream_obj = rb_hash_aref(hash, ID2SYM(rb_intern("stream")));
    } else {
      // Positional params
      group_obj = argv[2];
      if (argc > 3) {
        stream_obj = argv[3];
      }
    }
  }
  
  auto group = ruby_to_group(group_obj);
  auto stream_or_device = get_stream_or_device(stream_obj);
  
  mx::array result = mx::distributed::send(arr, dst, group, stream_or_device);
  return wrap_array(result);
}

static VALUE distributed_recv(int argc, VALUE* argv, VALUE self) {
  if (argc < 3 || argc > 5) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 3..5)", argc);
  }
  
  // Extract shape
  Check_Type(argv[0], T_ARRAY);
  std::vector<int> shape;
  for (long i = 0; i < RARRAY_LEN(argv[0]); i++) {
    VALUE item = rb_ary_entry(argv[0], i);
    shape.push_back(NUM2INT(item));
  }
  
  // Extract dtype
  int dtype_val = NUM2INT(argv[1]);
  mx::Dtype dtype = mx::float32; // Default to float32
  
  // Map integer to predefined dtype constants
  switch (dtype_val) {
    case 0: dtype = mx::bool_; break;
    case 1: dtype = mx::uint8; break;
    case 2: dtype = mx::uint16; break;
    case 3: dtype = mx::uint32; break;
    case 4: dtype = mx::uint64; break;
    case 5: dtype = mx::int8; break;
    case 6: dtype = mx::int16; break;
    case 7: dtype = mx::int32; break;
    case 8: dtype = mx::int64; break;
    case 9: dtype = mx::float16; break;
    case 10: dtype = mx::float32; break;
    case 11: dtype = mx::float64; break;
    case 12: dtype = mx::bfloat16; break;
    case 13: dtype = mx::complex64; break;
    default:
      rb_raise(rb_eArgError, "Invalid dtype value: %d", dtype_val);
  }
  
  // Extract source rank
  int src = NUM2INT(argv[2]);
  
  VALUE group_obj = Qnil;
  VALUE stream_obj = Qnil;
  
  if (argc > 3) {
    // Check if argc == 4 with named params
    if (argc == 4 && rb_obj_is_kind_of(argv[3], rb_cHash)) {
      VALUE hash = argv[3];
      group_obj = rb_hash_aref(hash, ID2SYM(rb_intern("group")));
      stream_obj = rb_hash_aref(hash, ID2SYM(rb_intern("stream")));
    } else {
      // Positional params
      group_obj = argv[3];
      if (argc > 4) {
        stream_obj = argv[4];
      }
    }
  }
  
  auto group = ruby_to_group(group_obj);
  auto stream_or_device = get_stream_or_device(stream_obj);
  
  mx::array result = mx::distributed::recv(shape, dtype, src, group, stream_or_device);
  return wrap_array(result);
}

static VALUE distributed_recv_like(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..4)", argc);
  }
  
  mx::array arr = get_array_or_scalar(argv[0]);
  int src = NUM2INT(argv[1]);
  
  VALUE group_obj = Qnil;
  VALUE stream_obj = Qnil;
  
  if (argc > 2) {
    // Check if argc == 3 with named params
    if (argc == 3 && rb_obj_is_kind_of(argv[2], rb_cHash)) {
      VALUE hash = argv[2];
      group_obj = rb_hash_aref(hash, ID2SYM(rb_intern("group")));
      stream_obj = rb_hash_aref(hash, ID2SYM(rb_intern("stream")));
    } else {
      // Positional params
      group_obj = argv[2];
      if (argc > 3) {
        stream_obj = argv[3];
      }
    }
  }
  
  auto group = ruby_to_group(group_obj);
  auto stream_or_device = get_stream_or_device(stream_obj);
  
  mx::array result = mx::distributed::recv_like(arr, src, group, stream_or_device);
  return wrap_array(result);
}

// Initialize distributed module
void init_distributed(VALUE module) {
  // Define Group class
  VALUE group_class = rb_define_class_under(module, "Group", rb_cObject);
  rb_define_alloc_func(group_class, group_alloc);
  rb_define_method(group_class, "initialize", RUBY_METHOD_FUNC(group_initialize), 2);
  rb_define_method(group_class, "rank", RUBY_METHOD_FUNC(group_rank), 0);
  rb_define_method(group_class, "size", RUBY_METHOD_FUNC(group_size), 0);
  rb_define_method(group_class, "split", RUBY_METHOD_FUNC(group_split), -1);
  
  // Define module functions
  rb_define_module_function(module, "is_available", RUBY_METHOD_FUNC(distributed_is_available), 0);
  rb_define_module_function(module, "init", RUBY_METHOD_FUNC(distributed_init), -1);
  rb_define_module_function(module, "all_sum", RUBY_METHOD_FUNC(distributed_all_sum), -1);
  rb_define_module_function(module, "all_gather", RUBY_METHOD_FUNC(distributed_all_gather), -1);
  rb_define_module_function(module, "send", RUBY_METHOD_FUNC(distributed_send), -1);
  rb_define_module_function(module, "recv", RUBY_METHOD_FUNC(distributed_recv), -1);
  rb_define_module_function(module, "recv_like", RUBY_METHOD_FUNC(distributed_recv_like), -1);
} 