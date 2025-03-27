#include <ruby.h>
#include "mlx/transforms.h"
#include "mlx/compile.h" // for compile / disable_compile / enable_compile
#include "mlx/ops.h"
#include "mlx/utils.h"
#include "utils.h"

namespace mx = mlx::core;

/*
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
*/

// Helper to extract Ruby array into C++ vector
static std::vector<int> ruby_array_to_vector(VALUE arr) {
  Check_Type(arr, T_ARRAY);
  
  std::vector<int> cpp_arr;
  for (long i = 0; i < RARRAY_LEN(arr); i++) {
    VALUE item = rb_ary_entry(arr, i);
    cpp_arr.push_back(NUM2INT(item));
  }
  
  return cpp_arr;
}

/*
// Helper to extract Stream or Device from Ruby VALUE
static std::variant<std::monostate, mx::Stream, mx::Device> get_stream_or_device(VALUE obj) {
  if (NIL_P(obj)) {
    return std::monostate{}; // Default empty stream/device
  }
  
  // Check if it's a Stream object
  if (rb_obj_is_kind_of(obj, rb_path2class("MLX::Core::Stream"))) {
    mx::Stream* stream_ptr;
    Data_Get_Struct(obj, mx::Stream, stream_ptr);
    return *stream_ptr;
  }
  
  // Check if it's a Device object
  if (rb_obj_is_kind_of(obj, rb_path2class("MLX::Core::Device"))) {
    mx::Device* device_ptr;
    Data_Get_Struct(obj, mx::Device, device_ptr);
    return *device_ptr;
  }
  
  rb_raise(rb_eTypeError, "Expected Stream or Device object");
  return std::monostate{}; // Never reached
}
*/

// Transforms module methods
static VALUE transforms_reshape(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  VALUE arr = argv[0];
  VALUE shape = argv[1];
  VALUE stream_obj = argc > 2 ? argv[2] : Qnil;
  
  mx::array& a = get_array(arr);
  std::vector<int> new_shape = ruby_array_to_vector(shape);
  auto stream = get_stream_or_device(stream_obj);
  
  mx::array result = mx::reshape(a, new_shape, stream);
  return wrap_array(result);
}

static VALUE transforms_transpose(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..3)", argc);
  }
  
  VALUE arr = argv[0];
  VALUE axes = argc > 1 ? argv[1] : Qnil;
  VALUE stream_obj = argc > 2 ? argv[2] : Qnil;
  
  mx::array& a = get_array(arr);
  auto stream = get_stream_or_device(stream_obj);
  
  if (NIL_P(axes)) {
    mx::array result = mx::transpose(a, stream);
    return wrap_array(result);
  } else {
    std::vector<int> perm = ruby_array_to_vector(axes);
    mx::array result = mx::transpose(a, perm, stream);
    return wrap_array(result);
  }
}

static VALUE transforms_squeeze(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..3)", argc);
  }
  
  VALUE arr = argv[0];
  VALUE axes = argc > 1 ? argv[1] : Qnil;
  VALUE stream_obj = argc > 2 ? argv[2] : Qnil;
  
  mx::array& a = get_array(arr);
  auto stream = get_stream_or_device(stream_obj);
  
  if (NIL_P(axes)) {
    mx::array result = mx::squeeze(a, stream);
    return wrap_array(result);
  } else if (RB_TYPE_P(axes, T_FIXNUM)) {
    int axis = NUM2INT(axes);
    mx::array result = mx::squeeze(a, axis, stream);
    return wrap_array(result);
  } else {
    std::vector<int> axes_vec = ruby_array_to_vector(axes);
    mx::array result = mx::squeeze(a, axes_vec, stream);
    return wrap_array(result);
  }
}

static VALUE transforms_expand_dims(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  VALUE arr = argv[0];
  VALUE axis = argv[1];
  VALUE stream_obj = argc > 2 ? argv[2] : Qnil;
  
  mx::array& a = get_array(arr);
  int ax = NUM2INT(axis);
  auto stream = get_stream_or_device(stream_obj);
  
  mx::array result = mx::expand_dims(a, ax, stream);
  return wrap_array(result);
}

static VALUE transforms_broadcast_to(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  VALUE arr = argv[0];
  VALUE shape = argv[1];
  VALUE stream_obj = argc > 2 ? argv[2] : Qnil;
  
  mx::array& a = get_array(arr);
  std::vector<int> new_shape = ruby_array_to_vector(shape);
  auto stream = get_stream_or_device(stream_obj);
  
  mx::array result = mx::broadcast_to(a, new_shape, stream);
  return wrap_array(result);
}

static VALUE transforms_pad(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..4)", argc);
  }
  
  VALUE arr = argv[0];
  VALUE pad_width = argv[1];
  VALUE value = argc > 2 ? argv[2] : Qnil;
  VALUE stream_obj = argc > 3 ? argv[3] : Qnil;
  
  mx::array& a = get_array(arr);
  auto stream = get_stream_or_device(stream_obj);
  
  // Convert pad_width from Ruby structure to vector<pair<int, int>>
  Check_Type(pad_width, T_ARRAY);
  std::vector<std::pair<int, int>> cpp_pad_width;
  
  for (long i = 0; i < RARRAY_LEN(pad_width); i++) {
    VALUE inner_arr = rb_ary_entry(pad_width, i);
    Check_Type(inner_arr, T_ARRAY);
    
    if (RARRAY_LEN(inner_arr) != 2) {
      rb_raise(rb_eArgError, "Each padding element must be a pair [low, high]");
    }
    
    int low = NUM2INT(rb_ary_entry(inner_arr, 0));
    int high = NUM2INT(rb_ary_entry(inner_arr, 1));
    cpp_pad_width.push_back(std::make_pair(low, high));
  }
  
  // Create the pad value
  mx::array pad_value = mx::array(0.0f);  // Default initialized with 0.0
  if (NIL_P(value)) {
    // Use the default pad_value already set
  } else if (RB_TYPE_P(value, T_FLOAT) || RB_TYPE_P(value, T_FIXNUM)) {
    pad_value = mx::array(NUM2DBL(value));
  } else {
    pad_value = get_array(value);
  }
  
  // Use the correct version of pad with std::vector<std::pair<int, int>>
  mx::array result = mx::pad(a, cpp_pad_width, pad_value, "constant", stream);
  return wrap_array(result);
}

static VALUE transforms_split(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..4)", argc);
  }
  
  VALUE arr = argv[0];
  VALUE num_or_indices = argv[1];
  VALUE axis = argc > 2 ? argv[2] : Qnil;
  VALUE stream_obj = argc > 3 ? argv[3] : Qnil;
  
  mx::array& a = get_array(arr);
  int ax = NIL_P(axis) ? 0 : NUM2INT(axis);
  auto stream = get_stream_or_device(stream_obj);
  
  std::vector<mx::array> result;
  
  if (RB_TYPE_P(num_or_indices, T_FIXNUM)) {
    int num = NUM2INT(num_or_indices);
    result = mx::split(a, num, ax, stream);
  } else {
    std::vector<int> indices = ruby_array_to_vector(num_or_indices);
    result = mx::split(a, indices, ax, stream);
  }
  
  VALUE rb_result = rb_ary_new();
  for (const auto& res : result) {
    rb_ary_push(rb_result, wrap_array(res));
  }
  
  return rb_result;
}

static VALUE transforms_concatenate(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..3)", argc);
  }
  
  VALUE arrays = argv[0];
  VALUE axis = argc > 1 ? argv[1] : Qnil;
  VALUE stream_obj = argc > 2 ? argv[2] : Qnil;
  
  Check_Type(arrays, T_ARRAY);
  
  std::vector<mx::array> cpp_arrays;
  for (long i = 0; i < RARRAY_LEN(arrays); i++) {
    VALUE arr = rb_ary_entry(arrays, i);
    cpp_arrays.push_back(get_array(arr));
  }
  
  int ax = NIL_P(axis) ? 0 : NUM2INT(axis);
  auto stream = get_stream_or_device(stream_obj);
  
  mx::array result = mx::concatenate(cpp_arrays, ax, stream);
  return wrap_array(result);
}

static VALUE transforms_stack(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..3)", argc);
  }
  
  VALUE arrays = argv[0];
  VALUE axis = argc > 1 ? argv[1] : Qnil;
  VALUE stream_obj = argc > 2 ? argv[2] : Qnil;
  
  Check_Type(arrays, T_ARRAY);
  
  std::vector<mx::array> cpp_arrays;
  for (long i = 0; i < RARRAY_LEN(arrays); i++) {
    VALUE arr = rb_ary_entry(arrays, i);
    cpp_arrays.push_back(get_array(arr));
  }
  
  int ax = NIL_P(axis) ? 0 : NUM2INT(axis);
  auto stream = get_stream_or_device(stream_obj);
  
  mx::array result = mx::stack(cpp_arrays, ax, stream);
  return wrap_array(result);
}

static VALUE transforms_tile(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  VALUE arr = argv[0];
  VALUE repeats = argv[1];
  VALUE stream_obj = argc > 2 ? argv[2] : Qnil;
  
  mx::array& a = get_array(arr);
  std::vector<int> cpp_repeats = ruby_array_to_vector(repeats);
  auto stream = get_stream_or_device(stream_obj);
  
  mx::array result = mx::tile(a, cpp_repeats, stream);
  return wrap_array(result);
}

static VALUE transforms_repeat(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..4)", argc);
  }
  
  VALUE arr = argv[0];
  VALUE repeats = argv[1];
  VALUE axis = argc > 2 ? argv[2] : Qnil;
  VALUE stream_obj = argc > 3 ? argv[3] : Qnil;
  
  mx::array& a = get_array(arr);
  int rep = NUM2INT(repeats);
  auto stream = get_stream_or_device(stream_obj);
  
  // Initialize with a valid value instead of using default constructor
  mx::array result = a; // Temporary initialization
  
  if (NIL_P(axis)) {
    result = mx::repeat(a, rep, stream);
  } else {
    int ax = NUM2INT(axis);
    result = mx::repeat(a, rep, ax, stream);
  }
  
  return wrap_array(result);
}

static VALUE transforms_moveaxis(int argc, VALUE* argv, VALUE self) {
  if (argc < 3 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 3..4)", argc);
  }
  
  VALUE arr = argv[0];
  VALUE source = argv[1];
  VALUE destination = argv[2];
  VALUE stream_obj = argc > 3 ? argv[3] : Qnil;
  
  mx::array& a = get_array(arr);
  auto stream = get_stream_or_device(stream_obj);
  
  // Initialize with a valid value instead of using default constructor
  mx::array result = a; // Temporary initialization
  
  if (RB_TYPE_P(source, T_FIXNUM) && RB_TYPE_P(destination, T_FIXNUM)) {
    int src = NUM2INT(source);
    int dst = NUM2INT(destination);
    result = mx::moveaxis(a, src, dst, stream);
  } else {
    std::vector<int> src_vec = ruby_array_to_vector(source);
    std::vector<int> dst_vec = ruby_array_to_vector(destination);
    
    // Use the moveaxis version that takes vector parameters
    if (src_vec.size() == dst_vec.size()) {
      result = mx::array(a); // Start with a copy
      for (size_t i = 0; i < src_vec.size(); i++) {
        result = mx::moveaxis(result, src_vec[i], dst_vec[i], stream);
      }
    } else {
      rb_raise(rb_eArgError, "Source and destination axes must have the same length");
    }
  }
  
  return wrap_array(result);
}

// Eval functions - with proper stream support
static VALUE transforms_eval(int argc, VALUE* argv, VALUE self) {
  if (argc < 1) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected at least 1)", argc);
  }
  
  // Process a batch of arrays to support mx.eval(a, b, c) syntax like in Python
  std::vector<mx::array> arrays;
  for (int i = 0; i < argc; i++) {
    VALUE arr = argv[i];
    if (!rb_obj_is_kind_of(arr, rb_path2class("MLX::Core::Array"))) {
      rb_raise(rb_eTypeError, "Expected MLX::Core::Array objects");
    }
    arrays.push_back(get_array(arr));
  }
  
  mx::eval(arrays);
  
  // Return the original array(s) - if just one array, return it directly
  return argc == 1 ? argv[0] : rb_ary_new4(argc, argv);
}

/*
 * async_eval(*arrays)
 *
 * Asynchronously evaluate one or more arrays.
 */
static VALUE transforms_async_eval(int argc, VALUE* argv, VALUE self) {
  if (argc < 1) {
    rb_raise(rb_eArgError, "wrong number of arguments for async_eval (given %d, expected >=1)", argc);
  }

  std::vector<mx::array> arrays;
  for (int i = 0; i < argc; i++) {
    VALUE arr = argv[i];
    if (!rb_obj_is_kind_of(arr, rb_path2class("MLX::Core::Array"))) {
      rb_raise(rb_eTypeError, "async_eval expects only MLX::Core::Array objects");
    }
    arrays.push_back(get_array(arr));
  }
  mx::async_eval(arrays);

  // Return original array(s)
  return argc == 1 ? argv[0] : rb_ary_new4(argc, argv);
}

/*
 * jvp(fun, primals, tangents)
 *   Currently a stub. You must implement logic to:
 *   - Flatten the 'primals' and 'tangents'
 *   - Call mx::jvp(...) on a suitable lambda
 */
static VALUE transforms_jvp(int argc, VALUE* argv, VALUE self) {
  rb_notimplement();
  return Qnil;
}

/*
 * vjp(fun, primals, cotangents)
 */
static VALUE transforms_vjp(int argc, VALUE* argv, VALUE self) {
  rb_notimplement();
  return Qnil;
}

/*
 * vmap(fun, in_axes=0, out_axes=0)
 */
static VALUE transforms_vmap(int argc, VALUE* argv, VALUE self) {
  rb_notimplement();
  return Qnil;
}

/*
 * compile(fun, inputs=nil, outputs=nil, shapeless=false)
 */
static VALUE transforms_compile(int argc, VALUE* argv, VALUE self) {
  rb_notimplement();
  return Qnil;
}

/*
 * disable_compile()
 */
static VALUE transforms_disable_compile(VALUE self) {
  mx::disable_compile();
  return Qnil;
}

/*
 * enable_compile()
 */
static VALUE transforms_enable_compile(VALUE self) {
  mx::enable_compile();
  return Qnil;
}

// Advanced transforms - placeholder implementations for now
static VALUE transforms_checkpoint(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..2)", argc);
  }
  
  rb_notimplement();
  return Qnil;
}

// value_and_grad(fun, argnums=nil, argnames=[])
static VALUE transforms_value_and_grad(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..3)", argc);
  }
  
  rb_notimplement();
  return Qnil;
}

// grad(fun, argnums=nil, argnames=[])
static VALUE transforms_grad(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..3)", argc);
  }
  
  rb_notimplement();
  return Qnil;
}

static VALUE transforms_stop_gradient(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..2)", argc);
  }
  
  VALUE arr = argv[0];
  VALUE stream_obj = argc > 1 ? argv[1] : Qnil;
  
  mx::array& a = get_array(arr);
  auto stream = get_stream_or_device(stream_obj);
  
  mx::array result = mx::stop_gradient(a, stream);
  return wrap_array(result);
}

static VALUE transforms_eval_batch(int argc, VALUE* argv, VALUE self) {
  if (argc != 1) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1)", argc);
  }
  
  VALUE arrays = argv[0];
  Check_Type(arrays, T_ARRAY);
  
  std::vector<mx::array> cpp_arrays;
  for (long i = 0; i < RARRAY_LEN(arrays); i++) {
    VALUE arr = rb_ary_entry(arrays, i);
    if (!rb_obj_is_kind_of(arr, rb_path2class("MLX::Core::Array"))) {
      rb_raise(rb_eTypeError, "Expected all elements to be MLX::Core::Array objects");
    }
    cpp_arrays.push_back(get_array(arr));
  }
  
  mx::eval(cpp_arrays);
  return arrays;
}

/*
 * class MLX::Core::CustomFunction
 *
 * This mirrors the Python-class `custom_function` (PyCustomFunction).
 * We'll stub out the behaviors so the interface is present. You can fill
 * in the details as needed.
 */
static VALUE cCustomFunction = Qnil;

// CustomFunction#initialize(fun)
static VALUE custom_function_initialize(VALUE self, VALUE fun) {
  // Store the 'fun' in an instance variable to call later
  rb_iv_set(self, "@fun", fun);
  return self;
}

// CustomFunction#call(*args, **kwargs)
static VALUE custom_function_call(int argc, VALUE* argv, VALUE self) {
  rb_notimplement();
  return Qnil;
}

// CustomFunction#vjp(f)
static VALUE custom_function_vjp(VALUE self, VALUE transform_func) {
  // store it to @vjp_func for example
  rb_iv_set(self, "@vjp_func", transform_func);
  return self;
}

// CustomFunction#jvp(f)
static VALUE custom_function_jvp(VALUE self, VALUE transform_func) {
  // store it to @jvp_func for example
  rb_iv_set(self, "@jvp_func", transform_func);
  return self;
}

// CustomFunction#vmap(f)
static VALUE custom_function_vmap(VALUE self, VALUE transform_func) {
  // store it to @vmap_func
  rb_iv_set(self, "@vmap_func", transform_func);
  return self;
}

// Initialize transforms module
void init_transforms(VALUE module) {
  // Define module functions
  // Array shape manipulation functions
  rb_define_module_function(module, "reshape", RUBY_METHOD_FUNC(transforms_reshape), -1);
  rb_define_module_function(module, "transpose", RUBY_METHOD_FUNC(transforms_transpose), -1);
  rb_define_module_function(module, "squeeze", RUBY_METHOD_FUNC(transforms_squeeze), -1);
  rb_define_module_function(module, "expand_dims", RUBY_METHOD_FUNC(transforms_expand_dims), -1);
  rb_define_module_function(module, "broadcast_to", RUBY_METHOD_FUNC(transforms_broadcast_to), -1);
  rb_define_module_function(module, "pad", RUBY_METHOD_FUNC(transforms_pad), -1);
  rb_define_module_function(module, "split", RUBY_METHOD_FUNC(transforms_split), -1);
  rb_define_module_function(module, "concatenate", RUBY_METHOD_FUNC(transforms_concatenate), -1);
  rb_define_module_function(module, "stack", RUBY_METHOD_FUNC(transforms_stack), -1);
  rb_define_module_function(module, "tile", RUBY_METHOD_FUNC(transforms_tile), -1);
  rb_define_module_function(module, "repeat", RUBY_METHOD_FUNC(transforms_repeat), -1);
  rb_define_module_function(module, "moveaxis", RUBY_METHOD_FUNC(transforms_moveaxis), -1);
  
  // Gradient and function transformation functions
  rb_define_module_function(module, "checkpoint", RUBY_METHOD_FUNC(transforms_checkpoint), -1);
  rb_define_module_function(module, "value_and_grad", RUBY_METHOD_FUNC(transforms_value_and_grad), -1);
  rb_define_module_function(module, "grad", RUBY_METHOD_FUNC(transforms_grad), -1);
  // The newly added stubs
  rb_define_module_function(module, "jvp", RUBY_METHOD_FUNC(transforms_jvp), -1);
  rb_define_module_function(module, "vjp", RUBY_METHOD_FUNC(transforms_vjp), -1);
  rb_define_module_function(module, "vmap", RUBY_METHOD_FUNC(transforms_vmap), -1);
  rb_define_module_function(module, "compile", RUBY_METHOD_FUNC(transforms_compile), -1);
  rb_define_module_function(module, "disable_compile", RUBY_METHOD_FUNC(transforms_disable_compile), 0);
  rb_define_module_function(module, "enable_compile", RUBY_METHOD_FUNC(transforms_enable_compile), 0);
  
  rb_define_module_function(module, "stop_gradient", RUBY_METHOD_FUNC(transforms_stop_gradient), -1);
  
  // Evaluation functions
  rb_define_module_function(module, "eval", RUBY_METHOD_FUNC(transforms_eval), -1);
  rb_define_module_function(module, "eval_batch", RUBY_METHOD_FUNC(transforms_eval_batch), -1);
  rb_define_module_function(module, "async_eval", RUBY_METHOD_FUNC(transforms_async_eval), -1);
  
  // Define class MLX::Core::CustomFunction for the custom_function decorator
  // (Under the same top-level module, or under "MLX::Core" if that's how your
  //  module structure is laid out; adjust as needed.)
  VALUE cCore = rb_path2class("MLX::Core");
  cCustomFunction = rb_define_class_under(cCore, "CustomFunction", rb_cObject);
  
  rb_define_method(cCustomFunction, "initialize", RUBY_METHOD_FUNC(custom_function_initialize), 1);
  // we let "def call(*args)" take -1
  rb_define_method(cCustomFunction, "call", RUBY_METHOD_FUNC(custom_function_call), -1);
  rb_define_method(cCustomFunction, "vjp", RUBY_METHOD_FUNC(custom_function_vjp), 1);
  rb_define_method(cCustomFunction, "jvp", RUBY_METHOD_FUNC(custom_function_jvp), 1);
  rb_define_method(cCustomFunction, "vmap", RUBY_METHOD_FUNC(custom_function_vmap), 1);
} 