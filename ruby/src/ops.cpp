#include <ruby.h>
#include <vector>
#include "mlx/ops.h"
#include "mlx_func.h"

namespace mx = mlx::core;

// ---------------------------------------------------------------------------
// NEW HELPER MACROS to shorten repeated code in these new methods
// ---------------------------------------------------------------------------
// Use these to parse either scalar or array from a VALUE:
#define SCALAR_OR_ARRAY(var, rb_val)                       \
  ScalarOrArray var##_soa = value_to_scalar_or_array(rb_val); \
  mx::array var = var##_soa.is_array() ? var##_soa.array() : mx::array(var##_soa.scalar());

#define GET_STREAM(var, rb_val) \
  mx::StreamOrDevice var = get_stream_or_device(rb_val);

#define CHECK_ARITY(func_name, min_args, max_args)                                \
  if (argc < (min_args) || argc > (max_args)) {                                   \
    rb_raise(rb_eArgError,                                                       \
             "%s: wrong number of arguments (given %%d, expected %%d..%%d)",      \
             func_name, argc, (min_args), (max_args));                           \
  }

#define RETURN_ARRAY(expr) \
  return wrap_array(expr);

// ScalarOrArray class for handling both numeric values and arrays
class ScalarOrArray {
private:
  bool is_array_;
  double scalar_value_;
  mx::array array_value_;

public:
  ScalarOrArray(double scalar) : is_array_(false), scalar_value_(scalar), array_value_(scalar) {}
  ScalarOrArray(const mx::array& arr) : is_array_(true), scalar_value_(0.0), array_value_(arr) {}

  bool is_array() const { return is_array_; }
  bool is_scalar() const { return !is_array_; }
  double scalar() const { return scalar_value_; }
  const mx::array& array() const { return array_value_; }
};

// Helper function to convert Ruby VALUE to ScalarOrArray
static ScalarOrArray value_to_scalar_or_array(VALUE val) {
  if (rb_obj_is_kind_of(val, rb_path2class("MLX::Core::Array"))) {
    mx::array* arr_ptr;
    Data_Get_Struct(val, mx::array, arr_ptr);
    return ScalarOrArray(*arr_ptr);
  } else if (FIXNUM_P(val) || RB_FLOAT_TYPE_P(val)) {
    return ScalarOrArray(NUM2DBL(val));
  } else {
    rb_raise(rb_eTypeError, "Expected Array or numeric value");
    return ScalarOrArray(0.0); // never reached
  }
}

// Helper function to extract mx::array from Ruby VALUE
static mx::array& get_array(VALUE obj) {
  mx::array* arr_ptr;
  Data_Get_Struct(obj, mx::array, arr_ptr);
  return *arr_ptr;
}

// Helper function to wrap mx::array into Ruby VALUE
static VALUE wrap_array(const mx::array& arr) {
  return Data_Wrap_Struct(rb_path2class("MLX::Core::Array"), 0, nullptr, new mx::array(arr));
}

// Helper to extract Stream or Device from Ruby VALUE
static mx::StreamOrDevice get_stream_or_device(VALUE obj) {
  if (NIL_P(obj)) {
    return mx::StreamOrDevice{}; // Default empty stream/device
  }
  
  // Check if it's a Stream object
  if (rb_obj_is_kind_of(obj, rb_path2class("MLX::Stream"))) {
    mx::Stream* stream_ptr;
    Data_Get_Struct(obj, mx::Stream, stream_ptr);
    return *stream_ptr;
  }
  
  // Check if it's a Device object
  if (rb_obj_is_kind_of(obj, rb_path2class("MLX::Device"))) {
    mx::Device* device_ptr;
    Data_Get_Struct(obj, mx::Device, device_ptr);
    return *device_ptr;
  }
  
  rb_raise(rb_eTypeError, "Expected Stream or Device object");
  return mx::StreamOrDevice{}; // Never reached
}

// Convert Ruby array to std::vector<int> for shape
static std::vector<int> ruby_array_to_shape(VALUE shape_array) {
  Check_Type(shape_array, T_ARRAY);
  
  std::vector<int> shape;
  for (long i = 0; i < RARRAY_LEN(shape_array); i++) {
    VALUE item = rb_ary_entry(shape_array, i);
    shape.push_back(NUM2INT(item));
  }
  
  return shape;
}

// Helper function to convert Ruby integer to mx::Dtype
static mx::Dtype int_to_dtype(int dtype_val) {
  switch (dtype_val) {
    case 0: return mx::bool_;
    case 1: return mx::uint8;
    case 2: return mx::uint16;
    case 3: return mx::uint32;
    case 4: return mx::uint64;
    case 5: return mx::int8;
    case 6: return mx::int16;
    case 7: return mx::int32;
    case 8: return mx::int64;
    case 9: return mx::float16;
    case 10: return mx::float32;
    case 11: return mx::float64;
    case 12: return mx::bfloat16;
    case 13: return mx::complex64;
    default: return mx::float32; // Default to float32
  }
}

// Array creation operations
static VALUE ops_zeros(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  VALUE shape = argv[0];
  VALUE dtype = argv[1];
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  
  std::vector<int> cpp_shape = ruby_array_to_shape(shape);
  mx::Dtype d = int_to_dtype(NUM2INT(dtype));
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  // Create zeros array with stream
  mx::array result = mx::zeros(cpp_shape, d, stream);
  
  return wrap_array(result);
}

static VALUE ops_ones(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  VALUE shape = argv[0];
  VALUE dtype = argv[1];
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  
  std::vector<int> cpp_shape = ruby_array_to_shape(shape);
  mx::Dtype d = int_to_dtype(NUM2INT(dtype));
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  // Create ones array with stream
  mx::array result = mx::ones(cpp_shape, d, stream);
  
  return wrap_array(result);
}

static VALUE ops_full(int argc, VALUE* argv, VALUE self) {
  if (argc < 3 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 3..4)", argc);
  }
  
  VALUE shape = argv[0];
  VALUE fill_value = argv[1];
  VALUE dtype = argv[2];
  VALUE stream_val = (argc > 3) ? argv[3] : Qnil;
  
  std::vector<int> cpp_shape = ruby_array_to_shape(shape);
  mx::Dtype d = int_to_dtype(NUM2INT(dtype));
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  // Get the fill value
  ScalarOrArray fill = value_to_scalar_or_array(fill_value);
  
  // Create full array with stream
  mx::array result = fill.is_scalar() ? 
    mx::full(cpp_shape, mx::array(fill.scalar()), d, stream) :
    mx::full(cpp_shape, fill.array(), d, stream);
  
  return wrap_array(result);
}

static VALUE ops_arange(int argc, VALUE* argv, VALUE self) {
  if (argc < 4 || argc > 5) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 4..5)", argc);
  }
  
  VALUE start = argv[0];
  VALUE stop = argv[1];
  VALUE step = argv[2];
  VALUE dtype = argv[3];
  VALUE stream_val = (argc > 4) ? argv[4] : Qnil;
  
  double start_val = NUM2DBL(start);
  double stop_val = NUM2DBL(stop);
  double step_val = NUM2DBL(step);
  
  mx::Dtype d = int_to_dtype(NUM2INT(dtype));
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::arange(start_val, stop_val, step_val, d, stream);
  
  return wrap_array(result);
}

static VALUE ops_identity(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  VALUE n = argv[0];
  VALUE dtype = argv[1];
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  
  int n_val = NUM2INT(n);
  mx::Dtype d = int_to_dtype(NUM2INT(dtype));
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::identity(n_val, d, stream);
  
  return wrap_array(result);
}

static VALUE ops_eye(int argc, VALUE* argv, VALUE self) {
  if (argc < 4 || argc > 5) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 4..5)", argc);
  }
  
  VALUE n = argv[0];
  VALUE m = argv[1];
  VALUE k = argv[2];
  VALUE dtype = argv[3];
  VALUE stream_val = (argc > 4) ? argv[4] : Qnil;
  
  int n_val = NUM2INT(n);
  int m_val = NIL_P(m) ? n_val : NUM2INT(m);
  int k_val = NUM2INT(k);
  mx::Dtype d = int_to_dtype(NUM2INT(dtype));
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::eye(n_val, m_val, k_val, d, stream);
  
  return wrap_array(result);
}

// Array manipulation operations
static VALUE ops_reshape(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  VALUE arr = argv[0];
  VALUE shape = argv[1];
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  
  mx::array& a = get_array(arr);
  std::vector<int> new_shape = ruby_array_to_shape(shape);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::reshape(a, new_shape, stream);
  return wrap_array(result);
}

static VALUE ops_flatten(int argc, VALUE* argv, VALUE self) {
  if (argc < 3 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 3..4)", argc);
  }
  
  VALUE arr = argv[0];
  VALUE start_axis = argv[1];
  VALUE end_axis = argv[2];
  VALUE stream_val = (argc > 3) ? argv[3] : Qnil;
  
  mx::array& a = get_array(arr);
  int start = NUM2INT(start_axis);
  int end = NUM2INT(end_axis);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::flatten(a, start, end, stream);
  return wrap_array(result);
}

static VALUE ops_squeeze(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..3)", argc);
  }
  
  mx::array& a = get_array(argv[0]);
  mx::array result = mx::zeros({1}, mx::float32); // Default initialization
  mx::StreamOrDevice stream;
  
  // Handle stream parameter which can be in different positions based on arguments
  VALUE stream_val = Qnil;
  VALUE axis_val = Qnil;
  
  if (argc == 2) {
    // Could be either axis or stream
    if (rb_obj_is_kind_of(argv[1], rb_path2class("MLX::Stream")) || 
        rb_obj_is_kind_of(argv[1], rb_path2class("MLX::Device"))) {
      stream_val = argv[1];
    } else {
      axis_val = argv[1];
    }
  } else if (argc == 3) {
    // Both axis and stream
    axis_val = argv[1];
    stream_val = argv[2];
  }
  
  stream = get_stream_or_device(stream_val);
  
  if (NIL_P(axis_val)) {
    // No axis specified, squeeze all dimensions of size 1
    result = mx::squeeze(a, stream);
  } else if (RB_TYPE_P(axis_val, T_FIXNUM)) {
    // Single axis
    int axis = NUM2INT(axis_val);
    result = mx::squeeze(a, axis, stream);
  } else if (RB_TYPE_P(axis_val, T_ARRAY)) {
    // Multiple axes
    std::vector<int> axes = ruby_array_to_shape(axis_val);
    result = mx::squeeze(a, axes, stream);
  } else {
    rb_raise(rb_eTypeError, "axis must be an integer or array");
  }
  
  return wrap_array(result);
}

static VALUE ops_expand_dims(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  VALUE arr = argv[0];
  VALUE axis = argv[1];
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  
  mx::array& a = get_array(arr);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::zeros({1}, mx::float32); // Default initialization
  if (RB_TYPE_P(axis, T_FIXNUM)) {
    int axis_val = NUM2INT(axis);
    result = mx::expand_dims(a, axis_val, stream);
  } else if (RB_TYPE_P(axis, T_ARRAY)) {
    std::vector<int> axes = ruby_array_to_shape(axis);
    result = mx::expand_dims(a, axes, stream);
  } else {
    rb_raise(rb_eTypeError, "axis must be an integer or array");
  }
  
  return wrap_array(result);
}

// Element-wise operations
static VALUE ops_abs(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..2)", argc);
  }
  
  VALUE arr = argv[0];
  VALUE stream_val = (argc > 1) ? argv[1] : Qnil;
  
  ScalarOrArray a = value_to_scalar_or_array(arr);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = a.is_array() ? 
    mx::abs(a.array(), stream) : 
    mx::abs(mx::array(a.scalar()), stream);
  
  return wrap_array(result);
}

static VALUE ops_sign(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..2)", argc);
  }
  
  VALUE arr = argv[0];
  VALUE stream_val = (argc > 1) ? argv[1] : Qnil;
  
  ScalarOrArray a = value_to_scalar_or_array(arr);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = a.is_array() ?
    mx::sign(a.array(), stream) :
    mx::sign(mx::array(a.scalar()), stream);
  
  return wrap_array(result);
}

static VALUE ops_negative(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..2)", argc);
  }
  
  VALUE arr = argv[0];
  VALUE stream_val = (argc > 1) ? argv[1] : Qnil;
  
  ScalarOrArray a = value_to_scalar_or_array(arr);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = a.is_array() ?
    mx::negative(a.array(), stream) :
    mx::negative(mx::array(a.scalar()), stream);
  
  return wrap_array(result);
}

// Basic operations
static VALUE ops_add(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  VALUE a = argv[0];
  VALUE b = argv[1];
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  
  ScalarOrArray arg_a = value_to_scalar_or_array(a);
  ScalarOrArray arg_b = value_to_scalar_or_array(b);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::zeros({1}, mx::float32); // Default initialization
  
  if (arg_a.is_scalar() && arg_b.is_scalar()) {
    result = mx::array(arg_a.scalar() + arg_b.scalar());
  } else if (arg_a.is_scalar() && arg_b.is_array()) {
    result = mx::add(mx::array(arg_a.scalar()), arg_b.array(), stream);
  } else if (arg_a.is_array() && arg_b.is_scalar()) {
    result = mx::add(arg_a.array(), mx::array(arg_b.scalar()), stream);
  } else {
    result = mx::add(arg_a.array(), arg_b.array(), stream);
  }
  
  return wrap_array(result);
}

static VALUE ops_subtract(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  VALUE a = argv[0];
  VALUE b = argv[1];
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  
  ScalarOrArray arg_a = value_to_scalar_or_array(a);
  ScalarOrArray arg_b = value_to_scalar_or_array(b);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::zeros({1}, mx::float32); // Default initialization
  
  if (arg_a.is_scalar() && arg_b.is_scalar()) {
    result = mx::array(arg_a.scalar() - arg_b.scalar());
  } else if (arg_a.is_scalar() && arg_b.is_array()) {
    result = mx::subtract(mx::array(arg_a.scalar()), arg_b.array(), stream);
  } else if (arg_a.is_array() && arg_b.is_scalar()) {
    result = mx::subtract(arg_a.array(), mx::array(arg_b.scalar()), stream);
  } else {
    result = mx::subtract(arg_a.array(), arg_b.array(), stream);
  }
  
  return wrap_array(result);
}

static VALUE ops_multiply(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  VALUE a = argv[0];
  VALUE b = argv[1];
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  
  ScalarOrArray arg_a = value_to_scalar_or_array(a);
  ScalarOrArray arg_b = value_to_scalar_or_array(b);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::zeros({1}, mx::float32); // Default initialization
  
  if (arg_a.is_scalar() && arg_b.is_scalar()) {
    result = mx::array(arg_a.scalar() * arg_b.scalar());
  } else if (arg_a.is_scalar() && arg_b.is_array()) {
    result = mx::multiply(mx::array(arg_a.scalar()), arg_b.array(), stream);
  } else if (arg_a.is_array() && arg_b.is_scalar()) {
    result = mx::multiply(arg_a.array(), mx::array(arg_b.scalar()), stream);
  } else {
    result = mx::multiply(arg_a.array(), arg_b.array(), stream);
  }
  
  return wrap_array(result);
}

static VALUE ops_divide(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  VALUE a = argv[0];
  VALUE b = argv[1];
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  
  ScalarOrArray arg_a = value_to_scalar_or_array(a);
  ScalarOrArray arg_b = value_to_scalar_or_array(b);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::zeros({1}, mx::float32); // Default initialization
  
  if (arg_a.is_scalar() && arg_b.is_scalar()) {
    result = mx::array(arg_a.scalar() / arg_b.scalar());
  } else if (arg_a.is_scalar() && arg_b.is_array()) {
    result = mx::divide(mx::array(arg_a.scalar()), arg_b.array(), stream);
  } else if (arg_a.is_array() && arg_b.is_scalar()) {
    result = mx::divide(arg_a.array(), mx::array(arg_b.scalar()), stream);
  } else {
    result = mx::divide(arg_a.array(), arg_b.array(), stream);
  }
  
  return wrap_array(result);
}

// Comparison operations
static VALUE ops_equal(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  VALUE a = argv[0];
  VALUE b = argv[1];
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  
  ScalarOrArray arg_a = value_to_scalar_or_array(a);
  ScalarOrArray arg_b = value_to_scalar_or_array(b);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::zeros({1}, mx::bool_); // Default initialization with bool type
  
  if (arg_a.is_array() && arg_b.is_array()) {
    result = mx::equal(arg_a.array(), arg_b.array(), stream);
  } else if (arg_a.is_scalar() && arg_b.is_array()) {
    result = mx::equal(mx::array(arg_a.scalar()), arg_b.array(), stream);
  } else if (arg_a.is_array() && arg_b.is_scalar()) {
    result = mx::equal(arg_a.array(), mx::array(arg_b.scalar()), stream);
  } else {
    result = mx::array(arg_a.scalar() == arg_b.scalar());
  }
  
  return wrap_array(result);
}

static VALUE ops_not_equal(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  VALUE a = argv[0];
  VALUE b = argv[1];
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  
  ScalarOrArray arg_a = value_to_scalar_or_array(a);
  ScalarOrArray arg_b = value_to_scalar_or_array(b);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::zeros({1}, mx::bool_); // Default initialization with bool type
  
  if (arg_a.is_array() && arg_b.is_array()) {
    result = mx::not_equal(arg_a.array(), arg_b.array(), stream);
  } else if (arg_a.is_scalar() && arg_b.is_array()) {
    result = mx::not_equal(mx::array(arg_a.scalar()), arg_b.array(), stream);
  } else if (arg_a.is_array() && arg_b.is_scalar()) {
    result = mx::not_equal(arg_a.array(), mx::array(arg_b.scalar()), stream);
  } else {
    result = mx::array(arg_a.scalar() != arg_b.scalar());
  }
  
  return wrap_array(result);
}

static VALUE ops_greater(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  VALUE a = argv[0];
  VALUE b = argv[1];
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  
  ScalarOrArray arg_a = value_to_scalar_or_array(a);
  ScalarOrArray arg_b = value_to_scalar_or_array(b);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::zeros({1}, mx::bool_); // Default initialization with bool type
  
  if (arg_a.is_array() && arg_b.is_array()) {
    result = mx::greater(arg_a.array(), arg_b.array(), stream);
  } else if (arg_a.is_scalar() && arg_b.is_array()) {
    result = mx::greater(mx::array(arg_a.scalar()), arg_b.array(), stream);
  } else if (arg_a.is_array() && arg_b.is_scalar()) {
    result = mx::greater(arg_a.array(), mx::array(arg_b.scalar()), stream);
  } else {
    result = mx::array(arg_a.scalar() > arg_b.scalar());
  }
  
  return wrap_array(result);
}

static VALUE ops_greater_equal(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  VALUE a = argv[0];
  VALUE b = argv[1];
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  
  ScalarOrArray arg_a = value_to_scalar_or_array(a);
  ScalarOrArray arg_b = value_to_scalar_or_array(b);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::zeros({1}, mx::bool_); // Default initialization with bool type
  
  if (arg_a.is_array() && arg_b.is_array()) {
    result = mx::greater_equal(arg_a.array(), arg_b.array(), stream);
  } else if (arg_a.is_scalar() && arg_b.is_array()) {
    result = mx::greater_equal(mx::array(arg_a.scalar()), arg_b.array(), stream);
  } else if (arg_a.is_array() && arg_b.is_scalar()) {
    result = mx::greater_equal(arg_a.array(), mx::array(arg_b.scalar()), stream);
  } else {
    result = mx::array(arg_a.scalar() >= arg_b.scalar());
  }
  
  return wrap_array(result);
}

static VALUE ops_less(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  VALUE a = argv[0];
  VALUE b = argv[1];
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  
  ScalarOrArray arg_a = value_to_scalar_or_array(a);
  ScalarOrArray arg_b = value_to_scalar_or_array(b);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::zeros({1}, mx::bool_); // Default initialization with bool type
  
  if (arg_a.is_array() && arg_b.is_array()) {
    result = mx::less(arg_a.array(), arg_b.array(), stream);
  } else if (arg_a.is_scalar() && arg_b.is_array()) {
    result = mx::less(mx::array(arg_a.scalar()), arg_b.array(), stream);
  } else if (arg_a.is_array() && arg_b.is_scalar()) {
    result = mx::less(arg_a.array(), mx::array(arg_b.scalar()), stream);
  } else {
    result = mx::array(arg_a.scalar() < arg_b.scalar());
  }
  
  return wrap_array(result);
}

static VALUE ops_less_equal(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  VALUE a = argv[0];
  VALUE b = argv[1];
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  
  ScalarOrArray arg_a = value_to_scalar_or_array(a);
  ScalarOrArray arg_b = value_to_scalar_or_array(b);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::zeros({1}, mx::bool_); // Default initialization with bool type
  
  if (arg_a.is_array() && arg_b.is_array()) {
    result = mx::less_equal(arg_a.array(), arg_b.array(), stream);
  } else if (arg_a.is_scalar() && arg_b.is_array()) {
    result = mx::less_equal(mx::array(arg_a.scalar()), arg_b.array(), stream);
  } else if (arg_a.is_array() && arg_b.is_scalar()) {
    result = mx::less_equal(arg_a.array(), mx::array(arg_b.scalar()), stream);
  } else {
    result = mx::array(arg_a.scalar() <= arg_b.scalar());
  }
  
  return wrap_array(result);
}

static VALUE ops_stop_gradient(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..2)", argc);
  }
  
  VALUE array_obj = argv[0];
  VALUE stream_val = (argc > 1) ? argv[1] : Qnil;
  
  if (!rb_obj_is_kind_of(array_obj, rb_path2class("MLX::Core::Array"))) {
    rb_raise(rb_eTypeError, "Argument must be an MLX::Core::Array");
  }
  
  mx::array& arr = get_array(array_obj);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  mx::array result = mx::stop_gradient(arr, stream);
  
  return wrap_array(result);
}

// Additional operations
static VALUE ops_trace(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..4)", argc);
  }
  
  VALUE arr = argv[0];
  mx::array& a = get_array(arr);
  
  int offset = 0;
  int axis1 = 0;
  int axis2 = 1;
  VALUE stream_val = Qnil;
  
  if (argc > 1) {
    offset = NUM2INT(argv[1]);
  }
  
  if (argc > 2) {
    axis1 = NUM2INT(argv[2]);
  }
  
  if (argc > 3) {
    if (rb_obj_is_kind_of(argv[3], rb_path2class("MLX::Stream")) ||
        rb_obj_is_kind_of(argv[3], rb_path2class("MLX::Device"))) {
      stream_val = argv[3];
      axis2 = axis1 + 1;
    } else {
      axis2 = NUM2INT(argv[3]);
    }
  }
  
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  mx::array result = mx::trace(a, offset, axis1, axis2, stream);
  
  return wrap_array(result);
}

static VALUE ops_atleast_1d(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..2)", argc);
  }
  
  VALUE arr = argv[0];
  VALUE stream_val = (argc > 1) ? argv[1] : Qnil;
  
  ScalarOrArray a = value_to_scalar_or_array(arr);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = a.is_array() ?
    mx::atleast_1d(a.array(), stream) :
    mx::atleast_1d(mx::array(a.scalar()), stream);
  
  return wrap_array(result);
}

static VALUE ops_issubdtype(int argc, VALUE* argv, VALUE self) {
  if (argc != 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2)", argc);
  }
  
  VALUE type1 = argv[0];
  VALUE type2 = argv[1];
  
  mx::Dtype d1 = int_to_dtype(NUM2INT(type1));
  mx::Dtype d2 = int_to_dtype(NUM2INT(type2));
  
  bool result = mx::issubdtype(d1, d2);
  
  return result ? Qtrue : Qfalse;
}

static VALUE ops_bitwise_and(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  VALUE a = argv[0];
  VALUE b = argv[1];
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  
  ScalarOrArray arg_a = value_to_scalar_or_array(a);
  ScalarOrArray arg_b = value_to_scalar_or_array(b);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::zeros({1}, mx::bool_); // Initialize with a placeholder
  
  if (arg_a.is_array() && arg_b.is_array()) {
    result = mx::bitwise_and(arg_a.array(), arg_b.array(), stream);
  } else if (arg_a.is_scalar() && arg_b.is_array()) {
    result = mx::bitwise_and(mx::array(arg_a.scalar()), arg_b.array(), stream);
  } else if (arg_a.is_array() && arg_b.is_scalar()) {
    result = mx::bitwise_and(arg_a.array(), mx::array(arg_b.scalar()), stream);
  } else {
    // Convert to integers for bitwise operations
    int val_a = static_cast<int>(arg_a.scalar());
    int val_b = static_cast<int>(arg_b.scalar());
    result = mx::array(val_a & val_b);
  }
  
  return wrap_array(result);
}

static VALUE ops_bitwise_or(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  VALUE a = argv[0];
  VALUE b = argv[1];
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  
  ScalarOrArray arg_a = value_to_scalar_or_array(a);
  ScalarOrArray arg_b = value_to_scalar_or_array(b);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::zeros({1}, mx::bool_); // Initialize with a placeholder
  
  if (arg_a.is_array() && arg_b.is_array()) {
    result = mx::bitwise_or(arg_a.array(), arg_b.array(), stream);
  } else if (arg_a.is_scalar() && arg_b.is_array()) {
    result = mx::bitwise_or(mx::array(arg_a.scalar()), arg_b.array(), stream);
  } else if (arg_a.is_array() && arg_b.is_scalar()) {
    result = mx::bitwise_or(arg_a.array(), mx::array(arg_b.scalar()), stream);
  } else {
    // Convert to integers for bitwise operations
    int val_a = static_cast<int>(arg_a.scalar());
    int val_b = static_cast<int>(arg_b.scalar());
    result = mx::array(val_a | val_b);
  }
  
  return wrap_array(result);
}

static VALUE ops_bitwise_xor(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  VALUE a = argv[0];
  VALUE b = argv[1];
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  
  ScalarOrArray arg_a = value_to_scalar_or_array(a);
  ScalarOrArray arg_b = value_to_scalar_or_array(b);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::zeros({1}, mx::bool_); // Initialize with a placeholder
  
  if (arg_a.is_array() && arg_b.is_array()) {
    result = mx::bitwise_xor(arg_a.array(), arg_b.array(), stream);
  } else if (arg_a.is_scalar() && arg_b.is_array()) {
    result = mx::bitwise_xor(mx::array(arg_a.scalar()), arg_b.array(), stream);
  } else if (arg_a.is_array() && arg_b.is_scalar()) {
    result = mx::bitwise_xor(arg_a.array(), mx::array(arg_b.scalar()), stream);
  } else {
    // Convert to integers for bitwise operations
    int val_a = static_cast<int>(arg_a.scalar());
    int val_b = static_cast<int>(arg_b.scalar());
    result = mx::array(val_a ^ val_b);
  }
  
  return wrap_array(result);
}

static VALUE ops_left_shift(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  VALUE a = argv[0];
  VALUE b = argv[1];
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  
  ScalarOrArray arg_a = value_to_scalar_or_array(a);
  ScalarOrArray arg_b = value_to_scalar_or_array(b);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::zeros({1}, mx::bool_); // Initialize with a placeholder
  
  if (arg_a.is_array() && arg_b.is_array()) {
    result = mx::left_shift(arg_a.array(), arg_b.array(), stream);
  } else if (arg_a.is_scalar() && arg_b.is_array()) {
    result = mx::left_shift(mx::array(arg_a.scalar()), arg_b.array(), stream);
  } else if (arg_a.is_array() && arg_b.is_scalar()) {
    result = mx::left_shift(arg_a.array(), mx::array(arg_b.scalar()), stream);
  } else {
    // Convert to integers for bitwise operations
    int val_a = static_cast<int>(arg_a.scalar());
    int val_b = static_cast<int>(arg_b.scalar());
    result = mx::array(val_a << val_b);
  }
  
  return wrap_array(result);
}

static VALUE ops_right_shift(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  VALUE a = argv[0];
  VALUE b = argv[1];
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  
  ScalarOrArray arg_a = value_to_scalar_or_array(a);
  ScalarOrArray arg_b = value_to_scalar_or_array(b);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::zeros({1}, mx::bool_); // Initialize with a placeholder
  
  if (arg_a.is_array() && arg_b.is_array()) {
    result = mx::right_shift(arg_a.array(), arg_b.array(), stream);
  } else if (arg_a.is_scalar() && arg_b.is_array()) {
    result = mx::right_shift(mx::array(arg_a.scalar()), arg_b.array(), stream);
  } else if (arg_a.is_array() && arg_b.is_scalar()) {
    result = mx::right_shift(arg_a.array(), mx::array(arg_b.scalar()), stream);
  } else {
    // Convert to integers for bitwise operations
    int val_a = static_cast<int>(arg_a.scalar());
    int val_b = static_cast<int>(arg_b.scalar());
    result = mx::array(val_a >> val_b);
  }
  
  return wrap_array(result);
}

static VALUE ops_bitwise_invert(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..2)", argc);
  }
  
  VALUE arr = argv[0];
  VALUE stream_val = (argc > 1) ? argv[1] : Qnil;
  
  ScalarOrArray a = value_to_scalar_or_array(arr);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::zeros({1}, mx::bool_); // Initialize with a placeholder
  
  if (a.is_array()) {
    result = mx::bitwise_invert(a.array(), stream);
  } else {
    // Convert to integer for bitwise operation
    int val = static_cast<int>(a.scalar());
    result = mx::array(~val);
  }
  
  return wrap_array(result);
}

static VALUE ops_view(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  VALUE arr = argv[0];
  VALUE dtype = argv[1];
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  
  mx::array& a = get_array(arr);
  mx::Dtype dt = int_to_dtype(NUM2INT(dtype));
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::view(a, dt, stream);
  return wrap_array(result);
}

static VALUE ops_hadamard_transform(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..3)", argc);
  }
  
  VALUE arr = argv[0];
  mx::array& a = get_array(arr);
  
  int axis = 0;
  if (argc > 1 && !NIL_P(argv[1])) {
    axis = NUM2INT(argv[1]);
  }
  
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::hadamard_transform(a, axis, stream);
  return wrap_array(result);
}

static VALUE ops_einsum_path(int argc, VALUE* argv, VALUE self) {
  rb_raise(rb_eNotImpError, "einsum_path not implemented in this example");
  return Qnil;
}

static VALUE ops_einsum(int argc, VALUE* argv, VALUE self) {
  if (argc < 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected at least 2)", argc);
  }
  
  VALUE equation = argv[0];
  VALUE stream_val = Qnil;
  
  // Extract arrays and find stream if present
  std::vector<mx::array> arrays;
  int arr_end = argc;
  
  // Check if last argument is stream
  if (rb_obj_is_kind_of(argv[argc-1], rb_path2class("MLX::Stream")) ||
      rb_obj_is_kind_of(argv[argc-1], rb_path2class("MLX::Device"))) {
    stream_val = argv[argc-1];
    arr_end = argc - 1;
  }
  
  // Convert arrays
  for (int i = 1; i < arr_end; i++) {
    mx::array& arr = get_array(argv[i]);
    arrays.push_back(arr);
  }
  
  std::string eq_str = RSTRING_PTR(equation);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::einsum(eq_str, arrays, stream);
  return wrap_array(result);
}

static VALUE ops_roll(int argc, VALUE* argv, VALUE self) {
  if (argc < 3 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 3..4)", argc);
  }
  
  VALUE arr = argv[0];
  VALUE shift = argv[1];
  VALUE axes = argv[2];
  VALUE stream_val = (argc > 3) ? argv[3] : Qnil;
  
  mx::array& a = get_array(arr);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result;
  
  if (RB_TYPE_P(shift, T_FIXNUM) && RB_TYPE_P(axes, T_FIXNUM)) {
    // Single shift, single axis
    int shift_val = NUM2INT(shift);
    int axis_val = NUM2INT(axes);
    result = mx::roll(a, shift_val, axis_val, stream);
  } else if (RB_TYPE_P(shift, T_ARRAY) && RB_TYPE_P(axes, T_ARRAY)) {
    // Multiple shifts, multiple axes
    std::vector<int> shifts = ruby_array_to_shape(shift);
    std::vector<int> axis_vals = ruby_array_to_shape(axes);
    result = mx::roll(a, shifts, axis_vals, stream);
  } else {
    rb_raise(rb_eTypeError, "shift and axes must be both integers or both arrays");
  }
  
  return wrap_array(result);
}

static VALUE ops_real(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..2)", argc);
  }
  
  VALUE arr = argv[0];
  VALUE stream_val = (argc > 1) ? argv[1] : Qnil;
  
  mx::array& a = get_array(arr);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::real(a, stream);
  return wrap_array(result);
}

static VALUE ops_imag(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..2)", argc);
  }
  
  VALUE arr = argv[0];
  VALUE stream_val = (argc > 1) ? argv[1] : Qnil;
  
  mx::array& a = get_array(arr);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::imag(a, stream);
  return wrap_array(result);
}

static VALUE ops_slice(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  VALUE arr = argv[0];
  VALUE slices = argv[1];
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  
  mx::array& a = get_array(arr);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  Check_Type(slices, T_ARRAY);
  
  // Instead of using the complex version with arrays of arrays, 
  // we'll just use the Shape version for simplicity
  if (RARRAY_LEN(slices) >= 3) {
    std::vector<int> start, stop, strides;
    
    VALUE start_val = rb_ary_entry(slices, 0);
    VALUE stop_val = rb_ary_entry(slices, 1);
    VALUE stride_val = rb_ary_entry(slices, 2);
    
    if (RB_TYPE_P(start_val, T_ARRAY)) start = ruby_array_to_shape(start_val);
    else start.push_back(NUM2INT(start_val));
    
    if (RB_TYPE_P(stop_val, T_ARRAY)) stop = ruby_array_to_shape(stop_val);
    else stop.push_back(NUM2INT(stop_val));
    
    if (RB_TYPE_P(stride_val, T_ARRAY)) strides = ruby_array_to_shape(stride_val);
    else strides.push_back(NUM2INT(stride_val));
    
    mx::array result = mx::slice(a, start, stop, strides, stream);
    return wrap_array(result);
  }
  
  // Fallback
  rb_raise(rb_eNotImpError, "slice with complex parameters not fully implemented in this example");
  return Qnil;
}

static VALUE ops_slice_update(int argc, VALUE* argv, VALUE self) {
  if (argc < 3 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 3..4)", argc);
  }
  
  VALUE arr = argv[0];
  VALUE update_val = argv[1];
  VALUE slices = argv[2];
  VALUE stream_val = (argc > 3) ? argv[3] : Qnil;
  
  mx::array& a = get_array(arr);
  mx::array& update = get_array(update_val);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  Check_Type(slices, T_ARRAY);
  
  // Use the simpler version with Shapes
  if (RARRAY_LEN(slices) >= 3) {
    std::vector<int> start, stop, strides;
    
    VALUE start_val = rb_ary_entry(slices, 0);
    VALUE stop_val = rb_ary_entry(slices, 1);
    VALUE stride_val = rb_ary_entry(slices, 2);
    
    if (RB_TYPE_P(start_val, T_ARRAY)) start = ruby_array_to_shape(start_val);
    else start.push_back(NUM2INT(start_val));
    
    if (RB_TYPE_P(stop_val, T_ARRAY)) stop = ruby_array_to_shape(stop_val);
    else stop.push_back(NUM2INT(stop_val));
    
    if (RB_TYPE_P(stride_val, T_ARRAY)) strides = ruby_array_to_shape(stride_val);
    else strides.push_back(NUM2INT(stride_val));
    
    mx::array result = mx::slice_update(a, update, start, stop, strides, stream);
    return wrap_array(result);
  }
  
  // Fallback
  rb_raise(rb_eNotImpError, "slice_update with complex parameters not fully implemented in this example");
  return Qnil;
}

static VALUE ops_contiguous(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..2)", argc);
  }
  
  VALUE arr = argv[0];
  VALUE stream_val = (argc > 1) ? argv[1] : Qnil;
  
  mx::array& a = get_array(arr);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::contiguous(a, stream);
  return wrap_array(result);
}
// unflatten(a, axis, shape, stream=None)
static VALUE ops_unflatten(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("unflatten", 3, 4);
  mx::array& a = get_array(argv[0]);
  int axis = NUM2INT(argv[1]);
  std::vector<int> new_shape = ruby_array_to_shape(argv[2]);
  GET_STREAM(stream, (argc == 4) ? argv[3] : Qnil);
  RETURN_ARRAY(mx::unflatten(a, axis, new_shape, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// divmod(a, b, stream=None)
static VALUE ops_divmod(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "divmod: wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  SCALAR_OR_ARRAY(a, argv[0]);
  SCALAR_OR_ARRAY(b, argv[1]);
  GET_STREAM(stream, (argc == 3) ? argv[2] : Qnil);
  
  // divmod returns a tuple [quotient, remainder]
  // Since divmod returns a std::tuple<array,array>, we need to return an array of arrays
  auto result = mx::divmod(a, b, stream);
  
  // Create Ruby array with two elements: quotient and remainder
  VALUE rb_result = rb_ary_new2(2);
  rb_ary_store(rb_result, 0, wrap_array(std::get<0>(result)));
  rb_ary_store(rb_result, 1, wrap_array(std::get<1>(result)));
  
  return rb_result;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// floor_divide(a, b, stream=None)
static VALUE ops_floor_divide(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("floor_divide", 2, 3);
  SCALAR_OR_ARRAY(a, argv[0]);
  SCALAR_OR_ARRAY(b, argv[1]);
  GET_STREAM(stream, (argc == 3) ? argv[2] : Qnil);
  RETURN_ARRAY(mx::floor_divide(a, b, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// remainder(a, b, stream=None)
static VALUE ops_remainder(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("remainder", 2, 3);
  SCALAR_OR_ARRAY(a, argv[0]);
  SCALAR_OR_ARRAY(b, argv[1]);
  GET_STREAM(stream, (argc == 3) ? argv[2] : Qnil);
  RETURN_ARRAY(mx::remainder(a, b, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// array_equal(a, b, *, equal_nan=False, stream=None)
static VALUE ops_array_equal(int argc, VALUE* argv, VALUE self) {
  // The Python signature is array_equal(a,b,equal_nan=False,stream=None).
  // We'll parse up to 4 args: a,b,[equal_nan],[stream].
  if (argc < 2 || argc > 4) {
    rb_raise(rb_eArgError, "array_equal: wrong number of args (given %d, expected 2..4)", argc);
  }
  SCALAR_OR_ARRAY(a, argv[0]);
  SCALAR_OR_ARRAY(b, argv[1]);

  bool equal_nan = false;
  VALUE stream_val = Qnil;
  if (argc >= 3) {
    if (RB_TYPE_P(argv[2], T_TRUE) || RB_TYPE_P(argv[2], T_FALSE)) {
      equal_nan = (argv[2] == Qtrue);
      if (argc == 4) stream_val = argv[3];
    } else {
      // third arg is actually the stream
      stream_val = argv[2];
      if (argc == 4) {
        // the 4th doesn't exist, so error
        rb_raise(rb_eArgError, "array_equal: too many arguments (did you mean to pass equal_nan first?)");
      }
    }
  }
  GET_STREAM(stream, stream_val);
  RETURN_ARRAY(mx::array_equal(a, b, equal_nan, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// matmul(a, b, stream=None)
static VALUE ops_matmul(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("matmul", 2, 3);
  mx::array& lhs = get_array(argv[0]);
  mx::array& rhs = get_array(argv[1]);
  GET_STREAM(stream, (argc == 3) ? argv[2] : Qnil);
  RETURN_ARRAY(mx::matmul(lhs, rhs, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// square(a, stream=None)
static VALUE ops_square(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("square", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::square(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sqrt(a, stream=None)
static VALUE ops_sqrt(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("sqrt", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::sqrt(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// rsqrt(a, stream=None)
static VALUE ops_rsqrt(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("rsqrt", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::rsqrt(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// reciprocal(a, stream=None)
static VALUE ops_reciprocal(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("reciprocal", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::reciprocal(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// logical_not(a, stream=None)
static VALUE ops_logical_not(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("logical_not", 1, 2);
SCALAR_OR_ARRAY(a, argv[0]);
GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::logical_not(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// logical_and(a, b, stream=None)
static VALUE ops_logical_and(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("logical_and", 2, 3);
  SCALAR_OR_ARRAY(a, argv[0]);
  SCALAR_OR_ARRAY(b, argv[1]);
  GET_STREAM(stream, (argc == 3) ? argv[2] : Qnil);
  RETURN_ARRAY(mx::logical_and(a, b, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// logical_or(a, b, stream=None)
static VALUE ops_logical_or(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("logical_or", 2, 3);
  SCALAR_OR_ARRAY(a, argv[0]);
  SCALAR_OR_ARRAY(b, argv[1]);
  GET_STREAM(stream, (argc == 3) ? argv[2] : Qnil);
  RETURN_ARRAY(mx::logical_or(a, b, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// logaddexp(a, b, stream=None)
static VALUE ops_logaddexp(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("logaddexp", 2, 3);
  SCALAR_OR_ARRAY(a, argv[0]);
  SCALAR_OR_ARRAY(b, argv[1]);
  GET_STREAM(stream, (argc == 3) ? argv[2] : Qnil);
  RETURN_ARRAY(mx::logaddexp(a, b, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// exp(a, stream=None)
static VALUE ops_exp(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("exp", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::exp(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// expm1(a, stream=None)
static VALUE ops_expm1(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("expm1", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::expm1(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// erf(a, stream=None)
static VALUE ops_erf(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("erf", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::erf(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// erfinv(a, stream=None)
static VALUE ops_erfinv(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("erfinv", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::erfinv(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sin(a, stream=None)
static VALUE ops_sin(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("sin", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::sin(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// cos(a, stream=None)
static VALUE ops_cos(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("cos", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::cos(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// tan(a, stream=None)
static VALUE ops_tan(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("tan", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::tan(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// arcsin(a, stream=None)
static VALUE ops_arcsin(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("arcsin", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::arcsin(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// arccos(a, stream=None)
static VALUE ops_arccos(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("arccos", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::arccos(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// arctan(a, stream=None)
static VALUE ops_arctan(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("arctan", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::arctan(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// arctan2(a, b, stream=None)
static VALUE ops_arctan2(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("arctan2", 2, 3);
  mx::array& a = get_array(argv[0]);
  mx::array& b = get_array(argv[1]);
  GET_STREAM(stream, (argc == 3) ? argv[2] : Qnil);
  RETURN_ARRAY(mx::arctan2(a, b, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sinh(a, stream=None)
static VALUE ops_sinh(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("sinh", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::sinh(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// cosh(a, stream=None)
static VALUE ops_cosh(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("cosh", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::cosh(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// tanh(a, stream=None)
static VALUE ops_tanh(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("tanh", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::tanh(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// arcsinh(a, stream=None)
static VALUE ops_arcsinh(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("arcsinh", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::arcsinh(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// arccosh(a, stream=None)
static VALUE ops_arccosh(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("arccosh", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::arccosh(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// arctanh(a, stream=None)
static VALUE ops_arctanh(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("arctanh", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::arctanh(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// degrees(a, stream=None)
static VALUE ops_degrees(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("degrees", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::degrees(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// radians(a, stream=None)
static VALUE ops_radians(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("radians", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::radians(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// log(a, stream=None)
static VALUE ops_log(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("log", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::log(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// log2(a, stream=None)
static VALUE ops_log2(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("log2", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::log2(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// log10(a, stream=None)
static VALUE ops_log10(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("log10", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::log10(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// log1p(a, stream=None)
static VALUE ops_log1p(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("log1p", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::log1p(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sigmoid(a, stream=None)
static VALUE ops_sigmoid(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("sigmoid", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::sigmoid(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// power(a, b, stream=None)
static VALUE ops_power(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("power", 2, 3);
  SCALAR_OR_ARRAY(a, argv[0]);
  SCALAR_OR_ARRAY(b, argv[1]);
  GET_STREAM(stream, (argc == 3) ? argv[2] : Qnil);
  RETURN_ARRAY(mx::power(a, b, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// linspace(start, stop, num=50, dtype=float32, stream=None)
static VALUE ops_linspace(int argc, VALUE* argv, VALUE self) {
  // We'll parse (start, stop, num=50, dtype=? int, stream?)
  if (argc < 2 || argc > 5) {
    rb_raise(rb_eArgError, "linspace: wrong number of args (given %d, expected 2..5)", argc);
  }
  double start = NUM2DBL(argv[0]);
  double stop  = NUM2DBL(argv[1]);

  int num = 50;
  mx::Dtype dt = mx::float32;
  VALUE stream_val = Qnil;

  if (argc >= 3) num = NUM2INT(argv[2]);
  if (argc >= 4 && !NIL_P(argv[3])) dt = int_to_dtype(NUM2INT(argv[3]));
  if (argc == 5) stream_val = argv[4];

  GET_STREAM(stream, stream_val);
  RETURN_ARRAY(mx::linspace(start, stop, num, dt, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// kron(a, b, stream=None)
static VALUE ops_kron(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("kron", 2, 3);
  mx::array& a = get_array(argv[0]);
  mx::array& b = get_array(argv[1]);
  GET_STREAM(stream, (argc == 3) ? argv[2] : Qnil);
  RETURN_ARRAY(mx::kron(a, b, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// take(a, indices, axis=None, stream=None)
static VALUE ops_take(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 4) {
    rb_raise(rb_eArgError, "take: wrong number of args (given %d, expected 2..4)", argc);
  }
  mx::array& arr = get_array(argv[0]);
  VALUE indices_val = argv[1];
  int axis = -999999; // sentinel
  VALUE stream_val = Qnil;

  if (argc >= 3 && !NIL_P(argv[2])) {
    // If argv[2] is an integer, that's axis
    if (RB_TYPE_P(argv[2], T_FIXNUM)) axis = NUM2INT(argv[2]);
    else stream_val = argv[2];
  }
  if (argc == 4) stream_val = argv[3];

  mx::array indices = mx::zeros({1}, mx::int32); // Default initialization
  if (rb_obj_is_kind_of(indices_val, rb_path2class("MLX::Core::Array"))) {
    indices = get_array(indices_val);
  } else if (FIXNUM_P(indices_val)) {
    // Single int -> we'll handle it with mx::take(a, int, axis, s)
  } else {
    rb_raise(rb_eTypeError, "take: indices must be an int or MLX::Core::Array");
  }

  GET_STREAM(stream, stream_val);
  if (FIXNUM_P(indices_val)) {
    int idx = NUM2INT(indices_val);
    if (axis == -999999) {
      RETURN_ARRAY(mx::take(arr, idx, stream));
    } else {
      RETURN_ARRAY(mx::take(arr, idx, axis, stream));
    }
  } else {
    if (axis == -999999) {
      RETURN_ARRAY(mx::take(arr, indices, stream));
    } else {
      RETURN_ARRAY(mx::take(arr, indices, axis, stream));
    }
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// take_along_axis(a, indices, axis=None, stream=None)
static VALUE ops_take_along_axis(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("take_along_axis", 2, 4);
  mx::array& a = get_array(argv[0]);
  mx::array& idx = get_array(argv[1]);
  int axis = -999999; // means None
  VALUE stream_val = Qnil;

  if (argc >= 3 && !NIL_P(argv[2])) axis = NUM2INT(argv[2]);
  if (argc == 4) stream_val = argv[3];

  GET_STREAM(stream, stream_val);
  if (axis == -999999) {
    // axis=None -> flatten a
    RETURN_ARRAY(mx::take_along_axis(mx::reshape(a, { -1 }, stream), idx, 0, stream));
  } else {
    RETURN_ARRAY(mx::take_along_axis(a, idx, axis, stream));
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// put_along_axis(a, indices, values, axis=None, stream=None)
static VALUE ops_put_along_axis(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("put_along_axis", 3, 5);
  mx::array& a = get_array(argv[0]);
  mx::array& idx = get_array(argv[1]);
  mx::array& vals = get_array(argv[2]);

  int axis = -999999;
  VALUE stream_val = Qnil;
  if (argc >= 4 && !NIL_P(argv[3])) axis = NUM2INT(argv[3]);
  if (argc == 5) stream_val = argv[4];

  GET_STREAM(stream, stream_val);
  if (axis == -999999) {
    // axis=None
    mx::array resh = mx::reshape(a, { -1 }, stream);
    mx::array out = mx::put_along_axis(resh, idx, vals, 0, stream);
    RETURN_ARRAY(mx::reshape(out, a.shape(), stream));
  } else {
    RETURN_ARRAY(mx::put_along_axis(a, idx, vals, axis, stream));
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// zeros_like(a, stream=None)
static VALUE ops_zeros_like(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("zeros_like", 1, 2);
  mx::array& src = get_array(argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::zeros_like(src, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ones_like(a, stream=None)
static VALUE ops_ones_like(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("ones_like", 1, 2);
  mx::array& src = get_array(argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::ones_like(src, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// tri(n, m=None, k=0, dtype=float32, stream=None)
static VALUE ops_tri(int argc, VALUE* argv, VALUE self) {
  // parse n, [m], [k], [dtype], [stream]
  if (argc < 1 || argc > 5) {
    rb_raise(rb_eArgError, "tri: wrong number of args (given %d, expected 1..5)", argc);
  }
  int n = NUM2INT(argv[0]);
  int m = n;
  int k = 0;
  mx::Dtype dt = mx::float32;
  VALUE stream_val = Qnil;

  if (argc >= 2 && !NIL_P(argv[1])) m = NUM2INT(argv[1]);
  if (argc >= 3 && !NIL_P(argv[2])) k = NUM2INT(argv[2]);
  if (argc >= 4 && !NIL_P(argv[3])) dt = int_to_dtype(NUM2INT(argv[3]));
  if (argc == 5) stream_val = argv[4];

  GET_STREAM(stream, stream_val);
  RETURN_ARRAY(mx::tri(n, m, k, dt, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// tril(x, k=0, stream=None)
static VALUE ops_tril(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("tril", 1, 3);
  mx::array& x = get_array(argv[0]);
  int k = 0;
  VALUE stream_val = Qnil;
  if (argc >= 2) k = NUM2INT(argv[1]);
  if (argc == 3) stream_val = argv[2];
  GET_STREAM(stream, stream_val);
  RETURN_ARRAY(mx::tril(x, k, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// triu(x, k=0, stream=None)
static VALUE ops_triu(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("triu", 1, 3);
  mx::array& x = get_array(argv[0]);
  int k = 0;
  VALUE stream_val = Qnil;
  if (argc >= 2) k = NUM2INT(argv[1]);
  if (argc == 3) stream_val = argv[2];
  GET_STREAM(stream, stream_val);
  RETURN_ARRAY(mx::triu(x, k, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=false, stream=None)
static VALUE ops_allclose(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 6) {
    rb_raise(rb_eArgError, "allclose: wrong number of args (given %d, expected 2..6)", argc);
  }
  mx::array& a = get_array(argv[0]);
  mx::array& b = get_array(argv[1]);
  double rtol = 1e-5, atol = 1e-8;
  bool equal_nan = false;
  VALUE stream_val = Qnil;

  if (argc >= 3) rtol = NUM2DBL(argv[2]);
  if (argc >= 4) atol = NUM2DBL(argv[3]);
  if (argc >= 5) equal_nan = (argv[4] == Qtrue);
  if (argc == 6) stream_val = argv[5];

  GET_STREAM(stream, stream_val);
  RETURN_ARRAY(mx::allclose(a, b, rtol, atol, equal_nan, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// isclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=false, stream=None)
static VALUE ops_isclose(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 6) {
    rb_raise(rb_eArgError, "isclose: wrong number of args (given %d, expected 2..6)", argc);
  }
  mx::array& a = get_array(argv[0]);
  mx::array& b = get_array(argv[1]);
  double rtol = 1e-5, atol = 1e-8;
  bool equal_nan = false;
  VALUE stream_val = Qnil;

  if (argc >= 3) rtol = NUM2DBL(argv[2]);
  if (argc >= 4) atol = NUM2DBL(argv[3]);
  if (argc >= 5) equal_nan = (argv[4] == Qtrue);
  if (argc == 6) stream_val = argv[5];

  GET_STREAM(stream, stream_val);
  RETURN_ARRAY(mx::isclose(a, b, rtol, atol, equal_nan, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// all(a, axis=None, keepdims=false, stream=None)
static VALUE ops_all(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 4) {
    rb_raise(rb_eArgError, "all: wrong number of args (given %d, expected 1..4)", argc);
  }
  mx::array& arr = get_array(argv[0]);
  int axis = -999999; // means None
  bool keepdims = false;
  VALUE stream_val = Qnil;

  if (argc >= 2 && !NIL_P(argv[1])) axis = NUM2INT(argv[1]);
  if (argc >= 3) keepdims = (argv[2] == Qtrue);
  if (argc == 4) stream_val = argv[3];

  GET_STREAM(stream, stream_val);
  // If axis == -999999 => None => reduce all dims
  if (axis == -999999) {
    RETURN_ARRAY(mx::all(arr, {}, keepdims, stream)); 
  } else {
    RETURN_ARRAY(mx::all(arr, {axis}, keepdims, stream));
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// any(a, axis=None, keepdims=false, stream=None)
static VALUE ops_any(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 4) {
    rb_raise(rb_eArgError, "any: wrong number of args (given %d, expected 1..4)", argc);
  }
  mx::array& arr = get_array(argv[0]);
  int axis = -999999;
  bool keepdims = false;
  VALUE stream_val = Qnil;

  if (argc >= 2 && !NIL_P(argv[1])) axis = NUM2INT(argv[1]);
  if (argc >= 3) keepdims = (argv[2] == Qtrue);
  if (argc == 4) stream_val = argv[3];

  GET_STREAM(stream, stream_val);
  if (axis == -999999) {
    RETURN_ARRAY(mx::any(arr, {}, keepdims, stream));
  } else {
    RETURN_ARRAY(mx::any(arr, {axis}, keepdims, stream));
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// minimum(a, b, stream=None)
static VALUE ops_minimum(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("minimum", 2, 3);
  SCALAR_OR_ARRAY(a, argv[0]);
  SCALAR_OR_ARRAY(b, argv[1]);
  GET_STREAM(stream, (argc == 3) ? argv[2] : Qnil);
  RETURN_ARRAY(mx::minimum(a, b, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// maximum(a, b, stream=None)
static VALUE ops_maximum(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("maximum", 2, 3);
  SCALAR_OR_ARRAY(a, argv[0]);
  SCALAR_OR_ARRAY(b, argv[1]);
  GET_STREAM(stream, (argc == 3) ? argv[2] : Qnil);
  RETURN_ARRAY(mx::maximum(a, b, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// floor(a, stream=None)
static VALUE ops_floor(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("floor", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::floor(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ceil(a, stream=None)
static VALUE ops_ceil(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("ceil", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::ceil(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// isnan(a, stream=None)
static VALUE ops_isnan(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("isnan", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::isnan(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// isinf(a, stream=None)
static VALUE ops_isinf(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("isinf", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::isinf(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// isfinite(a, stream=None)
static VALUE ops_isfinite(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("isfinite", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::isfinite(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// isposinf(a, stream=None)
static VALUE ops_isposinf(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("isposinf", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::isposinf(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// isneginf(a, stream=None)
static VALUE ops_isneginf(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("isneginf", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2) ? argv[1] : Qnil);
  RETURN_ARRAY(mx::isneginf(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// moveaxis(a, source, destination, stream=None)
static VALUE ops_moveaxis(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("moveaxis", 3, 4);
  mx::array& arr = get_array(argv[0]);
  int source = NUM2INT(argv[1]);
  int destination = NUM2INT(argv[2]);
  GET_STREAM(stream, (argc == 4) ? argv[3] : Qnil);
  RETURN_ARRAY(mx::moveaxis(arr, source, destination, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// swapaxes(a, axis1, axis2, stream=None)
static VALUE ops_swapaxes(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("swapaxes", 3, 4);
  mx::array& arr = get_array(argv[0]);
  int axis1 = NUM2INT(argv[1]);
  int axis2 = NUM2INT(argv[2]);
  GET_STREAM(stream, (argc == 4) ? argv[3] : Qnil);
  RETURN_ARRAY(mx::swapaxes(arr, axis1, axis2, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// transpose(a, axes=None, stream=None)
static VALUE ops_transpose(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 3) {
    rb_raise(rb_eArgError, "transpose: wrong number of args (given %d, expected 1..3)", argc);
  }
  mx::array& arr = get_array(argv[0]);
  VALUE axes_val = Qnil;
  VALUE stream_val = Qnil;

  if (argc >= 2) axes_val = argv[1];
  if (argc == 3) stream_val = argv[2];

  GET_STREAM(stream, stream_val);
  if (NIL_P(axes_val)) {
    RETURN_ARRAY(mx::transpose(arr, stream));
  } else {
    // parse as array of int
    std::vector<int> axes = ruby_array_to_shape(axes_val);
    RETURN_ARRAY(mx::transpose(arr, axes, stream));
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// permute_dims(a, axes=None, stream=None) (same as transpose)
static VALUE ops_permute_dims(int argc, VALUE* argv, VALUE self) {
  return ops_transpose(argc, argv, self); // identical
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sum(a, axis=None, keepdims=false, stream=None)
static VALUE ops_sum(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 4) {
    rb_raise(rb_eArgError, "sum: wrong number of args (given %d, expected 1..4)", argc);
  }
  mx::array& arr = get_array(argv[0]);
  int axis = -999999;
  bool keepdims = false;
  VALUE stream_val = Qnil;
  if (argc >= 2 && !NIL_P(argv[1])) axis = NUM2INT(argv[1]);
  if (argc >= 3) keepdims = (argv[2] == Qtrue);
  if (argc == 4) stream_val = argv[3];
  GET_STREAM(stream, stream_val);

  if (axis == -999999) {
    RETURN_ARRAY(mx::sum(arr, {}, keepdims, stream));
  } else {
    RETURN_ARRAY(mx::sum(arr, {axis}, keepdims, stream));
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// prod(a, axis=None, keepdims=false, stream=None)
static VALUE ops_prod(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 4) {
    rb_raise(rb_eArgError, "prod: wrong number of args (given %d, expected 1..4)", argc);
  }
  mx::array& arr = get_array(argv[0]);
  int axis = -999999;
  bool keepdims = false;
  VALUE stream_val = Qnil;
  if (argc >= 2 && !NIL_P(argv[1])) axis = NUM2INT(argv[1]);
  if (argc >= 3) keepdims = (argv[2] == Qtrue);
  if (argc == 4) stream_val = argv[3];

  GET_STREAM(stream, stream_val);
  if (axis == -999999) {
    RETURN_ARRAY(mx::prod(arr, {}, keepdims, stream));
  } else {
    RETURN_ARRAY(mx::prod(arr, {axis}, keepdims, stream));
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// min(a, axis=None, keepdims=false, stream=None)
static VALUE ops_min(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 4) {
    rb_raise(rb_eArgError, "min: wrong number of args (given %d, expected 1..4)", argc);
  }
  mx::array& arr = get_array(argv[0]);
  int axis = -999999;
  bool keepdims = false;
  VALUE stream_val = Qnil;
  if (argc >= 2 && !NIL_P(argv[1])) axis = NUM2INT(argv[1]);
  if (argc >= 3) keepdims = (argv[2] == Qtrue);
  if (argc == 4) stream_val = argv[3];

  GET_STREAM(stream, stream_val);
  if (axis == -999999) {
    RETURN_ARRAY(mx::min(arr, {}, keepdims, stream));
  } else {
    RETURN_ARRAY(mx::min(arr, {axis}, keepdims, stream));
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// max(a, axis=None, keepdims=false, stream=None)
static VALUE ops_max(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 4) {
    rb_raise(rb_eArgError, "max: wrong number of args (given %d, expected 1..4)", argc);
  }
  mx::array& arr = get_array(argv[0]);
  int axis = -999999;
  bool keepdims = false;
  VALUE stream_val = Qnil;
  if (argc >= 2 && !NIL_P(argv[1])) axis = NUM2INT(argv[1]);
  if (argc >= 3) keepdims = (argv[2] == Qtrue);
  if (argc == 4) stream_val = argv[3];

  GET_STREAM(stream, stream_val);
  if (axis == -999999) {
    RETURN_ARRAY(mx::max(arr, {}, keepdims, stream));
  } else {
    RETURN_ARRAY(mx::max(arr, {axis}, keepdims, stream));
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// logsumexp(a, axis=None, keepdims=false, stream=None)
static VALUE ops_logsumexp(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 4) {
    rb_raise(rb_eArgError, "logsumexp: wrong number of args (given %d, expected 1..4)", argc);
  }
  mx::array& arr = get_array(argv[0]);
  int axis = -999999;
  bool keepdims = false;
  VALUE stream_val = Qnil;
  if (argc >= 2 && !NIL_P(argv[1])) axis = NUM2INT(argv[1]);
  if (argc >= 3) keepdims = (argv[2] == Qtrue);
  if (argc == 4) stream_val = argv[3];

  GET_STREAM(stream, stream_val);
  if (axis == -999999) {
    RETURN_ARRAY(mx::logsumexp(arr, {}, keepdims, stream));
  } else {
    RETURN_ARRAY(mx::logsumexp(arr, {axis}, keepdims, stream));
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// mean(a, axis=None, keepdims=false, stream=None)
static VALUE ops_mean(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("mean", 1, 4);
  mx::array& arr = get_array(argv[0]);
  int axis = -999999;
  bool keepdims = false;
  VALUE stream_val = Qnil;
  if (argc >= 2 && !NIL_P(argv[1])) axis = NUM2INT(argv[1]);
  if (argc >= 3) keepdims = (argv[2] == Qtrue);
  if (argc == 4) stream_val = argv[3];
  GET_STREAM(stream, stream_val);
  if (axis == -999999) {
    RETURN_ARRAY(mx::mean(arr, {}, keepdims, stream));
  } else {
    RETURN_ARRAY(mx::mean(arr, {axis}, keepdims, stream));
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// var(a, axis=None, keepdims=false, ddof=0, stream=None)
static VALUE ops_var(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 5) {
    rb_raise(rb_eArgError, "var: wrong number of args (given %d, expected 1..5)", argc);
  }
  mx::array& arr = get_array(argv[0]);
  int axis = -999999;
  bool keepdims = false;
  int ddof = 0;
  VALUE stream_val = Qnil;
  if (argc >= 2 && !NIL_P(argv[1])) axis = NUM2INT(argv[1]);
  if (argc >= 3) keepdims = (argv[2] == Qtrue);
  if (argc >= 4) ddof = NUM2INT(argv[3]);
  if (argc == 5) stream_val = argv[4];
  GET_STREAM(stream, stream_val);

  if (axis == -999999) {
    RETURN_ARRAY(mx::var(arr, {}, keepdims, ddof, stream));
  } else {
    RETURN_ARRAY(mx::var(arr, {axis}, keepdims, ddof, stream));
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// std(a, axis=None, keepdims=false, ddof=0, stream=None)
static VALUE ops_std(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 5) {
    rb_raise(rb_eArgError, "std: wrong number of args (given %d, expected 1..5)", argc);
  }
  mx::array& arr = get_array(argv[0]);
  int axis = -999999;
  bool keepdims = false;
  int ddof = 0;
  VALUE stream_val = Qnil;
  if (argc >= 2 && !NIL_P(argv[1])) axis = NUM2INT(argv[1]);
  if (argc >= 3) keepdims = (argv[2] == Qtrue);
  if (argc >= 4) ddof = NUM2INT(argv[3]);
  if (argc == 5) stream_val = argv[4];
  GET_STREAM(stream, stream_val);

  if (axis == -999999) {
    RETURN_ARRAY(mx::std(arr, {}, keepdims, ddof, stream));
  } else {
    RETURN_ARRAY(mx::std(arr, {axis}, keepdims, ddof, stream));
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// split(a, indices_or_sections, axis=0, stream=None)
static VALUE ops_split(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("split", 2, 4);
  mx::array& arr = get_array(argv[0]);
  VALUE ios = argv[1]; // indices or sections
  int axis = 0;
  VALUE stream_val = Qnil;
  if (argc >= 3) axis = NUM2INT(argv[2]);
  if (argc == 4) stream_val = argv[3];
  GET_STREAM(stream, stream_val);

  // We'll return an array of arrays? Possibly we need a Ruby array of MLX::Arrays.
  // The underlying C++ function returns std::vector<mx::array>.
  // We'll just do "mx::split(...)" and then wrap each result.

  std::vector<mx::array> res;
  if (FIXNUM_P(ios)) {
    int sec = NUM2INT(ios);
    res = mx::split(arr, sec, axis, stream);
  } else if (RB_TYPE_P(ios, T_ARRAY)) {
    std::vector<int> idx = ruby_array_to_shape(ios);
    res = mx::split(arr, idx, axis, stream);
  } else {
    rb_raise(rb_eTypeError, "split: indices_or_sections must be int or int[]");
  }

  // wrap them in a Ruby array
  VALUE out = rb_ary_new2((long)res.size());
  for (size_t i=0; i<res.size(); i++) {
    rb_ary_store(out, i, wrap_array(res[i]));
  }
  return out;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// argmin(a, axis=None, keepdims=false, stream=None)
static VALUE ops_argmin(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 4) {
    rb_raise(rb_eArgError, "argmin: wrong number of args (given %d, expected 1..4)", argc);
  }
  mx::array& a = get_array(argv[0]);
  bool has_axis = false;
  int axis = 0;
  bool keepdims = false;
  VALUE stream_val = Qnil;

  if (argc >= 2 && !NIL_P(argv[1])) {
    has_axis = true; axis = NUM2INT(argv[1]);
  }
  if (argc >= 3) keepdims = (argv[2] == Qtrue);
  if (argc == 4) stream_val = argv[3];

  GET_STREAM(stream, stream_val);
  if (has_axis) {
    RETURN_ARRAY(mx::argmin(a, axis, keepdims, stream));
  } else {
    RETURN_ARRAY(mx::argmin(a, keepdims, stream));
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// argmax(a, axis=None, keepdims=false, stream=None)
static VALUE ops_argmax(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 4) {
    rb_raise(rb_eArgError, "argmax: wrong number of args (given %d, expected 1..4)", argc);
  }
  mx::array& a = get_array(argv[0]);
  bool has_axis = false;
  int axis = 0;
  bool keepdims = false;
  VALUE stream_val = Qnil;

  if (argc >= 2 && !NIL_P(argv[1])) {
    has_axis = true; axis = NUM2INT(argv[1]);
  }
  if (argc >= 3) keepdims = (argv[2] == Qtrue);
  if (argc == 4) stream_val = argv[3];

  GET_STREAM(stream, stream_val);
  if (has_axis) {
    RETURN_ARRAY(mx::argmax(a, axis, keepdims, stream));
  } else {
    RETURN_ARRAY(mx::argmax(a, keepdims, stream));
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sort(a, axis=-1, stream=None)
static VALUE ops_sort(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 3) {
    rb_raise(rb_eArgError, "sort: wrong number of args (given %d, expected 1..3)", argc);
  }
  mx::array& a = get_array(argv[0]);
  int axis = -1;
  VALUE stream_val = Qnil;
  if (argc >= 2 && !NIL_P(argv[1])) axis = NUM2INT(argv[1]);
  if (argc == 3) stream_val = argv[2];
  GET_STREAM(stream, stream_val);
  if (axis == -999999) axis = -1;
  RETURN_ARRAY(mx::sort(a, axis, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// argsort(a, axis=-1, stream=None)
static VALUE ops_argsort(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 3) {
    rb_raise(rb_eArgError, "argsort: wrong number of args (given %d, expected 1..3)", argc);
  }
  mx::array& a = get_array(argv[0]);
  int axis = -1;
  VALUE stream_val = Qnil;
  if (argc >= 2 && !NIL_P(argv[1])) axis = NUM2INT(argv[1]);
  if (argc == 3) stream_val = argv[2];
  GET_STREAM(stream, stream_val);
  RETURN_ARRAY(mx::argsort(a, axis, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// partition(a, kth, axis=-1, stream=None)
static VALUE ops_partition(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("partition", 2, 4);
  mx::array& a = get_array(argv[0]);
  int kth = NUM2INT(argv[1]);
  int axis = -1;
  VALUE stream_val = Qnil;
  if (argc >= 3 && !NIL_P(argv[2])) axis = NUM2INT(argv[2]);
  if (argc == 4) stream_val = argv[3];
  GET_STREAM(stream, stream_val);
  RETURN_ARRAY(mx::partition(a, kth, axis, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// argpartition(a, kth, axis=-1, stream=None)
static VALUE ops_argpartition(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("argpartition", 2, 4);
  mx::array& a = get_array(argv[0]);
  int kth = NUM2INT(argv[1]);
  int axis = -1;
  VALUE stream_val = Qnil;
  if (argc >= 3 && !NIL_P(argv[2])) axis = NUM2INT(argv[2]);
  if (argc == 4) stream_val = argv[3];
  GET_STREAM(stream, stream_val);
  RETURN_ARRAY(mx::argpartition(a, kth, axis, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// topk(a, k, axis=-1, stream=None)
static VALUE ops_topk(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("topk", 2, 4);
  mx::array& a = get_array(argv[0]);
  int k = NUM2INT(argv[1]);
  int axis = -1;
  VALUE stream_val = Qnil;
  if (argc >= 3 && !NIL_P(argv[2])) axis = NUM2INT(argv[2]);
  if (argc == 4) stream_val = argv[3];
  GET_STREAM(stream, stream_val);
  RETURN_ARRAY(mx::topk(a, k, axis, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// broadcast_to(a, shape, stream=None)
static VALUE ops_broadcast_to(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("broadcast_to", 2, 3);
  SCALAR_OR_ARRAY(a, argv[0]);
  std::vector<int> shape = ruby_array_to_shape(argv[1]);
  GET_STREAM(stream, (argc == 3) ? argv[2] : Qnil);
  RETURN_ARRAY(mx::broadcast_to(a, shape, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// broadcast_arrays(*arrays, stream=None)
static VALUE ops_broadcast_arrays(int argc, VALUE* argv, VALUE self) {
  // We have a variable number of arrays, last one might be stream?
  // We'll detect if final arg is a Stream or Device; if so, remove it from the list
  if (argc < 1) {
    rb_raise(rb_eArgError, "broadcast_arrays: no arguments given");
  }
  std::vector<mx::array> arrays;
  mx::StreamOrDevice stream;

  // Check last arg
  int up_to = argc;
  if (rb_obj_is_kind_of(argv[argc-1], rb_path2class("MLX::Stream")) ||
      rb_obj_is_kind_of(argv[argc-1], rb_path2class("MLX::Device"))) {
    stream = get_stream_or_device(argv[argc-1]);
    up_to = argc-1;
  }
  arrays.reserve(up_to);
  for (int i=0; i<up_to; i++) {
    SCALAR_OR_ARRAY(tmp, argv[i]);
    arrays.push_back(tmp);
  }
  auto out = mx::broadcast_arrays(arrays, stream);
  // pack into Ruby array of arrays
  VALUE ret = rb_ary_new2((long)out.size());
  for (size_t i=0; i<out.size(); i++) {
    rb_ary_store(ret, i, wrap_array(out[i]));
  }
  return ret;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// softmax(a, axis=None, precise=false, stream=None)
static VALUE ops_softmax(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 4) {
    rb_raise(rb_eArgError, "softmax: wrong number of args (given %d, expected 1..4)", argc);
  }
  mx::array& arr = get_array(argv[0]);
  int axis = -999999;
  bool precise = false;
  VALUE stream_val = Qnil;

  if (argc >= 2 && !NIL_P(argv[1])) axis = NUM2INT(argv[1]);
  if (argc >= 3) precise = (argv[2] == Qtrue);
  if (argc == 4) stream_val = argv[3];
  GET_STREAM(stream, stream_val);

  if (axis == -999999) {
    RETURN_ARRAY(mx::softmax(arr, {}, precise, stream));
  } else {
    RETURN_ARRAY(mx::softmax(arr, {axis}, precise, stream));
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// concatenate([arrays], axis=0, stream=None)
// alias: concat(...)
static VALUE ops_concatenate(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 2) {
    rb_raise(rb_eArgError, "concatenate: wrong number of args (given %d, expected 1..2)", argc);
  }
  VALUE arrays_val = argv[0];
  int axis = 0;
  VALUE stream_val = Qnil;
  if (argc == 2) {
    // Could be axis or stream
    // We'll guess it's axis if it's a fixnum, else it's a stream
    // But in your Python binding, we see "axis=None=0" is default. We'll just do:
    // We expect a shape or fixnum. If it's not an int, might be a Stream -> not typical usage though.
    // We'll keep it simple: if it's an integer, that's axis. Otherwise no stream usage here
    rb_raise(rb_eTypeError, "concatenate: usage is (arrays, [axis]) in this simplified approach");
  }
  // parse arrays
  if (RB_TYPE_P(arrays_val, T_ARRAY)) {
    // good
  } else {
    rb_raise(rb_eTypeError, "concatenate: first arg must be an array of MLX::Arrays");
  }
  // gather them
  std::vector<mx::array> arr_vec;
  long n = RARRAY_LEN(arrays_val);
  arr_vec.reserve(n);
  for (long i=0; i<n; i++) {
    VALUE elt = rb_ary_entry(arrays_val, i);
    arr_vec.push_back(get_array(elt));
  }
  // Actually do the op
  mx::array out = mx::concatenate(arr_vec, axis, {});
  RETURN_ARRAY(out);
}

// Because "concat" is just an alias, we can do:
static VALUE ops_concat(int argc, VALUE* argv, VALUE self) {
  return ops_concatenate(argc, argv, self);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// stack([arrays], axis=0, stream=None)
static VALUE ops_stack(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("stack", 1, 3);
  VALUE arr_list = argv[0];
  int axis = 0;
  VALUE stream_val = Qnil;
  if (argc >= 2) axis = NUM2INT(argv[1]);
  if (argc == 3) stream_val = argv[2];
  GET_STREAM(stream, stream_val);

  if (RB_TYPE_P(arr_list, T_ARRAY)) {
    long n = RARRAY_LEN(arr_list);
    std::vector<mx::array> arrays; arrays.reserve(n);
    for (long i=0; i<n; i++) {
      arrays.push_back(get_array(rb_ary_entry(arr_list, i)));
    }
    RETURN_ARRAY(mx::stack(arrays, axis, stream));
  } else {
    rb_raise(rb_eTypeError, "stack: first arg must be array of arrays");
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// meshgrid(*arrays, sparse=False, indexing="xy", stream=None)
static VALUE ops_meshgrid(int argc, VALUE* argv, VALUE self) {
  // Because Ruby doesn't have *args the same way, we'll do a minimal parse:
  // Suppose usage: meshgrid([arr1, arr2, ...], sparse=false, indexing="xy", stream=nil)
  if (argc < 1 || argc > 4) {
    rb_raise(rb_eArgError, "meshgrid: wrong number of args (given %d, expected 1..4)", argc);
  }
  VALUE arrays_val = argv[0];
  bool sparse = false;
  std::string indexing = "xy";
  VALUE stream_val = Qnil;
  if (argc >= 2) sparse = (argv[1] == Qtrue);
  if (argc >= 3 && RB_TYPE_P(argv[2], T_STRING)) {
    indexing = StringValueCStr(argv[2]);
  }
  if (argc == 4) stream_val = argv[3];
  GET_STREAM(stream, stream_val);

  // parse arrays
  if (RB_TYPE_P(arrays_val, T_ARRAY)) {
    long n = RARRAY_LEN(arrays_val);
    std::vector<mx::array> in_arrays; in_arrays.reserve(n);
    for (long i=0; i<n; i++) {
      in_arrays.push_back(get_array(rb_ary_entry(arrays_val, i)));
    }
    auto out = mx::meshgrid(in_arrays, sparse, indexing, stream);
    // returns vector<mx::array>
    VALUE ret = rb_ary_new2((long)out.size());
    for (size_t i=0; i<out.size(); i++) {
      rb_ary_store(ret, i, wrap_array(out[i]));
    }
    return ret;
  } else {
    rb_raise(rb_eTypeError, "meshgrid: first arg must be array of arrays");
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// repeat(a, repeats, axis=None, stream=None)
static VALUE ops_repeat(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("repeat", 2, 4);
  mx::array& arr = get_array(argv[0]);
  int repeats = NUM2INT(argv[1]);
  int axis = -999999;
  VALUE stream_val = Qnil;
  if (argc >= 3 && !NIL_P(argv[2])) axis = NUM2INT(argv[2]);
  if (argc == 4) stream_val = argv[3];
  GET_STREAM(stream, stream_val);

  if (axis == -999999) {
    RETURN_ARRAY(mx::repeat(arr, repeats, stream));
  } else {
    RETURN_ARRAY(mx::repeat(arr, repeats, axis, stream));
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// clip(a, a_min, a_max, stream=None)
static VALUE ops_clip(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 4) {
    rb_raise(rb_eArgError, "clip: wrong number of args (given %d, expected 1..4)", argc);
  }
  mx::array& arr = get_array(argv[0]);
  VALUE min_val = Qnil;
  VALUE max_val = Qnil;
  VALUE stream_val = Qnil;
  if (argc >= 2) min_val = argv[1];
  if (argc >= 3) max_val = argv[2];
  if (argc == 4) stream_val = argv[3];

  std::optional<mx::array> amn = std::nullopt;
  std::optional<mx::array> amx = std::nullopt;

  if (!NIL_P(min_val)) {
    SCALAR_OR_ARRAY(tmp_min, min_val);
    amn = tmp_min;
  }
  if (!NIL_P(max_val)) {
    SCALAR_OR_ARRAY(tmp_max, max_val);
    amx = tmp_max;
  }
  GET_STREAM(stream, stream_val);
  RETURN_ARRAY(mx::clip(arr, amn, amx, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// pad(a, pad_width, mode="constant", constant_values=0, stream=None)
static VALUE ops_pad(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("pad", 2, 5);
  mx::array& a = get_array(argv[0]);
  VALUE padw_val = argv[1];
  const char* mode = (argc >= 3 && RB_TYPE_P(argv[2], T_STRING)) ? StringValueCStr(argv[2]) : "constant";
  VALUE cv_val = Qnil; // default=0
  VALUE stream_val = Qnil;

  if (argc >= 4) cv_val = argv[3];
  if (argc == 5) stream_val = argv[4];
  SCALAR_OR_ARRAY(cval, (NIL_P(cv_val) ? INT2NUM(0) : cv_val));
  GET_STREAM(stream, stream_val);

  // The Python code supports many forms of pad_width, from int to pairs, etc.
  // For brevity here, let's only handle int or array-of-pairs.
  if (FIXNUM_P(padw_val)) {
    int pw = NUM2INT(padw_val);
    RETURN_ARRAY(mx::pad(a, pw, cval, mode, stream));
  } else if (RB_TYPE_P(padw_val, T_ARRAY)) {
    // We might try to parse as vector<std::pair<int,int>>...
    // For simplicity, assume user gave an array of pairs: [[l0,r0], [l1,r1], ...]
    std::vector<std::pair<int,int>> pad_vec;
    long n = RARRAY_LEN(padw_val);
    for (long i=0; i<n; i++) {
      VALUE sub = rb_ary_entry(padw_val, i);
      if (!RB_TYPE_P(sub, T_ARRAY) || RARRAY_LEN(sub) != 2) {
        rb_raise(rb_eTypeError, "pad_width: each element must be [int, int]");
      }
      int left = NUM2INT(rb_ary_entry(sub, 0));
      int right= NUM2INT(rb_ary_entry(sub, 1));
      pad_vec.push_back(std::make_pair(left, right));
    }
    RETURN_ARRAY(mx::pad(a, pad_vec, cval, mode, stream));
  } else {
    rb_raise(rb_eTypeError, "pad: pad_width must be int or array-of-pairs");
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// as_strided(a, shape=None, strides=None, offset=0, stream=None)
static VALUE ops_as_strided(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 5) {
    rb_raise(rb_eArgError, "as_strided: wrong number of args (given %d, expected 1..5)", argc);
  }
  mx::array& arr = get_array(argv[0]);
  VALUE shape_val = Qnil;
  VALUE strides_val = Qnil;
  size_t offset = 0;
  VALUE stream_val = Qnil;
  if (argc >= 2) shape_val = argv[1];
  if (argc >= 3) strides_val = argv[2];
  if (argc >= 4) offset = NUM2SIZET(argv[3]);
  if (argc == 5) stream_val = argv[4];
  GET_STREAM(stream, stream_val);

  std::vector<int> shape = arr.shape();
  if (!NIL_P(shape_val)) shape = ruby_array_to_shape(shape_val);

  mx::Strides st;
  if (!NIL_P(strides_val)) {
    std::vector<int> raw = ruby_array_to_shape(strides_val);
    st.assign(raw.begin(), raw.end());
  } else {
    // default: compute from shape row-major
    st.resize(shape.size(), 1);
    for (int i=(int)shape.size()-1; i>0; i--) {
      st[i-1] = st[i] * shape[i];
    }
  }
  RETURN_ARRAY(mx::as_strided(arr, shape, st, offset, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// cumsum(a, axis=None, reverse=false, inclusive=true, stream=None)
static VALUE ops_cumsum(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 5) {
    rb_raise(rb_eArgError, "cumsum: wrong number of args (given %d, expected 1..5)", argc);
  }
  mx::array& a = get_array(argv[0]);
  int axis = -999999;
  bool reverse = false;
  bool inclusive = true;
  VALUE stream_val = Qnil;

  if (argc >= 2 && !NIL_P(argv[1])) axis = NUM2INT(argv[1]);
  if (argc >= 3) reverse = (argv[2] == Qtrue);
  if (argc >= 4) inclusive = (argv[3] == Qtrue);
  if (argc == 5) stream_val = argv[4];
  GET_STREAM(stream, stream_val);

  if (axis == -999999) {
    // flatten
    mx::array flat = mx::reshape(a, {-1}, stream);
    RETURN_ARRAY(mx::cumsum(flat, 0, reverse, inclusive, stream));
  } else {
    RETURN_ARRAY(mx::cumsum(a, axis, reverse, inclusive, stream));
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// cumprod(a, axis=None, reverse=false, inclusive=true, stream=None)
static VALUE ops_cumprod(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 5) {
    rb_raise(rb_eArgError, "cumprod: wrong number of args (given %d, expected 1..5)", argc);
  }
  mx::array& a = get_array(argv[0]);
  int axis = -999999;
  bool reverse = false;
  bool inclusive = true;
  VALUE stream_val = Qnil;

  if (argc >= 2 && !NIL_P(argv[1])) axis = NUM2INT(argv[1]);
  if (argc >= 3) reverse = (argv[2] == Qtrue);
  if (argc >= 4) inclusive = (argv[3] == Qtrue);
  if (argc == 5) stream_val = argv[4];
  GET_STREAM(stream, stream_val);

  if (axis == -999999) {
    mx::array flat = mx::reshape(a, {-1}, stream);
    RETURN_ARRAY(mx::cumprod(flat, 0, reverse, inclusive, stream));
  } else {
    RETURN_ARRAY(mx::cumprod(a, axis, reverse, inclusive, stream));
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// cummax(a, axis=None, reverse=false, inclusive=true, stream=None)
static VALUE ops_cummax(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 5) {
    rb_raise(rb_eArgError, "cummax: wrong number of args (given %d, expected 1..5)", argc);
  }
  mx::array& a = get_array(argv[0]);
  int axis = -999999;
  bool reverse = false;
  bool inclusive = true;
  VALUE stream_val = Qnil;
  if (argc >= 2 && !NIL_P(argv[1])) axis = NUM2INT(argv[1]);
  if (argc >= 3) reverse = (argv[2] == Qtrue);
  if (argc >= 4) inclusive = (argv[3] == Qtrue);
  if (argc == 5) stream_val = argv[4];
  GET_STREAM(stream, stream_val);
  if (axis == -999999) {
    mx::array flat = mx::reshape(a, {-1}, stream);
    RETURN_ARRAY(mx::cummax(flat, 0, reverse, inclusive, stream));
  } else {
    RETURN_ARRAY(mx::cummax(a, axis, reverse, inclusive, stream));
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// cummin(a, axis=None, reverse=false, inclusive=true, stream=None)
static VALUE ops_cummin(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 5) {
    rb_raise(rb_eArgError, "cummin: wrong number of args (given %d, expected 1..5)", argc);
  }
  mx::array& a = get_array(argv[0]);
  int axis = -999999;
  bool reverse = false;
  bool inclusive = true;
  VALUE stream_val = Qnil;
  if (argc >= 2 && !NIL_P(argv[1])) axis = NUM2INT(argv[1]);
  if (argc >= 3) reverse = (argv[2] == Qtrue);
  if (argc >= 4) inclusive = (argv[3] == Qtrue);
  if (argc == 5) stream_val = argv[4];
  GET_STREAM(stream, stream_val);
  if (axis == -999999) {
    mx::array flat = mx::reshape(a, {-1}, stream);
    RETURN_ARRAY(mx::cummin(flat, 0, reverse, inclusive, stream));
  } else {
    RETURN_ARRAY(mx::cummin(a, axis, reverse, inclusive, stream));
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// conj(a, stream=None) => mx::conjugate
static VALUE ops_conj(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("conj", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2)? argv[1] : Qnil);
  RETURN_ARRAY(mx::conjugate(a, stream));
}
// alias conjugate
static VALUE ops_conjugate(int argc, VALUE* argv, VALUE self) {
  return ops_conj(argc, argv, self);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// convolve(a, v, mode="full", stream=None)
static VALUE ops_convolve(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 4) {
    rb_raise(rb_eArgError, "convolve: wrong number of args (given %d, expected 2..4)", argc);
  }
  mx::array& a = get_array(argv[0]);
  mx::array& v = get_array(argv[1]);
  const char* mode = "full";
  VALUE stream_val = Qnil;
  if (argc >= 3 && RB_TYPE_P(argv[2], T_STRING)) mode = StringValueCStr(argv[2]);
  if (argc == 4) stream_val = argv[3];
  GET_STREAM(stream, stream_val);
  RETURN_ARRAY(mx::convolve(a, v, mode, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// conv1d(input, weight, stride=1, padding=0, dilation=1, groups=1, stream=None)
static VALUE ops_conv1d(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("conv1d", 2, 7);
  mx::array& inp = get_array(argv[0]);
  mx::array& w   = get_array(argv[1]);
  int stride=1, padding=0, dilation=1, groups=1;
  VALUE stream_val = Qnil;
  if (argc >= 3) stride = NUM2INT(argv[2]);
  if (argc >= 4) padding= NUM2INT(argv[3]);
  if (argc >= 5) dilation=NUM2INT(argv[4]);
  if (argc >= 6) groups=  NUM2INT(argv[5]);
  if (argc == 7) stream_val = argv[6];
  GET_STREAM(stream, stream_val);
  RETURN_ARRAY(mx::conv1d(inp, w, stride, padding, dilation, groups, stream));
}

// Similarly for conv2d, conv3d, conv_transpose1d, etc.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
static VALUE ops_conv2d(int argc, VALUE* argv, VALUE self) {
  // We'll parse the variants similarly, ignoring the specialized variant-int for stride/padding/dilation
  // For brevity, parse them as int,int pairs if the user passes an array or so. We'll do minimal support:
  rb_raise(rb_eNotImpError, "conv2d not fully implemented in this patch example for all param variants");
}
// ... you'd implement the rest (conv3d, conv_transposeX, conv_general, etc.) the same way ...

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// save(file, arr)
static VALUE ops_save(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("save", 2, 2);
  VALUE fname = argv[0];
  mx::array& arr = get_array(argv[1]);
  if (!RB_TYPE_P(fname, T_STRING)) {
    rb_raise(rb_eTypeError, "save: file must be a string");
  }
  const char* path = StringValueCStr(fname);
  mlx_save_helper(path, arr);
  return Qnil;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// savez(file, *args, **kwargs) => For brevity, not fully replicating the Python's named kw
static VALUE ops_savez(int argc, VALUE* argv, VALUE self) {
  // Because Ruby doesn't do the same *args, **kwargs easily, you'd parse them here.
  // We'll just raise not implemented for brevity.
  rb_raise(rb_eNotImpError, "savez: not fully implemented in this example");
}

static VALUE ops_savez_compressed(int argc, VALUE* argv, VALUE self) {
  rb_raise(rb_eNotImpError, "savez_compressed: not fully implemented in this example");
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// load(file, format=None, return_metadata=false, stream=None)
static VALUE ops_load(int argc, VALUE* argv, VALUE self) {
  rb_raise(rb_eNotImpError, "load: not fully implemented in this example");
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// save_safetensors(file, arrays, metadata=None)
static VALUE ops_save_safetensors(int argc, VALUE* argv, VALUE self) {
  rb_raise(rb_eNotImpError, "save_safetensors: not fully implemented in this example");
}

static VALUE ops_save_gguf(int argc, VALUE* argv, VALUE self) {
  rb_raise(rb_eNotImpError, "save_gguf: not fully implemented in this example");
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// where(condition, x, y, stream=None)
static VALUE ops_where(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("where", 3, 4);
  SCALAR_OR_ARRAY(cond, argv[0]);
  SCALAR_OR_ARRAY(x, argv[1]);
  SCALAR_OR_ARRAY(y, argv[2]);
  GET_STREAM(stream, (argc == 4) ? argv[3] : Qnil);
  RETURN_ARRAY(mx::where(cond, x, y, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// nan_to_num(a, nan=0.0, posinf=None, neginf=None, stream=None)
static VALUE ops_nan_to_num(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 5) {
    rb_raise(rb_eArgError, "nan_to_num: wrong number of args (given %d, expected 1..5)", argc);
  }
  SCALAR_OR_ARRAY(a, argv[0]);
  float nanval = 0.0f;
  std::optional<float> posinf, neginf;
  VALUE stream_val = Qnil;

  if (argc >= 2) nanval = NUM2DBL(argv[1]);
  if (argc >= 3 && !NIL_P(argv[2])) posinf = (float)NUM2DBL(argv[2]);
  if (argc >= 4 && !NIL_P(argv[3])) neginf = (float)NUM2DBL(argv[3]);
  if (argc == 5) stream_val = argv[4];

  GET_STREAM(stream, stream_val);
  RETURN_ARRAY(mx::nan_to_num(a, nanval, posinf, neginf, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// round(a, decimals=0, stream=None)
static VALUE ops_round(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("round", 1, 3);
  SCALAR_OR_ARRAY(a, argv[0]);
  int decimals = 0;
  VALUE stream_val = Qnil;
  if (argc >= 2) decimals = NUM2INT(argv[1]);
  if (argc == 3) stream_val = argv[2];
  GET_STREAM(stream, stream_val);
  RETURN_ARRAY(mx::round(a, decimals, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// quantized_matmul(...) etc. We'll do minimal stubs
static VALUE ops_quantized_matmul(int argc, VALUE* argv, VALUE self) {
  rb_raise(rb_eNotImpError, "quantized_matmul not implemented in example");
}
static VALUE ops_quantize(int argc, VALUE* argv, VALUE self) {
  rb_raise(rb_eNotImpError, "quantize not implemented in example");
}
static VALUE ops_dequantize(int argc, VALUE* argv, VALUE self) {
  rb_raise(rb_eNotImpError, "dequantize not implemented in example");
}
static VALUE ops_gather_qmm(int argc, VALUE* argv, VALUE self) {
  rb_raise(rb_eNotImpError, "gather_qmm not implemented in example");
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// tensordot(a, b, axes=2, stream=None)
static VALUE ops_tensordot(int argc, VALUE* argv, VALUE self) {
  // We'll do minimal parse
  rb_raise(rb_eNotImpError, "tensordot not implemented in example");
}

static VALUE ops_inner(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("inner", 2, 3);
  mx::array& a = get_array(argv[0]);
  mx::array& b = get_array(argv[1]);
  GET_STREAM(stream, (argc == 3) ? argv[2] : Qnil);
  RETURN_ARRAY(mx::inner(a, b, stream));
}

static VALUE ops_outer(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("outer", 2, 3);
  mx::array& a = get_array(argv[0]);
  mx::array& b = get_array(argv[1]);
  GET_STREAM(stream, (argc == 3) ? argv[2] : Qnil);
  RETURN_ARRAY(mx::outer(a, b, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// tile(a, reps, stream=None)
static VALUE ops_tile(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("tile", 2, 3);
  mx::array& a = get_array(argv[0]);
  VALUE reps_val = argv[1];
  GET_STREAM(stream, (argc == 3) ? argv[2] : Qnil);
  if (FIXNUM_P(reps_val)) {
    int r = NUM2INT(reps_val);
    RETURN_ARRAY(mx::tile(a, {r}, stream));
  } else if (RB_TYPE_P(reps_val, T_ARRAY)) {
    std::vector<int> reps = ruby_array_to_shape(reps_val);
    RETURN_ARRAY(mx::tile(a, reps, stream));
  } else {
    rb_raise(rb_eTypeError, "tile: reps must be int or int[]");
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// addmm(c, a, b, alpha=1.0, beta=1.0, stream=None)
static VALUE ops_addmm(int argc, VALUE* argv, VALUE self) {
  if (argc < 3 || argc > 6) {
    rb_raise(rb_eArgError, "addmm: wrong number of args (given %d, expected 3..6)", argc);
  }
  mx::array& c = get_array(argv[0]);
  mx::array& a = get_array(argv[1]);
  mx::array& b = get_array(argv[2]);
  float alpha=1.0f, beta=1.0f;
  VALUE stream_val = Qnil;
  if (argc >= 4) alpha = NUM2DBL(argv[3]);
  if (argc >= 5) beta  = NUM2DBL(argv[4]);
  if (argc == 6) stream_val = argv[5];
  GET_STREAM(stream, stream_val);
  RETURN_ARRAY(mx::addmm(c, a, b, alpha, beta, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// block_masked_mm(a, b, block_size=64, mask_out=None, mask_lhs=None, mask_rhs=None, stream=None)
static VALUE ops_block_masked_mm(int argc, VALUE* argv, VALUE self) {
  rb_raise(rb_eNotImpError, "block_masked_mm not implemented in example");
}
static VALUE ops_gather_mm(int argc, VALUE* argv, VALUE self) {
  rb_raise(rb_eNotImpError, "gather_mm not implemented in example");
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// diagonal(a, offset=0, axis1=0, axis2=1, stream=None)
static VALUE ops_diagonal(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 5) {
    rb_raise(rb_eArgError, "diagonal: wrong number of args (given %d, expected 1..5)", argc);
  }
  mx::array& arr = get_array(argv[0]);
  int offset=0, axis1=0, axis2=1;
  VALUE stream_val = Qnil;
  if (argc >= 2) offset = NUM2INT(argv[1]);
  if (argc >= 3) axis1 = NUM2INT(argv[2]);
  if (argc >= 4) axis2 = NUM2INT(argv[3]);
  if (argc == 5) stream_val = argv[4];
  GET_STREAM(stream, stream_val);
  RETURN_ARRAY(mx::diagonal(arr, offset, axis1, axis2, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// diag(a, k=0, stream=None)
static VALUE ops_diag(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("diag", 1, 3);
  mx::array& arr = get_array(argv[0]);
  int k=0;
  VALUE stream_val = Qnil;
  if (argc >= 2) k = NUM2INT(argv[1]);
  if (argc == 3) stream_val = argv[2];
  GET_STREAM(stream, stream_val);
  RETURN_ARRAY(mx::diag(arr, k, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// trace(a, offset=0, axis1=0, axis2=1, dtype=None, stream=None)
static VALUE ops_trace(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 6) {
    rb_raise(rb_eArgError, "trace: wrong number of args (given %d, expected 1..6)", argc);
  }
  mx::array& arr = get_array(argv[0]);
  int offset=0, axis1=0, axis2=1;
  std::optional<mx::Dtype> dt = std::nullopt; // Use optional instead of sentinel value
  VALUE stream_val = Qnil;
  if (argc >= 2) offset = NUM2INT(argv[1]);
  if (argc >= 3) axis1 = NUM2INT(argv[2]);
  if (argc >= 4) axis2 = NUM2INT(argv[3]);
  if (argc >= 5 && !NIL_P(argv[4])) dt = int_to_dtype(NUM2INT(argv[4]));
  if (argc == 6) stream_val = argv[5];
  GET_STREAM(stream, stream_val);

  if (!dt.has_value()) {
    RETURN_ARRAY(mx::trace(arr, offset, axis1, axis2, stream));
  } else {
    RETURN_ARRAY(mx::trace(arr, offset, axis1, axis2, dt.value(), stream));
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// atleast_1d(*arys, stream=None)
// We'll parse (array_of_arrays, stream) or multiple arrays
static VALUE ops_atleast_1d(int argc, VALUE* argv, VALUE self) {
  if (argc < 1) rb_raise(rb_eArgError, "atleast_1d: no arguments");
  mx::StreamOrDevice stream;
  int up_to = argc;
  if (rb_obj_is_kind_of(argv[argc-1], rb_path2class("MLX::Stream")) ||
      rb_obj_is_kind_of(argv[argc-1], rb_path2class("MLX::Device"))) {
    stream = get_stream_or_device(argv[argc-1]);
    up_to = argc-1;
  }
  if (up_to == 1) {
    // single array
    SCALAR_OR_ARRAY(a, argv[0]);
    RETURN_ARRAY(mx::atleast_1d(a, stream));
  } else {
    // multiple
    std::vector<mx::array> in_arrs;
    in_arrs.reserve(up_to);
    for (int i=0; i<up_to; i++) {
      SCALAR_OR_ARRAY(tmp, argv[i]);
      in_arrs.push_back(tmp);
    }
    auto out = mx::atleast_1d(in_arrs, stream);
    VALUE ret = rb_ary_new2((long)out.size());
    for (size_t i=0; i<out.size(); i++) {
      rb_ary_store(ret, i, wrap_array(out[i]));
    }
    return ret;
  }
}

// We do similarly for atleast_2d and atleast_3d, skipping for brevity...

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// issubdtype(arg1, arg2)
static VALUE ops_issubdtype(int argc, VALUE* argv, VALUE self) {
  rb_raise(rb_eNotImpError, "issubdtype not fully implemented in example");
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// bitwise_and(a, b, stream=None)
static VALUE ops_bitwise_and(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("bitwise_and", 2, 3);
  SCALAR_OR_ARRAY(a, argv[0]);
  SCALAR_OR_ARRAY(b, argv[1]);
  GET_STREAM(stream, (argc == 3)? argv[2] : Qnil);
  RETURN_ARRAY(mx::bitwise_and(a, b, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// bitwise_or(a, b, stream=None)
static VALUE ops_bitwise_or(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("bitwise_or", 2, 3);
  SCALAR_OR_ARRAY(a, argv[0]);
  SCALAR_OR_ARRAY(b, argv[1]);
  GET_STREAM(stream, (argc == 3)? argv[2] : Qnil);
  RETURN_ARRAY(mx::bitwise_or(a, b, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// bitwise_xor(a, b, stream=None)
static VALUE ops_bitwise_xor(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("bitwise_xor", 2, 3);
  SCALAR_OR_ARRAY(a, argv[0]);
  SCALAR_OR_ARRAY(b, argv[1]);
  GET_STREAM(stream, (argc == 3)? argv[2] : Qnil);
  RETURN_ARRAY(mx::bitwise_xor(a, b, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// left_shift(a, b, stream=None)
static VALUE ops_left_shift(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("left_shift", 2, 3);
  SCALAR_OR_ARRAY(a, argv[0]);
  SCALAR_OR_ARRAY(b, argv[1]);
  GET_STREAM(stream, (argc == 3)? argv[2] : Qnil);
  RETURN_ARRAY(mx::left_shift(a, b, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// right_shift(a, b, stream=None)
static VALUE ops_right_shift(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("right_shift", 2, 3);
  SCALAR_OR_ARRAY(a, argv[0]);
  SCALAR_OR_ARRAY(b, argv[1]);
  GET_STREAM(stream, (argc == 3)? argv[2] : Qnil);
  RETURN_ARRAY(mx::right_shift(a, b, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// bitwise_invert(a, stream=None)
static VALUE ops_bitwise_invert(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("bitwise_invert", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2)? argv[1] : Qnil);
  RETURN_ARRAY(mx::bitwise_invert(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// view(a, dtype, stream=None)
static VALUE ops_view(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("view", 2, 3);
  SCALAR_OR_ARRAY(a, argv[0]);
  mx::Dtype dt = int_to_dtype(NUM2INT(argv[1]));
  GET_STREAM(stream, (argc == 3) ? argv[2] : Qnil);
  RETURN_ARRAY(mx::view(a, dt, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// hadamard_transform(a, scale=None, stream=None)
static VALUE ops_hadamard_transform(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 3) {
    rb_raise(rb_eArgError, "hadamard_transform: wrong number of args (given %d, expected 1..3)", argc);
  }
  mx::array& a = get_array(argv[0]);
  VALUE scale_val = Qnil;
  VALUE stream_val = Qnil;
  if (argc >= 2) scale_val = argv[1];
  if (argc == 3) stream_val = argv[2];
  float scale = 0.0f;
  bool use_scale = false;
  if (!NIL_P(scale_val)) {
    scale = NUM2DBL(scale_val);
    use_scale = true;
  }
  GET_STREAM(stream, stream_val);
  if (use_scale) {
    RETURN_ARRAY(mx::hadamard_transform(a, scale, stream));
  } else {
    RETURN_ARRAY(mx::hadamard_transform(a, {}, stream));
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// einsum_path(subscripts, *operands)
static VALUE ops_einsum_path(int argc, VALUE* argv, VALUE self) {
  rb_raise(rb_eNotImpError, "einsum_path not implemented in this example");
  return Qnil;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// einsum(subscripts, *operands, stream=None)
static VALUE ops_einsum(int argc, VALUE* argv, VALUE self) {
  rb_raise(rb_eNotImpError, "einsum not implemented in example");
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// roll(a, shift, axis=None, stream=None)
static VALUE ops_roll(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 4) {
    rb_raise(rb_eArgError, "roll: wrong number of args (given %d, expected 2..4)", argc);
  }
  mx::array& arr = get_array(argv[0]);
  VALUE shift_val = argv[1];
  VALUE axis_val = Qnil;
  VALUE stream_val = Qnil;

  if (argc >= 3) axis_val = argv[2];
  if (argc == 4) stream_val = argv[3];
  GET_STREAM(stream, stream_val);

  // parse shift as int or array-of-ints
  std::vector<int> ax;
  // If axis_val not nil, parse similarly
  // We'll do minimal handling:
  int shift = 0;
  if (FIXNUM_P(shift_val)) {
    shift = NUM2INT(shift_val);
    if (NIL_P(axis_val)) {
      RETURN_ARRAY(mx::roll(arr, shift, stream));
    } else {
      // parse axis
      if (FIXNUM_P(axis_val)) {
        int a = NUM2INT(axis_val);
        RETURN_ARRAY(mx::roll(arr, shift, a, stream));
      } else if (RB_TYPE_P(axis_val, T_ARRAY)) {
        std::vector<int> axes = ruby_array_to_shape(axis_val);
        RETURN_ARRAY(mx::roll(arr, shift, axes, stream));
      } else {
        rb_raise(rb_eTypeError, "roll: axis must be int or array of int");
      }
    }
  } else if (RB_TYPE_P(shift_val, T_ARRAY)) {
    // multi shift
    std::vector<int> sh = ruby_array_to_shape(shift_val);
    if (NIL_P(axis_val)) {
      RETURN_ARRAY(mx::roll(arr, sh, stream));
    } else {
      if (FIXNUM_P(axis_val)) {
        int a = NUM2INT(axis_val);
        RETURN_ARRAY(mx::roll(arr, sh, a, stream));
      } else {
        std::vector<int> axes = ruby_array_to_shape(axis_val);
        RETURN_ARRAY(mx::roll(arr, sh, axes, stream));
      }
    }
  } else {
    rb_raise(rb_eTypeError, "roll: shift must be int or array of int");
  }
  return Qnil; // unreachable
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// real(a, stream=None)
static VALUE ops_real(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("real", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2)? argv[1] : Qnil);
  RETURN_ARRAY(mx::real(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// imag(a, stream=None)
static VALUE ops_imag(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("imag", 1, 2);
  SCALAR_OR_ARRAY(a, argv[0]);
  GET_STREAM(stream, (argc == 2)? argv[1] : Qnil);
  RETURN_ARRAY(mx::imag(a, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// slice(a, start_indices, axes, slice_size, stream=None)
static VALUE ops_slice(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("slice", 4, 5);
  mx::array& arr = get_array(argv[0]);
  mx::array& start_idx = get_array(argv[1]);
  std::vector<int> axes = ruby_array_to_shape(argv[2]);
  std::vector<int> sz = ruby_array_to_shape(argv[3]);
  VALUE stream_val = (argc == 5) ? argv[4] : Qnil;
  GET_STREAM(stream, stream_val);
  RETURN_ARRAY(mx::slice(arr, start_idx, axes, sz, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// slice_update(a, update, start_indices, axes, stream=None)
static VALUE ops_slice_update(int argc, VALUE* argv, VALUE self) {
  CHECK_ARITY("slice_update", 4, 5);
  mx::array& src = get_array(argv[0]);
  mx::array& up = get_array(argv[1]);
  mx::array& start_idx = get_array(argv[2]);
  std::vector<int> axes = ruby_array_to_shape(argv[3]);
  VALUE stream_val = (argc == 5) ? argv[4] : Qnil;
  GET_STREAM(stream, stream_val);
  RETURN_ARRAY(mx::slice_update(src, up, start_idx, axes, stream));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// contiguous(a, allow_col_major=false, stream=None)
static VALUE ops_contiguous(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 3) {
    rb_raise(rb_eArgError, "contiguous: wrong number of args (given %d, expected 1..3)", argc);
  }
  mx::array& a = get_array(argv[0]);
  bool allow_col = false;
  VALUE stream_val = Qnil;
  if (argc >= 2) allow_col = (argv[1] == Qtrue);
  if (argc == 3) stream_val = argv[2];
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::contiguous(a, allow_col, stream);
  return wrap_array(result);
}

// Initialize ops module
void init_ops(VALUE module) {
  // Array creation
  rb_define_module_function(module, "zeros", RUBY_METHOD_FUNC(ops_zeros), -1);
  rb_define_module_function(module, "ones", RUBY_METHOD_FUNC(ops_ones), -1);
  rb_define_module_function(module, "full", RUBY_METHOD_FUNC(ops_full), -1);
  rb_define_module_function(module, "arange", RUBY_METHOD_FUNC(ops_arange), -1);
  rb_define_module_function(module, "identity", RUBY_METHOD_FUNC(ops_identity), -1);
  rb_define_module_function(module, "eye", RUBY_METHOD_FUNC(ops_eye), -1);
  
  // Array manipulation
  rb_define_module_function(module, "reshape", RUBY_METHOD_FUNC(ops_reshape), -1);
  rb_define_module_function(module, "flatten", RUBY_METHOD_FUNC(ops_flatten), -1);
  rb_define_module_function(module, "squeeze", RUBY_METHOD_FUNC(ops_squeeze), -1);
  rb_define_module_function(module, "expand_dims", RUBY_METHOD_FUNC(ops_expand_dims), -1);
  
  // Element-wise operations
  rb_define_module_function(module, "abs", RUBY_METHOD_FUNC(ops_abs), -1);
  rb_define_module_function(module, "sign", RUBY_METHOD_FUNC(ops_sign), -1);
  rb_define_module_function(module, "negative", RUBY_METHOD_FUNC(ops_negative), -1);
  
  // Basic operations
  rb_define_module_function(module, "add", RUBY_METHOD_FUNC(ops_add), -1);
  rb_define_module_function(module, "subtract", RUBY_METHOD_FUNC(ops_subtract), -1);
  rb_define_module_function(module, "multiply", RUBY_METHOD_FUNC(ops_multiply), -1);
  rb_define_module_function(module, "divide", RUBY_METHOD_FUNC(ops_divide), -1);
  
  // Comparison operations
  rb_define_module_function(module, "equal", RUBY_METHOD_FUNC(ops_equal), -1);
  rb_define_module_function(module, "not_equal", RUBY_METHOD_FUNC(ops_not_equal), -1);
  rb_define_module_function(module, "greater", RUBY_METHOD_FUNC(ops_greater), -1);
  rb_define_module_function(module, "greater_equal", RUBY_METHOD_FUNC(ops_greater_equal), -1);
  rb_define_module_function(module, "less", RUBY_METHOD_FUNC(ops_less), -1);
  rb_define_module_function(module, "less_equal", RUBY_METHOD_FUNC(ops_less_equal), -1);
  
  rb_define_module_function(module, "stop_gradient", RUBY_METHOD_FUNC(ops_stop_gradient), -1);
  
  // Additional operations
  rb_define_module_function(module, "trace", RUBY_METHOD_FUNC(ops_trace), -1);
  rb_define_module_function(module, "atleast_1d", RUBY_METHOD_FUNC(ops_atleast_1d), -1);
  // likewise for atleast_2d, atleast_3d if you wish
  rb_define_module_function(module, "issubdtype", RUBY_METHOD_FUNC(ops_issubdtype), -1);
  rb_define_module_function(module, "bitwise_and", RUBY_METHOD_FUNC(ops_bitwise_and), -1);
  rb_define_module_function(module, "bitwise_or", RUBY_METHOD_FUNC(ops_bitwise_or), -1);
  rb_define_module_function(module, "bitwise_xor", RUBY_METHOD_FUNC(ops_bitwise_xor), -1);
  rb_define_module_function(module, "left_shift", RUBY_METHOD_FUNC(ops_left_shift), -1);
  rb_define_module_function(module, "right_shift", RUBY_METHOD_FUNC(ops_right_shift), -1);
  rb_define_module_function(module, "bitwise_invert", RUBY_METHOD_FUNC(ops_bitwise_invert), -1);
  rb_define_module_function(module, "view", RUBY_METHOD_FUNC(ops_view), -1);
  rb_define_module_function(module, "hadamard_transform", RUBY_METHOD_FUNC(ops_hadamard_transform), -1);
  rb_define_module_function(module, "einsum_path", RUBY_METHOD_FUNC(ops_einsum_path), -1);
  rb_define_module_function(module, "einsum", RUBY_METHOD_FUNC(ops_einsum), -1);
  rb_define_module_function(module, "roll", RUBY_METHOD_FUNC(ops_roll), -1);
  rb_define_module_function(module, "real", RUBY_METHOD_FUNC(ops_real), -1);
  rb_define_module_function(module, "imag", RUBY_METHOD_FUNC(ops_imag), -1);
  rb_define_module_function(module, "slice", RUBY_METHOD_FUNC(ops_slice), -1);
  rb_define_module_function(module, "slice_update", RUBY_METHOD_FUNC(ops_slice_update), -1);
  rb_define_module_function(module, "contiguous", RUBY_METHOD_FUNC(ops_contiguous), -1);

  rb_define_module_function(module, "unflatten", RUBY_METHOD_FUNC(ops_unflatten), -1);
  rb_define_module_function(module, "divmod", RUBY_METHOD_FUNC(ops_divmod), -1);
  rb_define_module_function(module, "floor_divide", RUBY_METHOD_FUNC(ops_floor_divide), -1);
  rb_define_module_function(module, "remainder", RUBY_METHOD_FUNC(ops_remainder), -1);
  rb_define_module_function(module, "array_equal", RUBY_METHOD_FUNC(ops_array_equal), -1);
  rb_define_module_function(module, "matmul", RUBY_METHOD_FUNC(ops_matmul), -1);
  rb_define_module_function(module, "square", RUBY_METHOD_FUNC(ops_square), -1);
  rb_define_module_function(module, "sqrt", RUBY_METHOD_FUNC(ops_sqrt), -1);
  rb_define_module_function(module, "rsqrt", RUBY_METHOD_FUNC(ops_rsqrt), -1);
  rb_define_module_function(module, "reciprocal", RUBY_METHOD_FUNC(ops_reciprocal), -1);
  rb_define_module_function(module, "logical_not", RUBY_METHOD_FUNC(ops_logical_not), -1);
  rb_define_module_function(module, "logical_and", RUBY_METHOD_FUNC(ops_logical_and), -1);
  rb_define_module_function(module, "logical_or", RUBY_METHOD_FUNC(ops_logical_or), -1);
  rb_define_module_function(module, "logaddexp", RUBY_METHOD_FUNC(ops_logaddexp), -1);
  rb_define_module_function(module, "exp", RUBY_METHOD_FUNC(ops_exp), -1);
  rb_define_module_function(module, "expm1", RUBY_METHOD_FUNC(ops_expm1), -1);
  rb_define_module_function(module, "erf", RUBY_METHOD_FUNC(ops_erf), -1);
  rb_define_module_function(module, "erfinv", RUBY_METHOD_FUNC(ops_erfinv), -1);
  rb_define_module_function(module, "sin", RUBY_METHOD_FUNC(ops_sin), -1);
  rb_define_module_function(module, "cos", RUBY_METHOD_FUNC(ops_cos), -1);
  rb_define_module_function(module, "tan", RUBY_METHOD_FUNC(ops_tan), -1);
  rb_define_module_function(module, "arcsin", RUBY_METHOD_FUNC(ops_arcsin), -1);
  rb_define_module_function(module, "arccos", RUBY_METHOD_FUNC(ops_arccos), -1);
  rb_define_module_function(module, "arctan", RUBY_METHOD_FUNC(ops_arctan), -1);
  rb_define_module_function(module, "arctan2", RUBY_METHOD_FUNC(ops_arctan2), -1);
  rb_define_module_function(module, "sinh", RUBY_METHOD_FUNC(ops_sinh), -1);
  rb_define_module_function(module, "cosh", RUBY_METHOD_FUNC(ops_cosh), -1);
  rb_define_module_function(module, "tanh", RUBY_METHOD_FUNC(ops_tanh), -1);
  rb_define_module_function(module, "arcsinh", RUBY_METHOD_FUNC(ops_arcsinh), -1);
  rb_define_module_function(module, "arccosh", RUBY_METHOD_FUNC(ops_arccosh), -1);
  rb_define_module_function(module, "arctanh", RUBY_METHOD_FUNC(ops_arctanh), -1);
  rb_define_module_function(module, "degrees", RUBY_METHOD_FUNC(ops_degrees), -1);
  rb_define_module_function(module, "radians", RUBY_METHOD_FUNC(ops_radians), -1);
  rb_define_module_function(module, "log", RUBY_METHOD_FUNC(ops_log), -1);
  rb_define_module_function(module, "log2", RUBY_METHOD_FUNC(ops_log2), -1);
  rb_define_module_function(module, "log10", RUBY_METHOD_FUNC(ops_log10), -1);
  rb_define_module_function(module, "log1p", RUBY_METHOD_FUNC(ops_log1p), -1);
  rb_define_module_function(module, "sigmoid", RUBY_METHOD_FUNC(ops_sigmoid), -1);
  rb_define_module_function(module, "power", RUBY_METHOD_FUNC(ops_power), -1);
  rb_define_module_function(module, "linspace", RUBY_METHOD_FUNC(ops_linspace), -1);
  rb_define_module_function(module, "kron", RUBY_METHOD_FUNC(ops_kron), -1);
  rb_define_module_function(module, "take", RUBY_METHOD_FUNC(ops_take), -1);
  rb_define_module_function(module, "take_along_axis", RUBY_METHOD_FUNC(ops_take_along_axis), -1);
  rb_define_module_function(module, "put_along_axis", RUBY_METHOD_FUNC(ops_put_along_axis), -1);
  rb_define_module_function(module, "zeros_like", RUBY_METHOD_FUNC(ops_zeros_like), -1);
  rb_define_module_function(module, "ones_like", RUBY_METHOD_FUNC(ops_ones_like), -1);
  rb_define_module_function(module, "tri", RUBY_METHOD_FUNC(ops_tri), -1);
  rb_define_module_function(module, "tril", RUBY_METHOD_FUNC(ops_tril), -1);
  rb_define_module_function(module, "triu", RUBY_METHOD_FUNC(ops_triu), -1);
  rb_define_module_function(module, "allclose", RUBY_METHOD_FUNC(ops_allclose), -1);
  rb_define_module_function(module, "isclose", RUBY_METHOD_FUNC(ops_isclose), -1);
  rb_define_module_function(module, "all", RUBY_METHOD_FUNC(ops_all), -1);
  rb_define_module_function(module, "any", RUBY_METHOD_FUNC(ops_any), -1);
  rb_define_module_function(module, "minimum", RUBY_METHOD_FUNC(ops_minimum), -1);
  rb_define_module_function(module, "maximum", RUBY_METHOD_FUNC(ops_maximum), -1);
  rb_define_module_function(module, "floor", RUBY_METHOD_FUNC(ops_floor), -1);
  rb_define_module_function(module, "ceil", RUBY_METHOD_FUNC(ops_ceil), -1);
  rb_define_module_function(module, "isnan", RUBY_METHOD_FUNC(ops_isnan), -1);
  rb_define_module_function(module, "isinf", RUBY_METHOD_FUNC(ops_isinf), -1);
  rb_define_module_function(module, "isfinite", RUBY_METHOD_FUNC(ops_isfinite), -1);
  rb_define_module_function(module, "isposinf", RUBY_METHOD_FUNC(ops_isposinf), -1);
  rb_define_module_function(module, "isneginf", RUBY_METHOD_FUNC(ops_isneginf), -1);
  rb_define_module_function(module, "moveaxis", RUBY_METHOD_FUNC(ops_moveaxis), -1);
  rb_define_module_function(module, "swapaxes", RUBY_METHOD_FUNC(ops_swapaxes), -1);
  rb_define_module_function(module, "transpose", RUBY_METHOD_FUNC(ops_transpose), -1);
  rb_define_module_function(module, "permute_dims", RUBY_METHOD_FUNC(ops_permute_dims), -1);
  rb_define_module_function(module, "sum", RUBY_METHOD_FUNC(ops_sum), -1);
  rb_define_module_function(module, "prod", RUBY_METHOD_FUNC(ops_prod), -1);
  rb_define_module_function(module, "min", RUBY_METHOD_FUNC(ops_min), -1);
  rb_define_module_function(module, "max", RUBY_METHOD_FUNC(ops_max), -1);
  rb_define_module_function(module, "logsumexp", RUBY_METHOD_FUNC(ops_logsumexp), -1);
  rb_define_module_function(module, "mean", RUBY_METHOD_FUNC(ops_mean), -1);
  rb_define_module_function(module, "var", RUBY_METHOD_FUNC(ops_var), -1);
  rb_define_module_function(module, "std", RUBY_METHOD_FUNC(ops_std), -1);
  rb_define_module_function(module, "split", RUBY_METHOD_FUNC(ops_split), -1);
  rb_define_module_function(module, "argmin", RUBY_METHOD_FUNC(ops_argmin), -1);
  rb_define_module_function(module, "argmax", RUBY_METHOD_FUNC(ops_argmax), -1);
  rb_define_module_function(module, "sort", RUBY_METHOD_FUNC(ops_sort), -1);
  rb_define_module_function(module, "argsort", RUBY_METHOD_FUNC(ops_argsort), -1);
  rb_define_module_function(module, "partition", RUBY_METHOD_FUNC(ops_partition), -1);
  rb_define_module_function(module, "argpartition", RUBY_METHOD_FUNC(ops_argpartition), -1);
  rb_define_module_function(module, "topk", RUBY_METHOD_FUNC(ops_topk), -1);
  rb_define_module_function(module, "broadcast_to", RUBY_METHOD_FUNC(ops_broadcast_to), -1);
  rb_define_module_function(module, "broadcast_arrays", RUBY_METHOD_FUNC(ops_broadcast_arrays), -1);
  rb_define_module_function(module, "softmax", RUBY_METHOD_FUNC(ops_softmax), -1);
  rb_define_module_function(module, "concatenate", RUBY_METHOD_FUNC(ops_concatenate), -1);
  rb_define_module_function(module, "concat", RUBY_METHOD_FUNC(ops_concat), -1);
  rb_define_module_function(module, "stack", RUBY_METHOD_FUNC(ops_stack), -1);
  rb_define_module_function(module, "meshgrid", RUBY_METHOD_FUNC(ops_meshgrid), -1);
  rb_define_module_function(module, "repeat", RUBY_METHOD_FUNC(ops_repeat), -1);
  rb_define_module_function(module, "clip", RUBY_METHOD_FUNC(ops_clip), -1);
  rb_define_module_function(module, "pad", RUBY_METHOD_FUNC(ops_pad), -1);
  rb_define_module_function(module, "as_strided", RUBY_METHOD_FUNC(ops_as_strided), -1);
  rb_define_module_function(module, "cumsum", RUBY_METHOD_FUNC(ops_cumsum), -1);
  rb_define_module_function(module, "cumprod", RUBY_METHOD_FUNC(ops_cumprod), -1);
  rb_define_module_function(module, "cummax", RUBY_METHOD_FUNC(ops_cummax), -1);
  rb_define_module_function(module, "cummin", RUBY_METHOD_FUNC(ops_cummin), -1);
  rb_define_module_function(module, "conj", RUBY_METHOD_FUNC(ops_conj), -1);
  rb_define_module_function(module, "conjugate", RUBY_METHOD_FUNC(ops_conjugate), -1);
  rb_define_module_function(module, "convolve", RUBY_METHOD_FUNC(ops_convolve), -1);
  rb_define_module_function(module, "conv1d", RUBY_METHOD_FUNC(ops_conv1d), -1);
  rb_define_module_function(module, "conv2d", RUBY_METHOD_FUNC(ops_conv2d), -1);
  // likewise for conv3d, conv_transpose1d, conv_transpose2d, conv_transpose3d,
  // conv_general => you'd define them the same and register them here
  rb_define_module_function(module, "save", RUBY_METHOD_FUNC(ops_save), -1);
  rb_define_module_function(module, "savez", RUBY_METHOD_FUNC(ops_savez), -1);
  rb_define_module_function(module, "savez_compressed", RUBY_METHOD_FUNC(ops_savez_compressed), -1);
  rb_define_module_function(module, "load", RUBY_METHOD_FUNC(ops_load), -1);
  rb_define_module_function(module, "save_safetensors", RUBY_METHOD_FUNC(ops_save_safetensors), -1);
  rb_define_module_function(module, "save_gguf", RUBY_METHOD_FUNC(ops_save_gguf), -1);
  rb_define_module_function(module, "where", RUBY_METHOD_FUNC(ops_where), -1);
  rb_define_module_function(module, "nan_to_num", RUBY_METHOD_FUNC(ops_nan_to_num), -1);
  rb_define_module_function(module, "round", RUBY_METHOD_FUNC(ops_round), -1);
  rb_define_module_function(module, "quantized_matmul", RUBY_METHOD_FUNC(ops_quantized_matmul), -1);
  rb_define_module_function(module, "quantize", RUBY_METHOD_FUNC(ops_quantize), -1);
  rb_define_module_function(module, "dequantize", RUBY_METHOD_FUNC(ops_dequantize), -1);
  rb_define_module_function(module, "gather_qmm", RUBY_METHOD_FUNC(ops_gather_qmm), -1);
  rb_define_module_function(module, "tensordot", RUBY_METHOD_FUNC(ops_tensordot), -1);
  rb_define_module_function(module, "inner", RUBY_METHOD_FUNC(ops_inner), -1);
  rb_define_module_function(module, "outer", RUBY_METHOD_FUNC(ops_outer), -1);
  rb_define_module_function(module, "tile", RUBY_METHOD_FUNC(ops_tile), -1);
  rb_define_module_function(module, "addmm", RUBY_METHOD_FUNC(ops_addmm), -1);
  rb_define_module_function(module, "block_masked_mm", RUBY_METHOD_FUNC(ops_block_masked_mm), -1);
  rb_define_module_function(module, "gather_mm", RUBY_METHOD_FUNC(ops_gather_mm), -1);
  rb_define_module_function(module, "diagonal", RUBY_METHOD_FUNC(ops_diagonal), -1);
  rb_define_module_function(module, "diag", RUBY_METHOD_FUNC(ops_diag), -1);
  rb_define_module_function(module, "trace", RUBY_METHOD_FUNC(ops_trace), -1);
  rb_define_module_function(module, "atleast_1d", RUBY_METHOD_FUNC(ops_atleast_1d), -1);
  // likewise for atleast_2d, atleast_3d if you wish
  rb_define_module_function(module, "issubdtype", RUBY_METHOD_FUNC(ops_issubdtype), -1);
  rb_define_module_function(module, "bitwise_and", RUBY_METHOD_FUNC(ops_bitwise_and), -1);
  rb_define_module_function(module, "bitwise_or", RUBY_METHOD_FUNC(ops_bitwise_or), -1);
  rb_define_module_function(module, "bitwise_xor", RUBY_METHOD_FUNC(ops_bitwise_xor), -1);
  rb_define_module_function(module, "left_shift", RUBY_METHOD_FUNC(ops_left_shift), -1);
  rb_define_module_function(module, "right_shift", RUBY_METHOD_FUNC(ops_right_shift), -1);
  rb_define_module_function(module, "bitwise_invert", RUBY_METHOD_FUNC(ops_bitwise_invert), -1);
  rb_define_module_function(module, "view", RUBY_METHOD_FUNC(ops_view), -1);
  rb_define_module_function(module, "hadamard_transform", RUBY_METHOD_FUNC(ops_hadamard_transform), -1);
  rb_define_module_function(module, "einsum_path", RUBY_METHOD_FUNC(ops_einsum_path), -1);
  rb_define_module_function(module, "einsum", RUBY_METHOD_FUNC(ops_einsum), -1);
  rb_define_module_function(module, "roll", RUBY_METHOD_FUNC(ops_roll), -1);
  rb_define_module_function(module, "real", RUBY_METHOD_FUNC(ops_real), -1);
  rb_define_module_function(module, "imag", RUBY_METHOD_FUNC(ops_imag), -1);
  rb_define_module_function(module, "slice", RUBY_METHOD_FUNC(ops_slice), -1);
  rb_define_module_function(module, "slice_update", RUBY_METHOD_FUNC(ops_slice_update), -1);
  rb_define_module_function(module, "contiguous", RUBY_METHOD_FUNC(ops_contiguous), -1);
} 