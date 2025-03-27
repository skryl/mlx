#include <ruby.h>
#include <vector>
#include "mlx/ops.h"
#include "mlx_func.h"

namespace mx = mlx::core;

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
} 