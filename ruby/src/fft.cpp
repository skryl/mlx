#include <ruby.h>
#include <vector>
#include <numeric>
#include "mlx/fft.h"
#include "mlx/ops.h"

namespace mx = mlx::core;

// Helper to extract mx::array from Ruby VALUE
static mx::array& get_array(VALUE obj) {
  mx::array* arr_ptr;
  Data_Get_Struct(obj, mx::array, arr_ptr);
  return *arr_ptr;
}

// Helper function to wrap mx::array into Ruby VALUE
static VALUE wrap_array(const mx::array& arr) {
  return Data_Wrap_Struct(rb_path2class("MLX::Core::Array"), 0, nullptr, new mx::array(arr));
}

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

// Helper to extract stream from Ruby object
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

// FFT module methods
static VALUE fft_fft(int argc, VALUE* argv, VALUE self) {
  // Required: array
  // Optional: n (int), axis (int), stream
  if (argc < 1 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..4)", argc);
  }
  
  mx::array& a = get_array(argv[0]);
  VALUE n = (argc > 1) ? argv[1] : Qnil;
  int axis = (argc > 2) ? NUM2INT(argv[2]) : -1;
  VALUE stream_val = (argc > 3) ? argv[3] : Qnil;
  
  // Extract stream if provided
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  if (NIL_P(n)) {
    return wrap_array(mx::fft::fft(a, axis, stream));
  } else {
    return wrap_array(mx::fft::fft(a, NUM2INT(n), axis, stream));
  }
}

static VALUE fft_ifft(int argc, VALUE* argv, VALUE self) {
  // Required: array
  // Optional: n (int), axis (int), stream
  if (argc < 1 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..4)", argc);
  }
  
  mx::array& a = get_array(argv[0]);
  VALUE n = (argc > 1) ? argv[1] : Qnil;
  int axis = (argc > 2) ? NUM2INT(argv[2]) : -1;
  VALUE stream_val = (argc > 3) ? argv[3] : Qnil;
  
  // Extract stream if provided
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  if (NIL_P(n)) {
    return wrap_array(mx::fft::ifft(a, axis, stream));
  } else {
    return wrap_array(mx::fft::ifft(a, NUM2INT(n), axis, stream));
  }
}

static VALUE fft_rfft(int argc, VALUE* argv, VALUE self) {
  // Required: array
  // Optional: n (int), axis (int), stream
  if (argc < 1 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..4)", argc);
  }
  
  mx::array& a = get_array(argv[0]);
  VALUE n = (argc > 1) ? argv[1] : Qnil;
  int axis = (argc > 2) ? NUM2INT(argv[2]) : -1;
  VALUE stream_val = (argc > 3) ? argv[3] : Qnil;
  
  // Extract stream if provided
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  if (NIL_P(n)) {
    return wrap_array(mx::fft::rfft(a, axis, stream));
  } else {
    return wrap_array(mx::fft::rfft(a, NUM2INT(n), axis, stream));
  }
}

static VALUE fft_irfft(int argc, VALUE* argv, VALUE self) {
  // Required: array
  // Optional: n (int), axis (int), stream
  if (argc < 1 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..4)", argc);
  }
  
  mx::array& a = get_array(argv[0]);
  VALUE n = (argc > 1) ? argv[1] : Qnil;
  int axis = (argc > 2) ? NUM2INT(argv[2]) : -1;
  VALUE stream_val = (argc > 3) ? argv[3] : Qnil;
  
  // Extract stream if provided
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  if (NIL_P(n)) {
    return wrap_array(mx::fft::irfft(a, axis, stream));
  } else {
    return wrap_array(mx::fft::irfft(a, NUM2INT(n), axis, stream));
  }
}

static VALUE fft_fft2(int argc, VALUE* argv, VALUE self) {
  // Required: array
  // Optional: s (shape), axes (array of ints), stream
  if (argc < 1 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..4)", argc);
  }
  
  mx::array& a = get_array(argv[0]);
  VALUE s = (argc > 1) ? argv[1] : Qnil;
  VALUE axes = (argc > 2) ? argv[2] : Qnil;
  VALUE stream_val = (argc > 3) ? argv[3] : Qnil;
  
  // Extract stream if provided
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  // Default axes for fft2 is [-2, -1]
  std::vector<int> default_axes = {-2, -1};
  
  if (!NIL_P(s) && NIL_P(axes)) {
    // Match Python: raise error if s is given but axes is None
    rb_raise(rb_eArgError, "[fft2] `axes` should not be nil if `s` is not nil.");
  } else if (NIL_P(s) && NIL_P(axes)) {
    // Use default axes
    return wrap_array(mx::fft::fftn(a, default_axes, stream));
  }
  else if (NIL_P(s) && !NIL_P(axes)) {
    // Axes provided but no shape
    std::vector<int> ax = ruby_array_to_vector(axes);
    return wrap_array(mx::fft::fftn(a, ax, stream));
  } else {
    // Both shape and axes provided
    std::vector<int> shape = ruby_array_to_vector(s);
    std::vector<int> ax = ruby_array_to_vector(axes);
    return wrap_array(mx::fft::fftn(a, shape, ax, stream));
  }
}

static VALUE fft_ifft2(int argc, VALUE* argv, VALUE self) {
  // Required: array
  // Optional: s (shape), axes (array of ints), stream
  if (argc < 1 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..4)", argc);
  }
  
  mx::array& a = get_array(argv[0]);
  VALUE s = (argc > 1) ? argv[1] : Qnil;
  VALUE axes = (argc > 2) ? argv[2] : Qnil;
  VALUE stream_val = (argc > 3) ? argv[3] : Qnil;
  
  // Extract stream if provided
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  // Default axes for ifft2 is [-2, -1]
  std::vector<int> default_axes = {-2, -1};
  
  if (!NIL_P(s) && NIL_P(axes)) {
    rb_raise(rb_eArgError, "[ifft2] `axes` should not be nil if `s` is not nil.");
  } else if (NIL_P(s) && NIL_P(axes)) {
    // Use default axes
    return wrap_array(mx::fft::ifftn(a, default_axes, stream));
  } 
  else if (NIL_P(s) && !NIL_P(axes)) {
    std::vector<int> ax = ruby_array_to_vector(axes);
    return wrap_array(mx::fft::ifftn(a, ax, stream));
  } else {
    // Both shape and axes provided
    std::vector<int> shape = ruby_array_to_vector(s);
    std::vector<int> ax = ruby_array_to_vector(axes);
    return wrap_array(mx::fft::ifftn(a, shape, ax, stream));
  }
}

static VALUE fft_rfft2(int argc, VALUE* argv, VALUE self) {
  // Required: array
  // Optional: s (shape), axes (array of ints), stream
  if (argc < 1 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..4)", argc);
  }
  
  mx::array& a = get_array(argv[0]);
  VALUE s = (argc > 1) ? argv[1] : Qnil;
  VALUE axes = (argc > 2) ? argv[2] : Qnil;
  VALUE stream_val = (argc > 3) ? argv[3] : Qnil;
  
  // Extract stream if provided
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  // Default axes for rfft2 is [-2, -1]
  std::vector<int> default_axes = {-2, -1};
  
  if (!NIL_P(s) && NIL_P(axes)) {
    rb_raise(rb_eArgError, "[rfft2] `axes` should not be nil if `s` is not nil.");
  } else if (NIL_P(s) && NIL_P(axes)) {
    return wrap_array(mx::fft::rfftn(a, default_axes, stream));
  }
  else if (NIL_P(s) && !NIL_P(axes)) {
    std::vector<int> ax = ruby_array_to_vector(axes);
    return wrap_array(mx::fft::rfftn(a, ax, stream));
  } else {
    std::vector<int> shape = ruby_array_to_vector(s);
    std::vector<int> ax = ruby_array_to_vector(axes);
    return wrap_array(mx::fft::rfftn(a, shape, ax, stream));
  }
}

static VALUE fft_irfft2(int argc, VALUE* argv, VALUE self) {
  // Required: array
  // Optional: s (shape), axes (array of ints), stream
  if (argc < 1 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..4)", argc);
  }
  
  mx::array& a = get_array(argv[0]);
  VALUE s = (argc > 1) ? argv[1] : Qnil;
  VALUE axes = (argc > 2) ? argv[2] : Qnil;
  VALUE stream_val = (argc > 3) ? argv[3] : Qnil;
  
  // Extract stream if provided
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  // Default axes for irfft2 is [-2, -1]
  std::vector<int> default_axes = {-2, -1};
  
  if (!NIL_P(s) && NIL_P(axes)) {
    rb_raise(rb_eArgError, "[irfft2] `axes` should not be nil if `s` is not nil.");
  } else if (NIL_P(s) && NIL_P(axes)) {
    return wrap_array(mx::fft::irfftn(a, default_axes, stream));
  }
  else if (NIL_P(s) && !NIL_P(axes)) {
    std::vector<int> ax = ruby_array_to_vector(axes);
    return wrap_array(mx::fft::irfftn(a, ax, stream));
  } else {
    std::vector<int> shape = ruby_array_to_vector(s);
    std::vector<int> ax = ruby_array_to_vector(axes);
    return wrap_array(mx::fft::irfftn(a, shape, ax, stream));
  }
}

static VALUE fft_fftn(int argc, VALUE* argv, VALUE self) {
  // Required: array
  // Optional: s (shape), axes (array of ints), stream
  if (argc < 1 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..4)", argc);
  }
  
  mx::array& a = get_array(argv[0]);
  VALUE s = (argc > 1) ? argv[1] : Qnil;
  VALUE axes = (argc > 2) ? argv[2] : Qnil;
  VALUE stream_val = (argc > 3) ? argv[3] : Qnil;
  
  // Extract stream if provided
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  if (NIL_P(s) && NIL_P(axes)) {
    return wrap_array(mx::fft::fftn(a, stream));
  } else if (!NIL_P(s) && NIL_P(axes)) {
    rb_raise(rb_eArgError, "[fftn] `axes` should not be nil if `s` is not nil.");
  }
  else if (NIL_P(s) && !NIL_P(axes)) {
    std::vector<int> ax = ruby_array_to_vector(axes);
    return wrap_array(mx::fft::fftn(a, ax, stream));
  } else {
    std::vector<int> shape = ruby_array_to_vector(s);
    std::vector<int> ax = ruby_array_to_vector(axes);
    return wrap_array(mx::fft::fftn(a, shape, ax, stream));
  }
}

static VALUE fft_ifftn(int argc, VALUE* argv, VALUE self) {
  // Required: array
  // Optional: s (shape), axes (array of ints), stream
  if (argc < 1 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..4)", argc);
  }
  
  mx::array& a = get_array(argv[0]);
  VALUE s = (argc > 1) ? argv[1] : Qnil;
  VALUE axes = (argc > 2) ? argv[2] : Qnil;
  VALUE stream_val = (argc > 3) ? argv[3] : Qnil;
  
  // Extract stream if provided
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  if (NIL_P(s) && NIL_P(axes)) {
    return wrap_array(mx::fft::ifftn(a, stream));
  } else if (!NIL_P(s) && NIL_P(axes)) {
    rb_raise(rb_eArgError, "[ifftn] `axes` should not be nil if `s` is not nil.");
  }
  else if (NIL_P(s) && !NIL_P(axes)) {
    std::vector<int> ax = ruby_array_to_vector(axes);
    return wrap_array(mx::fft::ifftn(a, ax, stream));
  } else {
    std::vector<int> shape = ruby_array_to_vector(s);
    std::vector<int> ax = ruby_array_to_vector(axes);
    return wrap_array(mx::fft::ifftn(a, shape, ax, stream));
  }
}

static VALUE fft_rfftn(int argc, VALUE* argv, VALUE self) {
  // Required: array
  // Optional: s (shape), axes (array of ints), stream
  if (argc < 1 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..4)", argc);
  }
  
  mx::array& a = get_array(argv[0]);
  VALUE s = (argc > 1) ? argv[1] : Qnil;
  VALUE axes = (argc > 2) ? argv[2] : Qnil;
  VALUE stream_val = (argc > 3) ? argv[3] : Qnil;
  
  // Extract stream if provided
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  if (NIL_P(s) && NIL_P(axes)) {
    return wrap_array(mx::fft::rfftn(a, stream));
  } else if (!NIL_P(s) && NIL_P(axes)) {
    rb_raise(rb_eArgError, "[rfftn] `axes` should not be nil if `s` is not nil.");
  }
  else if (NIL_P(s) && !NIL_P(axes)) {
    std::vector<int> ax = ruby_array_to_vector(axes);
    return wrap_array(mx::fft::rfftn(a, ax, stream));
  } else {
    std::vector<int> shape = ruby_array_to_vector(s);
    std::vector<int> ax = ruby_array_to_vector(axes);
    return wrap_array(mx::fft::rfftn(a, shape, ax, stream));
  }
}

static VALUE fft_irfftn(int argc, VALUE* argv, VALUE self) {
  // Required: array
  // Optional: s (shape), axes (array of ints), stream
  if (argc < 1 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..4)", argc);
  }
  
  mx::array& a = get_array(argv[0]);
  VALUE s = (argc > 1) ? argv[1] : Qnil;
  VALUE axes = (argc > 2) ? argv[2] : Qnil;
  VALUE stream_val = (argc > 3) ? argv[3] : Qnil;
  
  // Extract stream if provided
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  if (NIL_P(s) && NIL_P(axes)) {
    return wrap_array(mx::fft::irfftn(a, stream));
  } else if (!NIL_P(s) && NIL_P(axes)) {
    rb_raise(rb_eArgError, "[irfftn] `axes` should not be nil if `s` is not nil.");
  }
  else if (NIL_P(s) && !NIL_P(axes)) {
    std::vector<int> ax = ruby_array_to_vector(axes);
    return wrap_array(mx::fft::irfftn(a, ax, stream));
  } else {
    std::vector<int> shape = ruby_array_to_vector(s);
    std::vector<int> ax = ruby_array_to_vector(axes);
    return wrap_array(mx::fft::irfftn(a, shape, ax, stream));
  }
}

// Initialize FFT module
void init_fft(VALUE module) {
  // Define module functions
  rb_define_module_function(module, "fft", RUBY_METHOD_FUNC(fft_fft), -1);
  rb_define_module_function(module, "ifft", RUBY_METHOD_FUNC(fft_ifft), -1);
  rb_define_module_function(module, "rfft", RUBY_METHOD_FUNC(fft_rfft), -1);
  rb_define_module_function(module, "irfft", RUBY_METHOD_FUNC(fft_irfft), -1);
  rb_define_module_function(module, "fft2", RUBY_METHOD_FUNC(fft_fft2), -1);
  rb_define_module_function(module, "ifft2", RUBY_METHOD_FUNC(fft_ifft2), -1);
  rb_define_module_function(module, "rfft2", RUBY_METHOD_FUNC(fft_rfft2), -1);
  rb_define_module_function(module, "irfft2", RUBY_METHOD_FUNC(fft_irfft2), -1);
  rb_define_module_function(module, "fftn", RUBY_METHOD_FUNC(fft_fftn), -1);
  rb_define_module_function(module, "ifftn", RUBY_METHOD_FUNC(fft_ifftn), -1);
  rb_define_module_function(module, "rfftn", RUBY_METHOD_FUNC(fft_rfftn), -1);
  rb_define_module_function(module, "irfftn", RUBY_METHOD_FUNC(fft_irfftn), -1);
} 