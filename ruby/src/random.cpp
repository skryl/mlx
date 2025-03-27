#include <ruby.h>
#include <chrono>
#include "mlx/random.h"
#include "mlx/ops.h"

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

// Helper to handle "scalar or MLX::Core::Array" the same way Python does.
static mx::array to_mx_array(VALUE obj) {
  if (rb_obj_is_kind_of(obj, rb_path2class("MLX::Core::Array"))) {
    return get_array(obj);
  }
  else if (RB_TYPE_P(obj, T_FIXNUM) || RB_TYPE_P(obj, T_BIGNUM)) {
    // integer -> wrap in an array
    int64_t val = NUM2LL(obj);
    return mx::array((int64_t) val);
  }
  else if (RB_TYPE_P(obj, T_FLOAT)) {
    double val = NUM2DBL(obj);
    return mx::array((double) val);
  }
  else if (NIL_P(obj)) {
    // Some code might pass nil -> treat as None => empty optional
    // Return a default array or handle it differently
    // e.g., raise an error or return an "empty" array
    // For demonstration, let's raise:
    rb_raise(rb_eArgError, "Expected numeric or MLX::Core::Array, got nil");
  }
  else {
    rb_raise(rb_eTypeError, "Expected numeric or MLX::Core::Array");
  }
  // Unreachable
  return mx::array(0.0f);
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

// Helper to extract Ruby array into C++ vector
static std::vector<int> ruby_array_to_vector(VALUE shape) {
  Check_Type(shape, T_ARRAY);
  
  std::vector<int> cpp_shape;
  for (long i = 0; i < RARRAY_LEN(shape); i++) {
    VALUE item = rb_ary_entry(shape, i);
    cpp_shape.push_back(NUM2INT(item));
  }
  
  return cpp_shape;
}

// KeySequence class to manage PRNG keys
class KeySequence {
public:
  explicit KeySequence(uint64_t seed) {
    state = mx::random::key(seed);
  }
  
  void seed(uint64_t seed_val) {
    state = mx::random::key(seed_val);
  }
  
  mx::array next() {
    // Get the split result using pair since the C++ API returns a pair
    auto split_result = mx::random::split(state);
    state = split_result.first;
    return split_result.second;
  }
  
  mx::array state = mx::array(0.0f); // Initialize with a default value
};

// Global default key generator
KeySequence& default_key() {
  auto get_current_time_seed = []() {
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               now.time_since_epoch())
        .count();
  };
  static KeySequence ks(get_current_time_seed());
  return ks;
}

// Convert Ruby key to optional array
static std::optional<mx::array> get_key_opt(VALUE key_obj) {
  if (NIL_P(key_obj)) {
    return std::nullopt;
  }
  return get_array(key_obj);
}

// Random module methods
static VALUE random_seed(VALUE self, VALUE seed) {
  uint64_t seed_val = NUM2ULL(seed);
  default_key().seed(seed_val);
  return Qnil;
}

static VALUE random_key(VALUE self, VALUE seed) {
  uint64_t seed_val = NUM2ULL(seed);
  mx::array key = mx::random::key(seed_val);
  return wrap_array(key);
}

static VALUE random_split(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..3)", argc);
  }
  
  VALUE key_obj = argv[0];
  VALUE num = (argc > 1) ? argv[1] : INT2NUM(2);
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  
  mx::array& key_arr = get_array(key_obj);
  int num_val = NUM2INT(num);
  mx::StreamOrDevice stream = get_stream_or_device(stream_val);
  
  mx::array result = mx::random::split(key_arr, num_val, stream);
  return wrap_array(result);
}

static VALUE random_uniform(int argc, VALUE* argv, VALUE self) {
  if (argc < 0 || argc > 6) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 0..6)", argc);
  }
  
  // Default values
  double low_val = 0.0;
  double high_val = 1.0;
  std::vector<int> shape = {};
  mx::Dtype dtype = mx::float32;
  std::optional<mx::array> key_opt = std::nullopt;
  mx::StreamOrDevice stream = {};
  
  // Parse arguments based on position and count
  if (argc >= 1) low_val = NUM2DBL(argv[0]);
  if (argc >= 2) high_val = NUM2DBL(argv[1]);
  if (argc >= 3) shape = ruby_array_to_vector(argv[2]);
  if (argc >= 4 && !NIL_P(argv[3])) dtype = int_to_dtype(NUM2INT(argv[3]));
  
  // Handle key - use default if not provided
  if (argc >= 5 && !NIL_P(argv[4])) {
    key_opt = get_array(argv[4]);
  }
  
  // Handle stream
  if (argc >= 6) stream = get_stream_or_device(argv[5]);
  
  mx::array low_arr = mx::array(low_val);
  mx::array high_arr = mx::array(high_val);
  
  mx::array result = mx::random::uniform(low_arr, high_arr, shape, dtype, key_opt, stream);
  return wrap_array(result);
}

static VALUE random_normal(int argc, VALUE* argv, VALUE self) {
  if (argc < 0 || argc > 6) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 0..6)", argc);
  }
  
  // Default values
  std::vector<int> shape = {};
  mx::Dtype dtype = mx::float32;
  float loc_val = 0.0f;
  float scale_val = 1.0f;
  std::optional<mx::array> key_opt = std::nullopt;
  mx::StreamOrDevice stream = {};
  
  // Parse arguments based on position and count
  if (argc >= 1 && !NIL_P(argv[0])) shape = ruby_array_to_vector(argv[0]);
  if (argc >= 2 && !NIL_P(argv[1])) dtype = int_to_dtype(NUM2INT(argv[1]));
  if (argc >= 3) loc_val = (float)NUM2DBL(argv[2]);
  if (argc >= 4) scale_val = (float)NUM2DBL(argv[3]);
  
  // Handle key - use default if not provided
  if (argc >= 5 && !NIL_P(argv[4])) {
    key_opt = get_array(argv[4]);
  }
  
  // Handle stream
  if (argc >= 6) stream = get_stream_or_device(argv[5]);
  
  mx::array result = mx::random::normal(shape, dtype, loc_val, scale_val, key_opt, stream);
  return wrap_array(result);
}

static VALUE random_multivariate_normal(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 6) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..6)", argc);
  }
  
  VALUE mean = argv[0];
  VALUE cov = argv[1];
  
  // Default values
  std::vector<int> shape = {};
  mx::Dtype dtype = mx::float32;
  std::optional<mx::array> key_opt = std::nullopt;
  mx::StreamOrDevice stream = {};
  
  // Parse arguments based on position and count
  if (argc >= 3 && !NIL_P(argv[2])) shape = ruby_array_to_vector(argv[2]);
  if (argc >= 4 && !NIL_P(argv[3])) dtype = int_to_dtype(NUM2INT(argv[3]));
  
  // Handle key - use default if not provided
  if (argc >= 5 && !NIL_P(argv[4])) {
    key_opt = get_array(argv[4]);
  }
  
  // Handle stream
  if (argc >= 6) stream = get_stream_or_device(argv[5]);
  
  mx::array& mean_arr = get_array(mean);
  mx::array& cov_arr = get_array(cov);
  
  mx::array result = mx::random::multivariate_normal(mean_arr, cov_arr, shape, dtype, key_opt, stream);
  return wrap_array(result);
}

static VALUE random_randint(int argc, VALUE* argv, VALUE self) {
  if (argc < 0 || argc > 6) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 0..6)", argc);
  }
  
  // Default values
  int low_val = 0;
  int high_val = 2; // Default for binary random variable
  std::vector<int> shape = {};
  mx::Dtype dtype = mx::int32;
  std::optional<mx::array> key_opt = std::nullopt;
  mx::StreamOrDevice stream = {};
  
  // Parse arguments based on position and count
  if (argc >= 1) low_val = NUM2INT(argv[0]);
  if (argc >= 2) high_val = NUM2INT(argv[1]);
  if (argc >= 3 && !NIL_P(argv[2])) {
    // shape can remain shape
    shape = ruby_array_to_vector(argv[2]);
  }
  if (argc >= 4 && !NIL_P(argv[3])) dtype = int_to_dtype(NUM2INT(argv[3]));
  
  // Handle key - use default if not provided
  if (argc >= 5 && !NIL_P(argv[4])) {
    key_opt = get_array(argv[4]);
  }
  
  // Handle stream
  if (argc >= 6) stream = get_stream_or_device(argv[5]);
  
  // Note: The Python randint() handles low/high as "scalar or array".
  // but the Python side can pass either a scalar or array for low/high.
  // To match that, we'd do something like:
  //
  // mx::array low_arr  = to_mx_array(argv[0]);
  // mx::array high_arr = to_mx_array(argv[1]);
  // mx::array result   = mx::random::randint(low_arr, high_arr, shape, dtype, key_opt, stream);
  //
  // For now, we continue with the scalar approach. If you want true parity with
  // Python, replace the lines below with the above approach:

  mx::array result = mx::random::randint(low_val, high_val, shape, dtype, key_opt, stream);
  return wrap_array(result);
}

static VALUE random_bernoulli(int argc, VALUE* argv, VALUE self) {
  if (argc < 0 || argc > 5) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 0..5)", argc);
  }
  
  // Default values
  float p_val = 0.5f;
  std::optional<std::vector<int>> shape_opt = std::nullopt;
  std::optional<mx::array> key_opt = std::nullopt;
  mx::StreamOrDevice stream = {};
  
  // Parse arguments based on position and count
  if (argc >= 1) p_val = (float)NUM2DBL(argv[0]);
  
  // Next argument could be shape or key
  int current_arg = 1;
  
  // Check if we have a shape array
  if (argc > current_arg && !NIL_P(argv[current_arg]) && 
      rb_obj_is_kind_of(argv[current_arg], rb_cArray)) {
    shape_opt = ruby_array_to_vector(argv[current_arg]);
    current_arg++;
  }
  
  // Check if we have a key
  if (argc > current_arg && !NIL_P(argv[current_arg]) &&
      rb_obj_is_kind_of(argv[current_arg], rb_path2class("MLX::Core::Array"))) {
    key_opt = get_array(argv[current_arg]);
    current_arg++;
  }
  
  // Finally, check for stream
  if (argc > current_arg) {
    stream = get_stream_or_device(argv[current_arg]);
  }
  
  // Call the appropriate API function based on arguments
  mx::array p_arr = mx::array(p_val);
  mx::array result = mx::array(0.0f); // Initialize with a valid value
  
  if (shape_opt.has_value()) {
    result = mx::random::bernoulli(p_arr, shape_opt.value(), key_opt, stream);
  } else {
    result = mx::random::bernoulli(p_arr, key_opt, stream);
  }
  
  return wrap_array(result);
}

static VALUE random_truncated_normal(int argc, VALUE* argv, VALUE self) {
  if (argc < 0 || argc > 6) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 0..6)", argc);
  }
  
  // Default values
  float lower_val = -2.0f;
  float upper_val = 2.0f;
  std::vector<int> shape = {};
  mx::Dtype dtype = mx::float32;
  std::optional<mx::array> key_opt = std::nullopt;
  mx::StreamOrDevice stream = {};
  
  // Parse arguments based on position and count
  if (argc >= 1) lower_val = (float)NUM2DBL(argv[0]);
  if (argc >= 2) upper_val = (float)NUM2DBL(argv[1]);
  if (argc >= 3 && !NIL_P(argv[2])) shape = ruby_array_to_vector(argv[2]);
  if (argc >= 4 && !NIL_P(argv[3])) dtype = int_to_dtype(NUM2INT(argv[3]));
  
  // Handle key - use default if not provided
  if (argc >= 5 && !NIL_P(argv[4])) {
    key_opt = get_array(argv[4]);
  }
  
  // Handle stream
  if (argc >= 6) stream = get_stream_or_device(argv[5]);
  
  mx::array lower_arr = mx::array(lower_val);
  mx::array upper_arr = mx::array(upper_val);
  
  mx::array result = mx::random::truncated_normal(lower_arr, upper_arr, shape, dtype, key_opt, stream);
  return wrap_array(result);
}

static VALUE random_categorical(int argc, VALUE* argv, VALUE self) {
  // Python signature:
  //   def categorical(logits, axis=-1, shape=None, num_samples=None, key=None, stream=None)
  // We want the same pattern. So we parse up to 5 or 6 arguments:
  //
  // Ruby side: "logits, axis, shape, num_samples, key, stream"
  if (argc < 1) {
    rb_raise(rb_eArgError, "categorical requires at least 'logits' argument");
  }

  VALUE logits_obj   = argv[0];
  VALUE axis_obj     = (argc >= 2 ? argv[1] : Qnil);
  VALUE shape_obj    = (argc >= 3 ? argv[2] : Qnil);
  VALUE n_samp_obj   = (argc >= 4 ? argv[3] : Qnil);
  VALUE key_obj      = (argc >= 5 ? argv[4] : Qnil);
  VALUE stream_obj   = (argc >= 6 ? argv[5] : Qnil);

  mx::array& logits_arr = get_array(logits_obj);
  int axis = NIL_P(axis_obj) ? -1 : NUM2INT(axis_obj);

  // shape or num_samples?
  bool have_shape      = (!NIL_P(shape_obj));
  bool have_num_samp   = (!NIL_P(n_samp_obj));
  if (have_shape && have_num_samp) {
    rb_raise(rb_eArgError, "At most one of shape or num_samples can be specified.");
  }

  std::optional<mx::array> key_opt = get_key_opt(key_obj);
  mx::StreamOrDevice stream = get_stream_or_device(stream_obj);

  if (have_shape) {
    // shape is a Ruby array
    std::vector<int> shape_vec = ruby_array_to_vector(shape_obj);
    return wrap_array(mx::random::categorical(logits_arr, axis, shape_vec, key_opt, stream));
  } else if (have_num_samp) {
    int ns = NUM2INT(n_samp_obj);
    return wrap_array(mx::random::categorical(logits_arr, axis, ns, key_opt, stream));
  } else {
    // neither shape nor num_samples => single-sample distribution
    return wrap_array(mx::random::categorical(logits_arr, axis, key_opt, stream));
  }
}

static VALUE random_gumbel(int argc, VALUE* argv, VALUE self) {
  if (argc < 0 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 0..4)", argc);
  }
  
  // Default values
  std::vector<int> shape = {};
  mx::Dtype dtype = mx::float32;
  std::optional<mx::array> key_opt = std::nullopt;
  mx::StreamOrDevice stream = {};
  
  // Parse arguments based on position and count
  if (argc >= 1 && !NIL_P(argv[0])) shape = ruby_array_to_vector(argv[0]);
  if (argc >= 2 && !NIL_P(argv[1])) dtype = int_to_dtype(NUM2INT(argv[1]));
  
  // Handle key - use default if not provided
  if (argc >= 3 && !NIL_P(argv[2])) {
    key_opt = get_array(argv[2]);
  }
  
  // Handle stream
  if (argc >= 4) stream = get_stream_or_device(argv[3]);
  
  mx::array result = mx::random::gumbel(shape, dtype, key_opt, stream);
  return wrap_array(result);
}

static VALUE random_laplace(int argc, VALUE* argv, VALUE self) {
  if (argc < 0 || argc > 6) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 0..6)", argc);
  }
  
  // Default values
  std::vector<int> shape = {};
  mx::Dtype dtype = mx::float32;
  float loc_val = 0.0f;
  float scale_val = 1.0f;
  std::optional<mx::array> key_opt = std::nullopt;
  mx::StreamOrDevice stream = {};
  
  // Parse arguments based on position and count
  if (argc >= 1 && !NIL_P(argv[0])) shape = ruby_array_to_vector(argv[0]);
  if (argc >= 2 && !NIL_P(argv[1])) dtype = int_to_dtype(NUM2INT(argv[1]));
  if (argc >= 3) loc_val = (float)NUM2DBL(argv[2]);
  if (argc >= 4) scale_val = (float)NUM2DBL(argv[3]);
  
  // Handle key - use default if not provided
  if (argc >= 5 && !NIL_P(argv[4])) {
    key_opt = get_array(argv[4]);
  }
  
  // Handle stream
  if (argc >= 6) stream = get_stream_or_device(argv[5]);
  
  mx::array result = mx::random::laplace(shape, dtype, loc_val, scale_val, key_opt, stream);
  return wrap_array(result);
}

static VALUE random_permutation(int argc, VALUE* argv, VALUE self) {
  if (argc < 1 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..4)", argc);
  }
  
  VALUE x = argv[0];
  
  // Default values
  int axis = 0;
  std::optional<mx::array> key_opt = std::nullopt;
  mx::StreamOrDevice stream = {};
  
  // Parse arguments based on position and count
  if (argc >= 2) axis = NUM2INT(argv[1]);
  
  // Skip the 'independent' parameter, it's not part of the API
  // Start looking for key at position 2
  if (argc >= 3 && !NIL_P(argv[2])) {
    key_opt = get_array(argv[2]);
  }
  
  // Handle stream
  if (argc >= 4) stream = get_stream_or_device(argv[3]);
  
  // Handle x as either an array or a number
  mx::array result = mx::array(0.0f); // Initialize with a valid value
  
  if (rb_obj_is_kind_of(x, rb_path2class("MLX::Core::Array"))) {
    mx::array& x_arr = get_array(x);
    result = mx::random::permutation(x_arr, axis, key_opt, stream);
  } else {
    int n = NUM2INT(x);
    result = mx::random::permutation(n, key_opt, stream);
  }
  
  return wrap_array(result);
}

// Expose the default PRNG state, similar to 'm.attr("state") = default_key().state()' in Python.
static VALUE random_state(VALUE self) {
  // The C++ KeySequence in Ruby only has a single mx::array 'state'.
  // Return that as an MLX::Core::Array Ruby object.
  return wrap_array(default_key().state);
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

// Initialize random module
void init_random(VALUE module) {
  // Define module methods
  rb_define_module_function(module, "seed", RUBY_METHOD_FUNC(random_seed), 1);
  rb_define_module_function(module, "key", RUBY_METHOD_FUNC(random_key), 1);
  rb_define_module_function(module, "split", RUBY_METHOD_FUNC(random_split), -1);
  rb_define_module_function(module, "uniform", RUBY_METHOD_FUNC(random_uniform), -1);
  rb_define_module_function(module, "normal", RUBY_METHOD_FUNC(random_normal), -1);
  rb_define_module_function(module, "multivariate_normal", RUBY_METHOD_FUNC(random_multivariate_normal), -1);
  rb_define_module_function(module, "randint", RUBY_METHOD_FUNC(random_randint), -1);
  rb_define_module_function(module, "state", RUBY_METHOD_FUNC(random_state), 0);
  rb_define_module_function(module, "bernoulli", RUBY_METHOD_FUNC(random_bernoulli), -1);
  rb_define_module_function(module, "truncated_normal", RUBY_METHOD_FUNC(random_truncated_normal), -1);
  rb_define_module_function(module, "categorical", RUBY_METHOD_FUNC(random_categorical), -1);
  rb_define_module_function(module, "gumbel", RUBY_METHOD_FUNC(random_gumbel), -1);
  rb_define_module_function(module, "laplace", RUBY_METHOD_FUNC(random_laplace), -1);
  rb_define_module_function(module, "permutation", RUBY_METHOD_FUNC(random_permutation), -1);
} 