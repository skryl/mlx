#include <ruby.h>
#include <optional>
#include <string>
#include <vector>
#include <tuple>
#include <memory>
#include <unordered_map>
#include "mlx/fast.h"
#include "mlx/ops.h"

namespace mx = mlx::core;

// Minimal helper to parse an optional 'stream' or device argument from Ruby.
// You will likely want to expand this logic to handle objects that wrap streams/devices.
static mx::StreamOrDevice parse_stream_or_device(VALUE v) {
  if (NIL_P(v)) {
    // No stream
    return mx::StreamOrDevice{};
  }
  else if (RB_TYPE_P(v, T_FIXNUM)) {
    // Interpret a fixnum as a device index
    int device_id = NUM2INT(v);
    // e.g. create a device from the ID
    return mx::StreamOrDevice(device_id);
  }
  else if (RB_TYPE_P(v, T_FLOAT)) {
    // Arbitrary example: raise an error if it's a float
    rb_raise(rb_eArgError, "Cannot interpret float as a stream/device");
  }
  else {
    // If you have a real 'MLX::Core::Stream' or 'MLX::Core::Device' class, you'd check for it here.
    // e.g. if (rb_obj_is_kind_of(v, rb_path2class("MLX::Core::Stream"))) { ... }
    // else if (rb_obj_is_kind_of(v, rb_path2class("MLX::Core::Device"))) { ... }
    // For now, error out.
    rb_raise(rb_eArgError, "Unsupported stream/device argument");
  }
  // Fallback
  return mx::StreamOrDevice{};
}

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

// Helper to extract Shape from Ruby array
static mx::Shape ruby_array_to_shape(VALUE arr) {
  Check_Type(arr, T_ARRAY);
  
  std::vector<int> dims;
  for (long i = 0; i < RARRAY_LEN(arr); i++) {
    VALUE item = rb_ary_entry(arr, i);
    dims.push_back(NUM2INT(item));
  }
  
  return mx::Shape(dims);
}

// Helper to convert Ruby array to vector of mx::arrays
static std::vector<mx::array> ruby_array_to_array_vector(VALUE arr) {
  Check_Type(arr, T_ARRAY);
  
  std::vector<mx::array> result;
  for (long i = 0; i < RARRAY_LEN(arr); i++) {
    VALUE item = rb_ary_entry(arr, i);
    result.push_back(get_array(item));
  }
  
  return result;
}

// Helper to convert Ruby array to vector of shapes
static std::vector<mx::Shape> ruby_array_to_shape_vector(VALUE arr) {
  Check_Type(arr, T_ARRAY);
  
  std::vector<mx::Shape> result;
  for (long i = 0; i < RARRAY_LEN(arr); i++) {
    VALUE item = rb_ary_entry(arr, i);
    result.push_back(ruby_array_to_shape(item));
  }
  
  return result;
}

// Helper to convert Ruby array to vector of dtypes
static std::vector<mx::Dtype> ruby_array_to_dtype_vector(VALUE arr) {
  Check_Type(arr, T_ARRAY);
  
  std::vector<mx::Dtype> result;
  for (long i = 0; i < RARRAY_LEN(arr); i++) {
    VALUE item = rb_ary_entry(arr, i);
    mx::Dtype* dtype_ptr;
    Data_Get_Struct(item, mx::Dtype, dtype_ptr);
    result.push_back(*dtype_ptr);
  }
  
  return result;
}

// Helper to convert Ruby array to 3-tuple
static std::tuple<int, int, int> ruby_array_to_tuple3(VALUE arr) {
  Check_Type(arr, T_ARRAY);
  
  if (RARRAY_LEN(arr) != 3) {
    rb_raise(rb_eArgError, "Expected array of length 3, got %ld", RARRAY_LEN(arr));
  }
  
  int x = NUM2INT(rb_ary_entry(arr, 0));
  int y = NUM2INT(rb_ary_entry(arr, 1));
  int z = NUM2INT(rb_ary_entry(arr, 2));
  
  return std::make_tuple(x, y, z);
}

// Fast module methods
static VALUE fast_gemm(int argc, VALUE* argv, VALUE self) {
  rb_raise(rb_eNotImpError, "gemm is not implemented in mlx::fast");
  return Qnil;
}

// Metal kernel wrapper
typedef struct {
  std::function<std::vector<mx::array>(
      const std::vector<mx::array>&,
      const std::vector<mx::Shape>&,
      const std::vector<mx::Dtype>&,
      std::tuple<int, int, int>,
      std::tuple<int, int, int>,
      const std::vector<std::pair<std::string, mx::fast::TemplateArg>>&,
      std::optional<float>,
      bool,
      mx::StreamOrDevice)> kernel_func;
} MetalKernelWrapper;

// Free callback for MetalKernelWrapper
static void metal_kernel_free(void* ptr) {
  MetalKernelWrapper* wrapper = static_cast<MetalKernelWrapper*>(ptr);
  delete wrapper;
}

// C callback function for Ruby proc
static VALUE metal_kernel_callback(VALUE self, VALUE args) {
  VALUE data = rb_iv_get(self, "@metal_kernel_wrapper");
  MetalKernelWrapper* wrapper;
  Data_Get_Struct(data, MetalKernelWrapper, wrapper);
  
  // Parse args
  VALUE inputs_val = rb_hash_aref(args, ID2SYM(rb_intern("inputs")));
  VALUE output_shapes_val = rb_hash_aref(args, ID2SYM(rb_intern("output_shapes")));
  VALUE output_dtypes_val = rb_hash_aref(args, ID2SYM(rb_intern("output_dtypes")));
  VALUE grid_val = rb_hash_aref(args, ID2SYM(rb_intern("grid")));
  VALUE threadgroup_val = rb_hash_aref(args, ID2SYM(rb_intern("threadgroup")));
  VALUE template_args_val = rb_hash_aref(args, ID2SYM(rb_intern("template")));
  VALUE init_value_val = rb_hash_aref(args, ID2SYM(rb_intern("init_value")));
  VALUE verbose_val = rb_hash_aref(args, ID2SYM(rb_intern("verbose")));
  VALUE stream_val = rb_hash_aref(args, ID2SYM(rb_intern("stream")));
  
  // We need at least inputs, output_shapes, output_dtypes, grid, and threadgroup
  if (NIL_P(inputs_val) || NIL_P(output_shapes_val) || NIL_P(output_dtypes_val) ||
      NIL_P(grid_val) || NIL_P(threadgroup_val)) {
    rb_raise(rb_eArgError, "Required arguments missing: inputs, output_shapes, output_dtypes, grid, threadgroup");
  }
  
  // Process inputs
  Check_Type(inputs_val, T_ARRAY);
  std::vector<mx::array> inputs;
  for (long i = 0; i < RARRAY_LEN(inputs_val); i++) {
    VALUE item = rb_ary_entry(inputs_val, i);
    if (rb_obj_is_kind_of(item, rb_path2class("MLX::Core::Array"))) {
      inputs.push_back(get_array(item));
    } else if (RB_TYPE_P(item, T_FIXNUM) || RB_TYPE_P(item, T_FLOAT)) {
      // Convert scalars to arrays
      double scalar_val = NUM2DBL(item);
      inputs.push_back(mx::array(scalar_val));
    } else {
      rb_raise(rb_eTypeError, "Expected array or scalar, got %s", rb_obj_classname(item));
    }
  }
  
  // Process output shapes
  std::vector<mx::Shape> output_shapes = ruby_array_to_shape_vector(output_shapes_val);
  
  // Process output dtypes
  std::vector<mx::Dtype> output_dtypes = ruby_array_to_dtype_vector(output_dtypes_val);
  
  // Process grid and threadgroup
  std::tuple<int, int, int> grid = ruby_array_to_tuple3(grid_val);
  std::tuple<int, int, int> threadgroup = ruby_array_to_tuple3(threadgroup_val);
  
  // Process template args (optional)
  std::vector<std::pair<std::string, mx::fast::TemplateArg>> template_args;
  if (!NIL_P(template_args_val)) {
    Check_Type(template_args_val, T_HASH);
    // Iterate over hash
    rb_hash_foreach(template_args_val, [](VALUE key, VALUE val, VALUE arg) {
      std::vector<std::pair<std::string, mx::fast::TemplateArg>>* template_args = 
        reinterpret_cast<std::vector<std::pair<std::string, mx::fast::TemplateArg>>*>(arg);
      
      // Key must be a string/symbol
      std::string key_str;
      if (RB_TYPE_P(key, T_STRING)) {
        key_str = StringValueCStr(key);
      } else if (RB_TYPE_P(key, T_SYMBOL)) {
        key_str = rb_id2name(SYM2ID(key));
      } else {
        rb_raise(rb_eTypeError, "Template key must be string or symbol");
      }
      
      // Value can be int, float, or bool
      mx::fast::TemplateArg arg_val;
      if (RB_TYPE_P(val, T_FIXNUM)) {
        arg_val = NUM2INT(val);
      } else if (RB_TYPE_P(val, T_FLOAT)) {
        arg_val = NUM2DBL(val);
      } else if (val == Qtrue || val == Qfalse) {
        arg_val = (val == Qtrue);
      } else {
        rb_raise(rb_eTypeError, "Template value must be int, float, or bool");
      }
      
      template_args->push_back(std::make_pair(key_str, arg_val));
      return ST_CONTINUE;
    }, reinterpret_cast<VALUE>(&template_args));
  }
  
  // Init value (optional)
  std::optional<float> init_value;
  if (!NIL_P(init_value_val)) {
    init_value = (float)NUM2DBL(init_value_val);
  }
  
  // Verbose (optional)
  bool verbose = NIL_P(verbose_val) ? false : RTEST(verbose_val);
  
  // Stream (optional)
  mx::StreamOrDevice stream = parse_stream_or_device(stream_val);
  
  // Call the kernel function
  std::vector<mx::array> results;
  try {
    results = wrapper->kernel_func(inputs, output_shapes, output_dtypes, grid, threadgroup, 
                                   template_args, init_value, verbose, stream);
  } catch (const std::exception& e) {
    rb_raise(rb_eRuntimeError, "Metal kernel execution failed: %s", e.what());
  }
  
  // Convert results to Ruby array
  VALUE rb_results = rb_ary_new2(results.size());
  for (size_t i = 0; i < results.size(); i++) {
    rb_ary_store(rb_results, i, wrap_array(results[i]));
  }
  
  return rb_results;
}

// Function to create a new metal kernel
static VALUE fast_metal_kernel(VALUE self, VALUE name, VALUE input_names, VALUE output_names, 
                                VALUE source, VALUE header, VALUE ensure_row_contiguous, VALUE atomic_outputs) {
  Check_Type(name, T_STRING);
  Check_Type(input_names, T_ARRAY);
  Check_Type(output_names, T_ARRAY);
  Check_Type(source, T_STRING);
  
  std::string name_str = StringValueCStr(name);
  std::string source_str = StringValueCStr(source);
  std::string header_str = "";
  
  if (!NIL_P(header)) {
    Check_Type(header, T_STRING);
    header_str = StringValueCStr(header);
  }
  
  bool ensure_contiguous = RTEST(ensure_row_contiguous);
  bool atomic_out = RTEST(atomic_outputs);
  
  // Convert Ruby arrays to C++ vectors
  std::vector<std::string> input_names_vec;
  for (long i = 0; i < RARRAY_LEN(input_names); i++) {
    VALUE item = rb_ary_entry(input_names, i);
    Check_Type(item, T_STRING);
    input_names_vec.push_back(StringValueCStr(item));
  }
  
  std::vector<std::string> output_names_vec;
  for (long i = 0; i < RARRAY_LEN(output_names); i++) {
    VALUE item = rb_ary_entry(output_names, i);
    Check_Type(item, T_STRING);
    output_names_vec.push_back(StringValueCStr(item));
  }
  
  // Create the metal kernel
  auto kernel = mx::fast::metal_kernel(
      name_str,
      input_names_vec,
      output_names_vec,
      source_str,
      header_str,
      ensure_contiguous,
      atomic_out);
  
  // Create a wrapper function to call the kernel
  auto wrapper = new MetalKernelWrapper();
  wrapper->kernel_func = std::move(kernel);
  
  // Wrap the kernel wrapper in a Ruby data object
  VALUE data = Data_Wrap_Struct(rb_cObject, 0, metal_kernel_free, wrapper);
  
  // Create a Ruby Proc that calls the kernel through our C callback
  VALUE proc = rb_proc_new(RUBY_METHOD_FUNC(metal_kernel_callback), Qnil);
  
  // Attach the wrapper to the proc as data
  rb_iv_set(proc, "@metal_kernel_wrapper", data);
  
  return proc;
}

static VALUE fast_scaled_dot_product_attention(int argc, VALUE* argv, VALUE self) {
  // Python: scaled_dot_product_attention(
  //     q, k, v, scale=None, mask=None, *, causal=False, 
  //     dropout=None, dropout_p=0.0, stream=None)
  // -> We need at least (q, k, v), optional scale, mask, causal, dropout_seed, dropout_p, stream.
  // In Ruby, 3..9 arguments.
  
  if (argc < 3 || argc > 9) {
    rb_raise(rb_eArgError,
             "wrong number of arguments (given %d, expected 3..9). "
             "Signature: scaled_dot_product_attention(q, k, v, scale=nil, mask=nil, "
             "causal=false, dropout_seed=nil, dropout_p=0.0, stream=nil)",
             argc);
  }

  mx::array& q = get_array(argv[0]);
  mx::array& k = get_array(argv[1]);
  mx::array& v = get_array(argv[2]);
  
  // Optional args
  std::optional<float> scale_opt;
  if (argc > 3 && !NIL_P(argv[3])) {
    scale_opt = (float)NUM2DBL(argv[3]);
  }
  
  std::optional<mx::array> mask_opt;
  if (argc > 4 && !NIL_P(argv[4])) {
    mask_opt = get_array(argv[4]);
  }
  
  bool causal = (argc > 5) ? RTEST(argv[5]) : false;
  
  std::optional<int> dropout_seed;
  if (argc > 6 && !NIL_P(argv[6])) {
    dropout_seed = NUM2INT(argv[6]);
  }
  
  float dropout_p = (argc > 7) ? (float)NUM2DBL(argv[7]) : 0.0f;
  
  // Optional stream param
  VALUE stream_val = (argc > 8) ? argv[8] : Qnil;
  mx::StreamOrDevice s = parse_stream_or_device(stream_val);
  
  mx::array result = mx::fast::scaled_dot_product_attention(
      q, k, v, scale_opt, mask_opt, causal, dropout_seed, dropout_p, s);
  
  return wrap_array(result);
}

// Remove multi_head_attention since it doesn't exist in mx::fast
static VALUE fast_multi_head_attention(int argc, VALUE* argv, VALUE self) {
  rb_raise(rb_eNotImpError, "multi_head_attention is not implemented in mlx::fast");
  return Qnil;
}

static VALUE fast_rms_norm(int argc, VALUE* argv, VALUE self) {
  // Python: rms_norm(x, weight=None, eps, *, stream=None)
  // -> We need at least (x, weight, eps), optional stream.
  // So total 3..4 arguments in Ruby.
  if (argc < 3 || argc > 4) {
    rb_raise(rb_eArgError,
             "wrong number of arguments (given %d, expected 3..4). "
             "Signature: rms_norm(x, weight=nil, eps, stream=nil)",
             argc);
  }

  mx::array& arr_x = get_array(argv[0]);
  VALUE weight_val = argv[1];
  double eps_val = NUM2DBL(argv[2]);
  std::optional<mx::array> weight_opt;
  if (NIL_P(weight_val)) {
    weight_opt = std::nullopt;
  } else {
    weight_opt = get_array(weight_val);  
  }

  // Optional 4th param: stream
  VALUE stream_val = (argc > 3) ? argv[3] : Qnil;
  mx::StreamOrDevice s = parse_stream_or_device(stream_val);

  mx::array result = mx::fast::rms_norm(arr_x, weight_opt, eps_val, s);
  return wrap_array(result);
}

static VALUE fast_layer_norm(int argc, VALUE* argv, VALUE self) {
  // Python: layer_norm(x, weight=None, bias=None, eps=1e-5, *, stream=None)
  // -> We need at least (x), optional weight, bias, eps, stream.
  // So total 1..5 arguments in Ruby.
  if (argc < 1 || argc > 5) {
    rb_raise(rb_eArgError,
             "wrong number of arguments (given %d, expected 1..5). "
             "Signature: layer_norm(x, weight=nil, bias=nil, eps=1e-5, stream=nil)",
             argc);
  }

  mx::array& arr_x = get_array(argv[0]);
  
  VALUE weight_val = (argc > 1) ? argv[1] : Qnil;
  VALUE bias_val = (argc > 2) ? argv[2] : Qnil;
  double eps_val = (argc > 3) ? NUM2DBL(argv[3]) : 1e-5;
  
  std::optional<mx::array> weight_opt;
  std::optional<mx::array> bias_opt;
  
  if (!NIL_P(weight_val)) {
    weight_opt = get_array(weight_val);
  }

  if (!NIL_P(bias_val)) {
    bias_opt = get_array(bias_val);
  }
  
  // Optional 5th param: stream
  VALUE stream_val = (argc > 4) ? argv[4] : Qnil;
  mx::StreamOrDevice s = parse_stream_or_device(stream_val);

  mx::array result = mx::fast::layer_norm(arr_x, weight_opt, bias_opt, eps_val, s);
  return wrap_array(result);
}

static VALUE fast_rope(int argc, VALUE* argv, VALUE self) {
  // Python: rope(x, dims, traditional=False, base=10000.0, dynamic=True, *, stream=None)
  // -> So total 1..6 arguments in Ruby.
  if (argc < 2 || argc > 6) {
    rb_raise(rb_eArgError,
             "wrong number of arguments (given %d, expected 2..6). "
             "Signature: rope(x, dims, traditional=false, base=10000.0, dynamic=true, stream=nil)",
             argc);
  }

  mx::array& arr_x = get_array(argv[0]);
  int64_t rope_dims = NUM2INT(argv[1]);
  bool traditional = (argc > 2) ? RTEST(argv[2]) : false;
  double base = (argc > 3) ? NUM2DBL(argv[3]) : 10000.0;
  bool dynamic = (argc > 4) ? RTEST(argv[4]) : true;
  
  // Optional 6th param: stream
  VALUE stream_val = (argc > 5) ? argv[5] : Qnil;
  mx::StreamOrDevice s = parse_stream_or_device(stream_val);

  mx::array result = mx::fast::rope(arr_x, rope_dims, traditional, base, dynamic, s);
  return wrap_array(result);
}

// Remove rope_inplace since it doesn't exist in mx::fast
static VALUE fast_rope_inplace(int argc, VALUE* argv, VALUE self) {
  rb_raise(rb_eNotImpError, "rope_inplace is not implemented in mlx::fast");
  return Qnil;
}

// Initialize fast module
void init_fast(VALUE module) {
  // Define module functions
  rb_define_module_function(module, "gemm", RUBY_METHOD_FUNC(fast_gemm), -1);
  rb_define_module_function(module, "scaled_dot_product_attention", RUBY_METHOD_FUNC(fast_scaled_dot_product_attention), -1);
  rb_define_module_function(module, "multi_head_attention", RUBY_METHOD_FUNC(fast_multi_head_attention), -1);
  rb_define_module_function(module, "rms_norm", RUBY_METHOD_FUNC(fast_rms_norm), -1);
  rb_define_module_function(module, "layer_norm", RUBY_METHOD_FUNC(fast_layer_norm), -1);
  rb_define_module_function(module, "rope", RUBY_METHOD_FUNC(fast_rope), -1);
  rb_define_module_function(module, "rope_inplace", RUBY_METHOD_FUNC(fast_rope_inplace), -1);
  rb_define_module_function(module, "metal_kernel", RUBY_METHOD_FUNC(fast_metal_kernel), 7);
} 