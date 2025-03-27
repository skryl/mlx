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
  Check_Type(output_shapes_val, T_ARRAY);
  std::vector<mx::Shape> output_shapes = ruby_array_to_shape_vector(output_shapes_val);
  
  // Process output dtypes
  Check_Type(output_dtypes_val, T_ARRAY);
  std::vector<mx::Dtype> output_dtypes = ruby_array_to_dtype_vector(output_dtypes_val);
  
  // Process grid and threadgroup
  auto grid = ruby_array_to_tuple3(grid_val);
  auto threadgroup = ruby_array_to_tuple3(threadgroup_val);
  
  // Optional parameters
  std::vector<std::pair<std::string, mx::fast::TemplateArg>> template_args;
  if (!NIL_P(template_args_val)) {
    Check_Type(template_args_val, T_ARRAY);
    
    for (long i = 0; i < RARRAY_LEN(template_args_val); i++) {
      VALUE pair_val = rb_ary_entry(template_args_val, i);
      Check_Type(pair_val, T_ARRAY);
      if (RARRAY_LEN(pair_val) != 2) {
        rb_raise(rb_eArgError, "Template argument pair must have 2 elements");
      }
      
      VALUE name_val = rb_ary_entry(pair_val, 0);
      VALUE arg_val = rb_ary_entry(pair_val, 1);
      Check_Type(name_val, T_STRING);
      std::string name_str = StringValueCStr(name_val);
      
      // Handle different template argument types
      if (RTEST(rb_obj_is_kind_of(arg_val, rb_path2class("TrueClass"))) || 
          RTEST(rb_obj_is_kind_of(arg_val, rb_path2class("FalseClass")))) {
        bool bool_val = RTEST(arg_val);
        template_args.emplace_back(name_str, bool_val);
      } else if (RB_TYPE_P(arg_val, T_FIXNUM)) {
        int int_val = NUM2INT(arg_val);
        template_args.emplace_back(name_str, int_val);
      } else if (rb_obj_is_kind_of(arg_val, rb_path2class("MLX::Core::Dtype"))) {
        mx::Dtype* dtype_ptr;
        Data_Get_Struct(arg_val, mx::Dtype, dtype_ptr);
        template_args.emplace_back(name_str, *dtype_ptr);
      } else {
        rb_raise(rb_eTypeError, "Template argument must be a boolean, integer, or Dtype");
      }
    }
  }
  
  // Optional init value
  std::optional<float> init_value;
  if (!NIL_P(init_value_val)) {
    init_value = (float)NUM2DBL(init_value_val);
  }
  
  // Verbose flag
  bool verbose = RTEST(verbose_val);
  
  // Call the kernel function
  auto result_arrays = wrapper->kernel_func(
      inputs,
      output_shapes,
      output_dtypes,
      grid,
      threadgroup,
      template_args,
      init_value,
      verbose,
      {});
  
  // Convert results to Ruby array
  VALUE result = rb_ary_new2(result_arrays.size());
  for (size_t i = 0; i < result_arrays.size(); i++) {
    rb_ary_store(result, i, wrap_array(result_arrays[i]));
  }
  
  return result;
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
  // Required: queries, keys, values, scale
  // Optional: mask
  if (argc < 4 || argc > 6) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 4..6)", argc);
  }
  
  mx::array& q = get_array(argv[0]);
  mx::array& k = get_array(argv[1]);
  mx::array& v = get_array(argv[2]);
  
  double scale_val = NUM2DBL(argv[3]);
  VALUE mask = (argc > 4) ? argv[4] : Qnil;
  VALUE memory_efficient_threshold = (argc > 5) ? argv[5] : Qnil;
  
  if (NIL_P(mask)) {
    // No mask
    if (NIL_P(memory_efficient_threshold)) {
      // No memory threshold
      mx::array result = mx::fast::scaled_dot_product_attention(q, k, v, scale_val);
      return wrap_array(result);
    } else {
      // With memory threshold
      int threshold = NUM2INT(memory_efficient_threshold);
      // Use an empty monostate for the mask
      mx::array result = mx::fast::scaled_dot_product_attention(q, k, v, scale_val, std::monostate{}, threshold);
      return wrap_array(result);
    }
  } else if (RB_TYPE_P(mask, T_STRING)) {
    // String mask (like "causal")
    std::string mask_str = StringValueCStr(mask);
    if (NIL_P(memory_efficient_threshold)) {
      // No memory threshold
      mx::array result = mx::fast::scaled_dot_product_attention(q, k, v, scale_val, mask_str);
      return wrap_array(result);
    } else {
      // With memory threshold
      int threshold = NUM2INT(memory_efficient_threshold);
      mx::array result = mx::fast::scaled_dot_product_attention(q, k, v, scale_val, mask_str, threshold);
      return wrap_array(result);
    }
  } else {
    // Array mask
    mx::array& m = get_array(mask);
    if (NIL_P(memory_efficient_threshold)) {
      // No memory threshold
      mx::array result = mx::fast::scaled_dot_product_attention(q, k, v, scale_val, m);
      return wrap_array(result);
    } else {
      // With memory threshold
      int threshold = NUM2INT(memory_efficient_threshold);
      mx::array result = mx::fast::scaled_dot_product_attention(q, k, v, scale_val, m, threshold);
      return wrap_array(result);
    }
  }
}

// Remove multi_head_attention since it doesn't exist in mx::fast
static VALUE fast_multi_head_attention(int argc, VALUE* argv, VALUE self) {
  rb_raise(rb_eNotImpError, "multi_head_attention is not implemented in mlx::fast");
  return Qnil;
}

static VALUE fast_rms_norm(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  mx::array& arr_x = get_array(argv[0]);
  VALUE weight_val = argv[1];
  
  double eps_val = (argc > 2) ? NUM2DBL(argv[2]) : 1e-5;
  
  if (NIL_P(weight_val)) {
    // No weight
    mx::array result = mx::fast::rms_norm(arr_x, std::nullopt, eps_val);
    return wrap_array(result);
  } else {
    // With weight
    mx::array& arr_w = get_array(weight_val);
    mx::array result = mx::fast::rms_norm(arr_x, arr_w, eps_val);
    return wrap_array(result);
  }
}

static VALUE fast_layer_norm(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..4)", argc);
  }
  
  mx::array& arr_x = get_array(argv[0]);
  VALUE weight_val = argv[1];
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
  
  mx::array result = mx::fast::layer_norm(arr_x, weight_opt, bias_opt, eps_val);
  return wrap_array(result);
}

static VALUE fast_rope(int argc, VALUE* argv, VALUE self) {
  if (argc < 2 || argc > 7) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..7)", argc);
  }
  
  mx::array& arr_x = get_array(argv[0]);
  int dims_val = NUM2INT(argv[1]);
  
  // Default values
  bool traditional_val = (argc > 2) ? RTEST(argv[2]) : false;
  std::optional<float> base_val;
  if (argc > 3 && !NIL_P(argv[3])) {
    base_val = (float)NUM2DBL(argv[3]);
  }
  float scale_val = (argc > 4) ? (float)NUM2DBL(argv[4]) : 1.0f;
  
  // Handle offset: can be int or array
  mx::array offset_arr = mx::array(0); // Initialize with an int value
  if (argc > 5) {
    if (NIL_P(argv[5])) {
      // Use default 0
      offset_arr = mx::array(0);
    } else if (RB_TYPE_P(argv[5], T_FIXNUM)) {
      // Integer offset
      offset_arr = mx::array(NUM2INT(argv[5]));
    } else {
      // Assume array
      offset_arr = get_array(argv[5]);
    }
  } else {
    // Default offset is 0
    offset_arr = mx::array(0);
  }
  
  // Handle freqs: optional array
  std::optional<mx::array> freqs_opt;
  if (argc > 6 && !NIL_P(argv[6])) {
    freqs_opt = get_array(argv[6]);
  }
  
  mx::array result = mx::fast::rope(arr_x, dims_val, traditional_val, base_val, scale_val, offset_arr, freqs_opt);
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