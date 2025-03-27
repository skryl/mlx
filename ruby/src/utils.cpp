#include <ruby.h>
#include <vector>
#include <sstream>
#include <functional>
#include <iostream>
#include <numeric>
#include "mlx/utils.h"
#include "mlx/ops.h"
#include "utils.h"
#include "mlx/dtype.h"

namespace mx = mlx::core;

// Forward declaration of internal Ruby class
VALUE rb_cMLXArray = Qnil;

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

// Helper to extract Stream or Device from Ruby VALUE
StreamOrDevice get_stream_or_device(VALUE obj) {
  if (NIL_P(obj)) {
    return std::monostate{}; // Default empty stream/device
  }
  
  // Check if it's a Stream object
  if (rb_obj_is_kind_of(obj, rb_path2class("MLX::Core::Stream"))) {
    typedef struct {
      mx::Stream stream;
    } StreamWrapper;
    
    StreamWrapper* wrapper;
    Data_Get_Struct(obj, StreamWrapper, wrapper);
    return wrapper->stream;
  }
  
  // Check if it's a Device object
  if (rb_obj_is_kind_of(obj, rb_path2class("MLX::Core::Device"))) {
    typedef struct {
      mx::Device device;
    } DeviceWrapper;
    
    DeviceWrapper* wrapper;
    Data_Get_Struct(obj, DeviceWrapper, wrapper);
    return wrapper->device;
  }
  
  // Check if it's a string or symbol representing a device
  if (RB_TYPE_P(obj, T_STRING) || RB_TYPE_P(obj, T_SYMBOL)) {
    const char* dev_str = RB_TYPE_P(obj, T_STRING) ? 
                          StringValueCStr(obj) : 
                          rb_id2name(SYM2ID(obj));
    
    std::string device_str(dev_str);
    
    if (device_str == "cpu") {
      return mx::Device(mx::Device::cpu);
    } else if (device_str == "gpu") {
      return mx::Device(mx::Device::gpu);
    } else {
      rb_raise(rb_eArgError, "Unknown device type: %s", dev_str);
    }
  }
  
  rb_raise(rb_eTypeError, "Expected Stream, Device, or device name");
  return std::monostate{}; // To satisfy compiler
}

// Get reduce axes for reduction operations
std::vector<int> get_reduce_axes(const std::variant<std::monostate, int, std::vector<int>>& v, int dims) {
  std::vector<int> axes;
  if (std::holds_alternative<std::monostate>(v)) {
    axes.resize(dims);
    std::iota(axes.begin(), axes.end(), 0);
  } else if (auto pv = std::get_if<int>(&v); pv) {
    axes.push_back(*pv);
  } else {
    axes = std::get<std::vector<int>>(v);
  }
  return axes;
}

// Helper function that mimics Python's to_array_with_accessor logic
mx::array to_array_with_accessor(VALUE obj, std::optional<mx::Dtype> dtype) {
  // If it's already an MLX::Core::Array, use it:
  if (rb_obj_is_kind_of(obj, rb_path2class("MLX::Core::Array"))) {
    return get_array(obj);
  }

  // If it responds to __mlx_array__, call that method to retrieve an MLX::Core::Array
  if (rb_respond_to(obj, rb_intern("__mlx_array__"))) {
    // obj.__mlx_array__() => presumably returns an MLX::Core::Array
    VALUE arr_obj = rb_funcall(obj, rb_intern("__mlx_array__"), 0);
    // Now convert whatever is returned from __mlx_array__ into a real mx::array
    return to_array(arr_obj, dtype);
  }

  // Otherwise, raise an error
  std::ostringstream msg;
  msg << "Invalid type " << rb_obj_classname(obj)
      << " received in array initialization: no __mlx_array__ accessor.";
  rb_raise(rb_eTypeError, "%s", msg.str().c_str());

  // Unreachable, but for completeness:
  return mx::array(0.0f);
}

// Check whether the Ruby object is an MLX array OR array-like (has __mlx_array__).
static bool is_mlx_array_or_arraylike(VALUE obj) {
  // Is it already an MLX::Core::Array?
  if (rb_obj_is_kind_of(obj, rb_path2class("MLX::Core::Array"))) {
    return true;
  }

  // Does it respond to __mlx_array__?
  if (rb_respond_to(obj, rb_intern("__mlx_array__"))) {
    return true;
  }

  return false;
}

// Convert Ruby VALUE to mx::array
mx::array to_array(VALUE v, std::optional<mx::Dtype> dtype) {
  if (NIL_P(v)) {
    // Return empty array with default dtype
    auto dt = dtype.value_or(mx::float32);
    if (dt == mx::float32) {
      return mx::array(0.0f);
    } else if (dt == mx::float64) {
      return mx::array(0.0);
    } else if (dt == mx::int32) {
      return mx::array(0);
    } else if (dt == mx::int64) {
      return mx::array(0L);
    } else if (dt == mx::bool_) {
      return mx::array(false);
    } else {
      // Default to float32 and convert
      auto arr = mx::array(0.0f);
      return mx::astype(arr, dt);
    }
  }
  
  if (rb_obj_is_kind_of(v, rb_path2class("MLX::Core::Array"))) {
    return get_array(v);
  } else if (RB_TYPE_P(v, T_TRUE) || RB_TYPE_P(v, T_FALSE)) {
    return mx::array(RTEST(v), dtype.value_or(mx::bool_));
  } else if (RB_TYPE_P(v, T_FIXNUM)) {
    long val = NUM2LONG(v);
    auto default_type = (val > std::numeric_limits<int>::max() ||
                         val < std::numeric_limits<int>::min())
        ? mx::int64
        : mx::int32;
    auto out_t = dtype.value_or(default_type);
    // bool_ is an exception and is always promoted
    return mx::array(val, (out_t == mx::bool_) ? mx::int32 : out_t);
  } else if (RB_TYPE_P(v, T_FLOAT)) {
    auto out_t = dtype.value_or(mx::float32);
    return mx::array(
        NUM2DBL(v),
        mx::issubdtype(out_t, mx::floating) ? out_t : mx::float32);
  } else if (RB_TYPE_P(v, T_COMPLEX)) {
    return mx::array(
        mx::complex64_t(NUM2DBL(rb_funcall(v, rb_intern("real"), 0)),
                      NUM2DBL(rb_funcall(v, rb_intern("imag"), 0))),
        mx::complex64);
  } else if (RB_TYPE_P(v, T_ARRAY)) {
    // Handle Ruby array conversion
    long length = RARRAY_LEN(v);
    
    if (length == 0) {
      // Create an empty array with the appropriate dtype
      auto dt = dtype.value_or(mx::float32);
      if (dt == mx::float32) {
        return mx::array(0.0f);
      } else if (dt == mx::float64) {
        return mx::array(0.0);
      } else if (dt == mx::int32) {
        return mx::array(0);
      } else if (dt == mx::int64) {
        return mx::array(0L);
      } else if (dt == mx::bool_) {
        return mx::array(false);
      } else {
        // Default to float32 and convert
        auto arr = mx::array(0.0f);
        return mx::astype(arr, dt);
      }
    }
    
    // Handle simple 1D numeric arrays more efficiently
    bool all_numeric = true;
    bool all_integer = true;
    bool all_float = true;
    
    for (long i = 0; i < length; i++) {
      VALUE item = rb_ary_entry(v, i);
      if (!RB_TYPE_P(item, T_FIXNUM) && !RB_TYPE_P(item, T_FLOAT)) {
        all_numeric = false;
        break;
      }
      if (RB_TYPE_P(item, T_FLOAT)) {
        all_integer = false;
      }
      if (RB_TYPE_P(item, T_FIXNUM)) {
        all_float = false;
      }
    }
    
    if (all_numeric) {
      if (all_integer) {
        // Convert int array element by element
        std::vector<int32_t> data;
        for (long i = 0; i < length; i++) {
          data.push_back(NUM2INT(rb_ary_entry(v, i)));
        }
        
        // Create the shape for a 1D array
        std::vector<int> shape = {static_cast<int>(length)};
        
        // Create the array with the specified shape and dtype
        mx::Dtype output_dtype = dtype.value_or(mx::int32);
        
        // Create a new array with the proper shape and type
        if (output_dtype == mx::int32) {
          return mx::array(data.data(), shape, mx::int32);
        } else {
          // If output dtype is different, create with int32 and then convert
          auto temp = mx::array(data.data(), shape, mx::int32);
          return mx::astype(temp, output_dtype);
        }
      } else if (all_float) {
        // Convert float array element by element
        std::vector<float> data;
        for (long i = 0; i < length; i++) {
          data.push_back(NUM2DBL(rb_ary_entry(v, i)));
        }
        
        // Create the shape for a 1D array
        std::vector<int> shape = {static_cast<int>(length)};
        
        // Create the array with the specified shape and dtype
        mx::Dtype output_dtype = dtype.value_or(mx::float32);
        
        // Create a new array with the proper shape and type
        if (output_dtype == mx::float32) {
          return mx::array(data.data(), shape, mx::float32);
        } else {
          // If output dtype is different, create with float32 and then convert
          auto temp = mx::array(data.data(), shape, mx::float32);
          return mx::astype(temp, output_dtype);
        }
      } else {
        // Mixed integers and floats - convert all to float
        std::vector<float> data;
        for (long i = 0; i < length; i++) {
          data.push_back(NUM2DBL(rb_ary_entry(v, i)));
        }
        
        // Create the shape for a 1D array
        std::vector<int> shape = {static_cast<int>(length)};
        
        // Create the array with the specified shape and dtype
        mx::Dtype output_dtype = dtype.value_or(mx::float32);
        
        // Create a new array with the proper shape and type
        if (output_dtype == mx::float32) {
          return mx::array(data.data(), shape, mx::float32);
        } else {
          // If output dtype is different, create with float32 and then convert
          auto temp = mx::array(data.data(), shape, mx::float32);
          return mx::astype(temp, output_dtype);
        }
      }
    }
    
    // Fallback - probably nested arrays
    rb_raise(rb_eNotImpError, "Nested array conversion not yet implemented");
  }
  
  rb_raise(rb_eTypeError, "Cannot convert to MLX array: unexpected type");

  // Instead of raising, we could do the following to call the accessor fallback:
  // return to_array_with_accessor(v, dtype);
  // but we'll just raise for clarity. If you prefer, do that fallback approach:
  // return to_array_with_accessor(v, dtype);
  return mx::array(0.0f); // To satisfy compiler
}

// Convert two Ruby values to MLX arrays with compatible types
std::pair<mx::array, mx::array> to_arrays(VALUE a, VALUE b) {
  auto is_mlx_array = [](VALUE x) {
    return rb_obj_is_kind_of(x, rb_path2class("MLX::Core::Array"));
  };
  
  // In Python, we also accept "arraylike" (has __mlx_array__). So do that here:
  if (is_mlx_array_or_arraylike(a)) {
    auto arr_a = get_array(a);
    if (is_mlx_array_or_arraylike(b)) {
      auto arr_b = get_array(b);
      return {arr_a, arr_b};
    }
    return {arr_a, to_array(b, arr_a.dtype())};
  } else if (is_mlx_array_or_arraylike(b)) {
    auto arr_b = get_array(b);
    return {to_array(a, arr_b.dtype()), arr_b};
  } else {
    // Neither 'a' nor 'b' is an MLX array or array-like, so we treat them both
    // as scalars or raw Ruby types.
    // This matches the python logic:
    // return {to_array(a), to_array(b)};

    return {to_array(a), to_array(b)};
  }
}

// Ruby wrapper functions

// Create a StreamContext from Ruby
VALUE rb_create_stream_context(int argc, VALUE* argv, VALUE self) {
  if (argc > 1) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 0..1)", argc);
  }
  
  StreamOrDevice stream_or_device = std::monostate{};
  if (argc == 1) {
    stream_or_device = get_stream_or_device(argv[0]);
  }
  
  // Create a new StreamContext with the passed StreamOrDevice
  StreamContext* context = new StreamContext(stream_or_device);
  
  return Data_Wrap_Struct(rb_path2class("MLX::Core::StreamContext"), 0, 
                          [](void* ptr){ delete static_cast<StreamContext*>(ptr); }, 
                          context);
}

// Tree manipulation (custom implementation for Ruby)
static VALUE utils_tree_flatten(VALUE self, VALUE arr) {
  if (!rb_obj_is_kind_of(arr, rb_path2class("MLX::Core::Array"))) {
    rb_raise(rb_eTypeError, "Expected an MLX::Core::Array");
  }
  
  // Just return the array itself since we don't have proper tree_flatten
  VALUE result = rb_ary_new();
  rb_ary_push(result, arr);
  
  return result;
}

static VALUE utils_is_array_like(VALUE self, VALUE obj) {
  bool is_like = false;
  
  if (rb_obj_is_kind_of(obj, rb_path2class("MLX::Core::Array"))) {
    is_like = true;
  }
  
  return is_like ? Qtrue : Qfalse;
}

static VALUE utils_is_pytree_leaf(VALUE self, VALUE obj) {
  return utils_is_array_like(self, obj);
}

static VALUE utils_dtype_to_string(VALUE self, VALUE dtype) {
  int dtype_val = NUM2INT(dtype);
  
  std::string dtype_str;
  // Handle each dtype value
  if (dtype_val == 0) {
    dtype_str = "bool";
  } else if (dtype_val == 1) {
    dtype_str = "uint8";
  } else if (dtype_val == 2) {
    dtype_str = "uint16";
  } else if (dtype_val == 3) {
    dtype_str = "uint32";
  } else if (dtype_val == 4) {
    dtype_str = "uint64";
  } else if (dtype_val == 5) {
    dtype_str = "int8";
  } else if (dtype_val == 6) {
    dtype_str = "int16";
  } else if (dtype_val == 7) {
    dtype_str = "int32";
  } else if (dtype_val == 8) {
    dtype_str = "int64";
  } else if (dtype_val == 9) {
    dtype_str = "float16";
  } else if (dtype_val == 10) {
    dtype_str = "float32";
  } else if (dtype_val == 11) {
    dtype_str = "float64";
  } else if (dtype_val == 12) {
    dtype_str = "bfloat16";
  } else if (dtype_val == 13) {
    dtype_str = "complex64";
  } else {
    dtype_str = "unknown";
  }
  
  return rb_str_new_cstr(dtype_str.c_str());
}

static VALUE utils_size_to_string(int argc, VALUE* argv, VALUE self) {
  if (argc != 1) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1)", argc);
  }
  
  VALUE size = argv[0];
  Check_Type(size, T_FIXNUM);
  
  long size_val = NUM2LONG(size);
  std::ostringstream ss;
  
  if (size_val < 1024) {
    ss << size_val << " B";
  } else if (size_val < 1024 * 1024) {
    ss << (size_val / 1024.0) << " KB";
  } else if (size_val < 1024 * 1024 * 1024) {
    ss << (size_val / (1024.0 * 1024.0)) << " MB";
  } else {
    ss << (size_val / (1024.0 * 1024.0 * 1024.0)) << " GB";
  }
  
  return rb_str_new_cstr(ss.str().c_str());
}

static VALUE utils_eval_counter(VALUE self) {
  // Placeholder since we don't have direct access to mx::eval_count()
  return LONG2NUM(0);
}

// Helper function to create correct Dtype from integer value
static mx::Dtype int_to_dtype(int val) {
  switch (val) {
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
    default: return mx::float32; // Default
  }
}

static VALUE utils_issubdtype(VALUE self, VALUE dtype1, VALUE dtype2) {
  int d1_val = NUM2INT(dtype1);
  int d2_val = NUM2INT(dtype2);
  
  mx::Dtype dt1 = int_to_dtype(d1_val);
  mx::Dtype dt2 = int_to_dtype(d2_val);
  
  bool result = mx::issubdtype(dt1, dt2);
  return result ? Qtrue : Qfalse;
}

static VALUE utils_promote_types(VALUE self, VALUE dtype1, VALUE dtype2) {
  int d1_val = NUM2INT(dtype1);
  int d2_val = NUM2INT(dtype2);
  
  mx::Dtype dt1 = int_to_dtype(d1_val);
  mx::Dtype dt2 = int_to_dtype(d2_val);
  
  mx::Dtype result = mx::promote_types(dt1, dt2);
  
  // Create an integer that represents the dtype
  for (int i = 0; i <= 13; i++) {
    if (result == int_to_dtype(i)) {
      return INT2NUM(i);
    }
  }
  
  // Default to float32 (10)
  return INT2NUM(10);
}

// Initialize utils module
void init_utils(VALUE module) {
  // Get the MLX::Array class
  rb_cMLXArray = rb_const_get(rb_cObject, rb_intern("MLX::Array"));
  
  // Define module functions
  rb_define_module_function(module, "create_stream_context", RUBY_METHOD_FUNC(rb_create_stream_context), -1);
  rb_define_module_function(module, "tree_flatten", RUBY_METHOD_FUNC(utils_tree_flatten), 1);
  rb_define_module_function(module, "is_array_like", RUBY_METHOD_FUNC(utils_is_array_like), 1);
  rb_define_module_function(module, "is_pytree_leaf", RUBY_METHOD_FUNC(utils_is_pytree_leaf), 1);
  rb_define_module_function(module, "dtype_to_string", RUBY_METHOD_FUNC(utils_dtype_to_string), 1);
  rb_define_module_function(module, "size_to_string", RUBY_METHOD_FUNC(utils_size_to_string), -1);
  rb_define_module_function(module, "eval_counter", RUBY_METHOD_FUNC(utils_eval_counter), 0);
  rb_define_module_function(module, "issubdtype", RUBY_METHOD_FUNC(utils_issubdtype), 2);
  rb_define_module_function(module, "promote_types", RUBY_METHOD_FUNC(utils_promote_types), 2);
}