#include <ruby.h>
#include <vector>
#include <complex>
#include "mlx/ops.h"
#include <stdexcept>
#include <sstream>
#include "convert.h"

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

// Template specializations for converting Ruby types to C++ scalars
template<>
bool ruby_to_scalar<bool>(VALUE obj) {
  return RTEST(obj);
}

template<>
uint8_t ruby_to_scalar<uint8_t>(VALUE obj) {
  return NUM2UINT(obj);
}

template<>
uint16_t ruby_to_scalar<uint16_t>(VALUE obj) {
  return NUM2UINT(obj);
}

template<>
uint32_t ruby_to_scalar<uint32_t>(VALUE obj) {
  return NUM2UINT(obj);
}

template<>
uint64_t ruby_to_scalar<uint64_t>(VALUE obj) {
  return NUM2ULL(obj);
}

template<>
int8_t ruby_to_scalar<int8_t>(VALUE obj) {
  return NUM2INT(obj);
}

template<>
int16_t ruby_to_scalar<int16_t>(VALUE obj) {
  return NUM2INT(obj);
}

template<>
int32_t ruby_to_scalar<int32_t>(VALUE obj) {
  return NUM2INT(obj);
}

template<>
int64_t ruby_to_scalar<int64_t>(VALUE obj) {
  return NUM2LL(obj);
}

template<>
float ruby_to_scalar<float>(VALUE obj) {
  return (float)NUM2DBL(obj);
}

template<>
double ruby_to_scalar<double>(VALUE obj) {
  return NUM2DBL(obj);
}

template<>
std::complex<float> ruby_to_scalar<std::complex<float>>(VALUE obj) {
  if (rb_respond_to(obj, rb_intern("real")) && rb_respond_to(obj, rb_intern("imag"))) {
    float real = (float)NUM2DBL(rb_funcall(obj, rb_intern("real"), 0));
    float imag = (float)NUM2DBL(rb_funcall(obj, rb_intern("imag"), 0));
    return std::complex<float>(real, imag);
  } else {
    return std::complex<float>((float)NUM2DBL(obj), 0.0f);
  }
}

// Convert Ruby array to std::vector
template<typename T>
std::vector<T> ruby_array_to_vector(VALUE rb_array) {
  Check_Type(rb_array, T_ARRAY);
  
  std::vector<T> result;
  long size = RARRAY_LEN(rb_array);
  result.reserve(size);
  
  for (long i = 0; i < size; i++) {
    VALUE item = rb_ary_entry(rb_array, i);
    result.push_back(ruby_to_scalar<T>(item));
  }
  
  return result;
}

// Function to convert Ruby objects to MLX arrays
mx::array ruby_to_array(VALUE obj) {
  if (rb_obj_is_kind_of(obj, rb_path2class("MLX::Core::Array"))) {
    // Object is already an MLX array
    return get_array(obj);
  } else if (RB_TYPE_P(obj, T_ARRAY)) {
    // Convert Ruby array to MLX array
    std::vector<double> data;
    for (long i = 0; i < RARRAY_LEN(obj); i++) {
      VALUE item = rb_ary_entry(obj, i);
      data.push_back(NUM2DBL(item));
    }
    
    // Create a 1D array with the appropriate shape instead of passing vector directly
    mx::Shape shape = {static_cast<int>(data.size())};
    return mx::array(data.data(), shape, mx::float32);
  } else if (rb_obj_is_kind_of(obj, rb_cNumeric)) {
    // Convert Ruby numeric to MLX scalar array
    return mx::array(NUM2DBL(obj));
  } else {
    rb_raise(rb_eTypeError, "Cannot convert Ruby object to MLX array");
    return mx::array({}, mx::float32); // Return empty array with float32 type
  }
}

// ---------------------------------------------------------------------------
// 1) Convert an mx::array of size=1 to a Ruby scalar
//    (equivalent to Python's to_scalar(mx::array&))
// ---------------------------------------------------------------------------
static VALUE
convert_to_scalar(VALUE self, VALUE rb_arr)
{
  mx::array& a = get_array(rb_arr);
  if (a.size() != 1) {
    rb_raise(rb_eArgError, "[convert_to_scalar] Only length-1 arrays can be converted to Ruby scalars.");
    return Qnil; // never reached
  }

  // Force evaluation if it's a lazy array
  a.eval();

  switch (a.dtype()) {
    case mx::bool_:
      return a.item<bool>() ? Qtrue : Qfalse;
    case mx::uint8:
      return UINT2NUM(a.item<uint8_t>());
    case mx::uint16:
      return UINT2NUM(a.item<uint16_t>());
    case mx::uint32:
      return UINT2NUM(a.item<uint32_t>());
    case mx::uint64:
      return ULL2NUM(a.item<uint64_t>());
    case mx::int8:
      return INT2NUM(a.item<int8_t>());
    case mx::int16:
      return INT2NUM(a.item<int16_t>());
    case mx::int32:
      return INT2NUM(a.item<int32_t>());
    case mx::int64:
      return LL2NUM(a.item<int64_t>());
    case mx::float16:
      // cast float16 -> float -> Ruby double
      return rb_float_new((double) static_cast<float>(a.item<mx::float16_t>()));
    case mx::bfloat16:
      return rb_float_new((double) static_cast<float>(a.item<mx::bfloat16_t>()));
    case mx::float32:
      return rb_float_new((double) a.item<float>());
    case mx::float64:
      return rb_float_new(a.item<double>());
    case mx::complex64: {
      std::complex<float> cval = a.item<std::complex<float>>();
      // Return a Ruby Complex
      VALUE realPart = rb_float_new((double) cval.real());
      VALUE imagPart = rb_float_new((double) cval.imag());
      return rb_funcall(rb_cComplex, rb_intern("rect"), 2, realPart, imagPart);
    }
    default:
      rb_raise(rb_eTypeError, "Cannot convert array of this dtype to a Ruby scalar.");
      return Qnil; // never reached
  }
}

// ---------------------------------------------------------------------------
// 2) Convert mx::array -> nested Ruby array
//    (equivalent to Python's tolist(mx::array&))
// ---------------------------------------------------------------------------

static VALUE array_to_ruby_recursive(mx::array &a, size_t offset, int dim)
{
  // If we are at the last dimension, build a flat Ruby array
  int sizeHere = a.shape(dim);
  VALUE rb_out = rb_ary_new2(sizeHere);
  auto stride = a.strides()[dim];

  if (dim == a.ndim() - 1) {
    // Last dimension -> pack scalars
    for (int i = 0; i < sizeHere; i++) {
      // Reuse convert_to_scalar logic for single items if you prefer,
      // but simpler is to do a switch on dtype:
      switch (a.dtype()) {
        case mx::bool_:
          rb_ary_push(rb_out, a.data<bool>()[offset + i] ? Qtrue : Qfalse);
          break;
        case mx::uint8:
          rb_ary_push(rb_out, UINT2NUM(a.data<uint8_t>()[offset + i]));
          break;
        case mx::uint16:
          rb_ary_push(rb_out, UINT2NUM(a.data<uint16_t>()[offset + i]));
          break;
        case mx::uint32:
          rb_ary_push(rb_out, UINT2NUM(a.data<uint32_t>()[offset + i]));
          break;
        case mx::uint64:
          rb_ary_push(rb_out, ULL2NUM(a.data<uint64_t>()[offset + i]));
          break;
        case mx::int8:
          rb_ary_push(rb_out, INT2NUM(a.data<int8_t>()[offset + i]));
          break;
        case mx::int16:
          rb_ary_push(rb_out, INT2NUM(a.data<int16_t>()[offset + i]));
          break;
        case mx::int32:
          rb_ary_push(rb_out, INT2NUM(a.data<int32_t>()[offset + i]));
          break;
        case mx::int64:
          rb_ary_push(rb_out, LL2NUM(a.data<int64_t>()[offset + i]));
          break;
        case mx::float16: {
          float val = (float) a.data<mx::float16_t>()[offset + i];
          rb_ary_push(rb_out, rb_float_new((double)val));
          break;
        }
        case mx::bfloat16: {
          float val = (float) a.data<mx::bfloat16_t>()[offset + i];
          rb_ary_push(rb_out, rb_float_new((double)val));
          break;
        }
        case mx::float32:
          rb_ary_push(rb_out, rb_float_new((double) a.data<float>()[offset + i]));
          break;
        case mx::float64:
          rb_ary_push(rb_out, rb_float_new(a.data<double>()[offset + i]));
          break;
        case mx::complex64: {
          auto c = a.data<std::complex<float>>()[offset + i];
          VALUE realPart = rb_float_new((double) c.real());
          VALUE imagPart = rb_float_new((double) c.imag());
          VALUE cplx = rb_funcall(rb_cComplex, rb_intern("rect"), 2, realPart, imagPart);
          rb_ary_push(rb_out, cplx);
          break;
        }
        default:
          rb_raise(rb_eTypeError, "Unsupported dtype in to_list conversion.");
      }
    }
  } else {
    // Recursively build subarrays
    size_t current = offset;
    for (int i = 0; i < sizeHere; i++) {
      VALUE sub = array_to_ruby_recursive(a, current, dim + 1);
      rb_ary_push(rb_out, sub);
      current += stride;
    }
  }
  return rb_out;
}

static VALUE
convert_to_list(VALUE self, VALUE rb_arr)
{
  mx::array& a = get_array(rb_arr);
  a.eval();
  if (a.ndim() == 0) {
    // 0D -> scalar
    return convert_to_scalar(self, rb_arr);
  }
  return array_to_ruby_recursive(a, 0, 0);
}

// ---------------------------------------------------------------------------
// 3) Upgrade ruby_to_array to handle nested Ruby arrays (like Python's array_from_list).
//    We do shape inference and create an mx::array of an inferred or default dtype.
// ---------------------------------------------------------------------------

static void infer_shape(VALUE rbObj, std::vector<int> &shp)
{
  // Expect an Array; push dimension length
  long len = RARRAY_LEN(rbObj);
  shp.push_back((int)len);
  if (len == 0) return; // shape is [0], done

  // Check first element to see if we nest further
  VALUE first = rb_ary_entry(rbObj, 0);
  if (RB_TYPE_P(first, T_ARRAY)) {
    // Nested array dimension
    infer_shape(first, shp);
  }
  // else if you wanted to handle e.g. MLX::Core::Array or scalars, that is more advanced logic
}

static void flatten_ruby_array(VALUE rbObj, std::vector<double> &accum)
{
  Check_Type(rbObj, T_ARRAY);
  long len = RARRAY_LEN(rbObj);
  for (long i = 0; i < len; i++) {
    VALUE item = rb_ary_entry(rbObj, i);
    if (RB_TYPE_P(item, T_ARRAY)) {
      // recurse
      flatten_ruby_array(item, accum);
    } else {
      accum.push_back(NUM2DBL(item));
    }
  }
}

static mx::array
ruby_nested_array_to_mx(VALUE rbObj)
{
  // shape inference
  std::vector<int> shape;
  infer_shape(rbObj, shape);
  // flatten data as double
  std::vector<double> data;
  flatten_ruby_array(rbObj, data);
  // create the array (default float32)
  mx::Shape mxshape(shape.begin(), shape.end());
  return mx::astype(mx::array((const double*)data.data(), mxshape, mx::float64), mx::float32);
}

// Convert module methods
static VALUE convert_to_float16(VALUE self, VALUE arr) {
  mx::array& a = get_array(arr);
  mx::array result = mx::astype(a, mx::float16);
  return wrap_array(result);
}

static VALUE convert_to_float32(VALUE self, VALUE arr) {
  mx::array& a = get_array(arr);
  mx::array result = mx::astype(a, mx::float32);
  return wrap_array(result);
}

static VALUE convert_to_int32(VALUE self, VALUE arr) {
  mx::array& a = get_array(arr);
  mx::array result = mx::astype(a, mx::int32);
  return wrap_array(result);
}

static VALUE convert_to_bool(VALUE self, VALUE arr) {
  mx::array& a = get_array(arr);
  mx::array result = mx::astype(a, mx::bool_);
  return wrap_array(result);
}

static VALUE convert_to_type(VALUE self, VALUE arr, VALUE dtype) {
  mx::array& a = get_array(arr);
  // Use the predefined Dtype constants instead of trying to create a default-constructed Dtype
  mx::Dtype d = mx::float32; // Default to float32
  int dtype_val = NUM2INT(dtype);
  switch (dtype_val) {
    case 0: d = mx::bool_; break;
    case 1: d = mx::uint8; break;
    case 2: d = mx::uint16; break;
    case 3: d = mx::uint32; break;
    case 4: d = mx::uint64; break;
    case 5: d = mx::int8; break;
    case 6: d = mx::int16; break;
    case 7: d = mx::int32; break;
    case 8: d = mx::int64; break;
    case 9: d = mx::float16; break;
    case 10: d = mx::float32; break;
    case 11: d = mx::float64; break;
    case 12: d = mx::bfloat16; break;
    case 13: d = mx::complex64; break;
    default: break; // Already defaulted to float32
  }
  mx::array result = mx::astype(a, d);
  return wrap_array(result);
}

// Initialize convert module
void init_convert(VALUE module) {
  // Define module functions
  rb_define_module_function(module, "to_float16", RUBY_METHOD_FUNC(convert_to_float16), 1);
  rb_define_module_function(module, "to_float32", RUBY_METHOD_FUNC(convert_to_float32), 1);
  rb_define_module_function(module, "to_int32", RUBY_METHOD_FUNC(convert_to_int32), 1);
  rb_define_module_function(module, "to_bool", RUBY_METHOD_FUNC(convert_to_bool), 1);
  rb_define_module_function(module, "to_type", RUBY_METHOD_FUNC(convert_to_type), 2);
  rb_define_module_function(module, "to_scalar", RUBY_METHOD_FUNC(convert_to_scalar), 1);
  rb_define_module_function(module, "to_list", RUBY_METHOD_FUNC(convert_to_list), 1);

//   // You might want a "ruby_to_array" entry point that handles nested arrays:
//   //   MLX::Convert.ruby_obj_to_mx(obj) -> MLX::Core::Array
//   //
//   // For example:
//   rb_define_module_function(module, "ruby_obj_to_mx", RUBY_METHOD_FUNC(
//     [](VALUE self, VALUE obj) -> VALUE {
//       // This can now handle nested arrays or existing MLX arrays or scalars:
//       if (rb_obj_is_kind_of(obj, rb_path2class("MLX::Core::Array"))) {
//         mx::array& a = get_array(obj);
//         return wrap_array(a);
//       } else if (RB_TYPE_P(obj, T_ARRAY)) {
//         mx::array arr = ruby_nested_array_to_mx(obj);
//         return wrap_array(arr);
//       } else if (rb_obj_is_kind_of(obj, rb_cNumeric)) {
//         return wrap_array(mx::array(NUM2DBL(obj)));
//       }
//       rb_raise(rb_eTypeError, "Cannot convert to MLX::Core::Array");
//       return Qnil;
//     }
//   ), 1);
// } 