#include <ruby.h>
#include <algorithm>
#include <complex>
#include <iostream>
#include <sstream>
#include "mlx/ops.h"
#include "mlx/array.h"
#include "utils.h"

namespace mx = mlx::core;

// Helper struct for scalar or array inputs
struct ScalarOrArray {
  std::variant<VALUE, mx::array> value;
  
  ScalarOrArray(VALUE v) : value(v) {}
  ScalarOrArray(mx::array a) : value(std::move(a)) {}
};

// Convert Ruby value to MLX array with optional dtype
mx::array to_array(const ScalarOrArray& v, std::optional<mx::Dtype> dtype = std::nullopt) {
  if (std::holds_alternative<mx::array>(v.value)) {
    auto& arr = std::get<mx::array>(v.value);
    if (dtype.has_value() && arr.dtype() != dtype.value()) {
      return mx::astype(arr, *dtype);
    }
    return arr;
  }
  
  VALUE ruby_val = std::get<VALUE>(v.value);
  
  if (NIL_P(ruby_val)) {
    rb_raise(rb_eTypeError, "Cannot convert nil to MLX array");
  }
  
  if (RB_TYPE_P(ruby_val, T_FLOAT) || RB_TYPE_P(ruby_val, T_FIXNUM)) {
    double val = NUM2DBL(ruby_val);
    return mx::array(val, dtype.value_or(mx::float32));
  }
  
  if (RB_TYPE_P(ruby_val, T_TRUE) || RB_TYPE_P(ruby_val, T_FALSE)) {
    bool val = RTEST(ruby_val);
    return mx::array(val, dtype.value_or(mx::bool_));
  }
  
  if (RB_TYPE_P(ruby_val, T_ARRAY)) {
    long len = RARRAY_LEN(ruby_val);
    
    // For simplicity, first check if all elements are numeric
    for (long i = 0; i < len; i++) {
      VALUE item = rb_ary_entry(ruby_val, i);
      if (!RB_TYPE_P(item, T_FLOAT) && !RB_TYPE_P(item, T_FIXNUM)) {
        rb_raise(rb_eTypeError, "Array elements must be numeric");
      }
    }
    
    // Create 1D array with the given dtype or default
    mx::Dtype target_dtype = dtype.value_or(mx::float32);
    
    // Create array with shape [len]
    mx::array result({len}, target_dtype);
    
    // Fill array element by element
    for (long i = 0; i < len; i++) {
      VALUE item = rb_ary_entry(ruby_val, i);
      double value = NUM2DBL(item);
      
      // Set array element at index i
      mx::array idx = mx::array(static_cast<int>(i));
      mx::array val = mx::array(value, target_dtype);
      result = mx::scatter(result, {idx}, val, 0);
    }
    
    return result;
  }
  
  // Check if it's an MLX array
  if (rb_obj_is_kind_of(ruby_val, rb_path2class("MLX::Array"))) {
    mx::array* ptr;
    Data_Get_Struct(ruby_val, mx::array, ptr);
    if (dtype.has_value() && ptr->dtype() != dtype.value()) {
      return mx::astype(*ptr, *dtype);
    }
    return *ptr;
  }
  
  rb_raise(rb_eTypeError, "Cannot convert Ruby object to MLX array");
  return mx::array({}, mx::float32); // Return empty float32 array instead of mx::array()
}

// Helper to check if value can be used with array operations
bool is_comparable_with_array(const ScalarOrArray& v) {
  if (std::holds_alternative<mx::array>(v.value)) {
    return true;
  }
  
  VALUE ruby_val = std::get<VALUE>(v.value);
  
  return RB_TYPE_P(ruby_val, T_FLOAT) || 
         RB_TYPE_P(ruby_val, T_FIXNUM) || 
         RB_TYPE_P(ruby_val, T_TRUE) || 
         RB_TYPE_P(ruby_val, T_FALSE) ||
         RB_TYPE_P(ruby_val, T_ARRAY) ||
         rb_obj_is_kind_of(ruby_val, rb_path2class("MLX::Array"));
}

// Helper for invalid operation error
void throw_invalid_operation(const char* op_name, const ScalarOrArray& v) {
  rb_raise(rb_eTypeError, "Invalid %s operation with non-numeric type", op_name);
}

// Helper to convert MLX array to Ruby scalar
VALUE to_scalar(const mx::array& arr) {
  if (arr.size() != 1) {
    rb_raise(rb_eRuntimeError, "Can only convert size 1 arrays to scalar");
  }
  
  auto dtype = arr.dtype();
  
  if (dtype == mx::bool_) {
    return arr.item<bool>() ? Qtrue : Qfalse;
  } else if (mx::issubdtype(dtype, mx::integer)) {
    return INT2NUM(arr.item<int>());
  } else if (mx::issubdtype(dtype, mx::floating)) {
    return DBL2NUM(arr.item<double>());
  }
  
  rb_raise(rb_eTypeError, "Unsupported dtype for scalar conversion");
  return Qnil; // Never reached
}

// Helper to convert MLX array to Ruby array
VALUE tolist(const mx::array& arr) {
  if (arr.size() == 1) {
    return to_scalar(arr);
  }
  
  VALUE result = rb_ary_new2(arr.shape(0));
  
  if (arr.ndim() == 1) {
    for (int i = 0; i < arr.shape(0); i++) {
      auto idx = mx::array(i);
      auto val = mx::take(arr, idx, 0);
      rb_ary_push(result, to_scalar(val));
    }
  } else {
    for (int i = 0; i < arr.shape(0); i++) {
      auto idx = mx::array(i);
      auto slice = mx::take(arr, idx, 0);
      auto squeezed = mx::squeeze(slice, 0);
      rb_ary_push(result, tolist(squeezed));
    }
  }
  
  return result;
}

// Helper class for array iteration
class ArrayRubyIterator {
public:
  ArrayRubyIterator(mx::array x) : idx_(0), x_(std::move(x)) {
    if (x_.shape(0) > 0 && x_.shape(0) < 10) {
      splits_ = mx::split(x_, x_.shape(0));
    }
  }

  mx::array next() {
    if (idx_ >= x_.shape(0)) {
      return mx::array({}, mx::float32); // Return empty float32 array instead of mx::array()
    }

    if (idx_ >= 0 && idx_ < splits_.size()) {
      return mx::squeeze(splits_[idx_++], 0);
    }

    return *(x_.begin() + idx_++);
  }

  bool done() {
    return idx_ >= x_.shape(0);
  }

private:
  int idx_;
  mx::array x_;
  std::vector<mx::array> splits_;
};

// Helper class for array indexing operations
class ArrayAt {
public:
  ArrayAt(mx::array x) : x_(std::move(x)) {}
  
  ArrayAt& set_indices(VALUE indices) {
    indices_ = indices;
    return *this;
  }
  
  mx::array add(const ScalarOrArray& v) {
    auto val = to_array(v, x_.dtype());
    
    // Convert Ruby indices to array indices
    if (RB_TYPE_P(indices_, T_FIXNUM)) {
      // Single integer index
      int idx = NUM2INT(indices_);
      mx::array index = mx::array(idx);
      return mx::scatter_add(x_, {index}, val, 0);
    } else if (RB_TYPE_P(indices_, T_ARRAY)) {
      // Array of indices
      long len = RARRAY_LEN(indices_);
      // Create 1D array for indices
      mx::array index({len}, mx::int32);
      
      // Fill array element by element
      for (long i = 0; i < len; i++) {
        VALUE item = rb_ary_entry(indices_, i);
        int value = NUM2INT(item);
        
        // Set index element
        mx::array idx = mx::array(static_cast<int>(i));
        mx::array val = mx::array(value);
        index = mx::scatter(index, {idx}, val, 0);
      }
      return mx::scatter_add(x_, {index}, val, 0);
    } else if (rb_obj_is_kind_of(indices_, rb_path2class("MLX::Array"))) {
      // MLX array as index
      mx::array index = to_array(ScalarOrArray(indices_));
      return mx::scatter_add(x_, {index}, val, 0);
    }
    
    rb_raise(rb_eTypeError, "Invalid index type for at operation");
    return x_; // Unreachable
  }
  
  mx::array subtract(const ScalarOrArray& v) {
    auto val = to_array(v, x_.dtype());
    
    // Convert Ruby indices to array indices
    if (RB_TYPE_P(indices_, T_FIXNUM)) {
      // Single integer index
      int idx = NUM2INT(indices_);
      mx::array index = mx::array(idx);
      return mx::scatter_add(x_, {index}, -val, 0);
    } else if (RB_TYPE_P(indices_, T_ARRAY)) {
      // Array of indices
      long len = RARRAY_LEN(indices_);
      // Create 1D array for indices
      mx::array index({len}, mx::int32);
      
      // Fill array element by element
      for (long i = 0; i < len; i++) {
        VALUE item = rb_ary_entry(indices_, i);
        int value = NUM2INT(item);
        
        // Set index element
        mx::array idx = mx::array(static_cast<int>(i));
        mx::array val = mx::array(value);
        index = mx::scatter(index, {idx}, val, 0);
      }
      return mx::scatter_add(x_, {index}, -val, 0);
    } else if (rb_obj_is_kind_of(indices_, rb_path2class("MLX::Array"))) {
      // MLX array as index
      mx::array index = to_array(ScalarOrArray(indices_));
      return mx::scatter_add(x_, {index}, -val, 0);
    }
    
    rb_raise(rb_eTypeError, "Invalid index type for at operation");
    return x_; // Unreachable
  }
  
  mx::array multiply(const ScalarOrArray& v) {
    auto val = to_array(v, x_.dtype());
    
    // Convert Ruby indices to array indices
    if (RB_TYPE_P(indices_, T_FIXNUM)) {
      // Single integer index
      int idx = NUM2INT(indices_);
      mx::array index = mx::array(idx);
      return mx::scatter_prod(x_, {index}, val, 0);
    } else if (RB_TYPE_P(indices_, T_ARRAY)) {
      // Array of indices
      long len = RARRAY_LEN(indices_);
      // Create 1D array for indices
      mx::array index({len}, mx::int32);
      
      // Fill array element by element
      for (long i = 0; i < len; i++) {
        VALUE item = rb_ary_entry(indices_, i);
        int value = NUM2INT(item);
        
        // Set index element
        mx::array idx = mx::array(static_cast<int>(i));
        mx::array val = mx::array(value);
        index = mx::scatter(index, {idx}, val, 0);
      }
      return mx::scatter_prod(x_, {index}, val, 0);
    } else if (rb_obj_is_kind_of(indices_, rb_path2class("MLX::Array"))) {
      // MLX array as index
      mx::array index = to_array(ScalarOrArray(indices_));
      return mx::scatter_prod(x_, {index}, val, 0);
    }
    
    rb_raise(rb_eTypeError, "Invalid index type for at operation");
    return x_; // Unreachable
  }
  
  mx::array divide(const ScalarOrArray& v) {
    auto val = to_array(v, x_.dtype());
    
    // Convert Ruby indices to array indices
    if (RB_TYPE_P(indices_, T_FIXNUM)) {
      // Single integer index
      int idx = NUM2INT(indices_);
      mx::array index = mx::array(idx);
      return mx::scatter_prod(x_, {index}, mx::reciprocal(val), 0);
    } else if (RB_TYPE_P(indices_, T_ARRAY)) {
      // Array of indices
      long len = RARRAY_LEN(indices_);
      // Create 1D array for indices
      mx::array index({len}, mx::int32);
      
      // Fill array element by element
      for (long i = 0; i < len; i++) {
        VALUE item = rb_ary_entry(indices_, i);
        int value = NUM2INT(item);
        
        // Set index element
        mx::array idx = mx::array(static_cast<int>(i));
        mx::array val = mx::array(value);
        index = mx::scatter(index, {idx}, val, 0);
      }
      return mx::scatter_prod(x_, {index}, mx::reciprocal(val), 0);
    } else if (rb_obj_is_kind_of(indices_, rb_path2class("MLX::Array"))) {
      // MLX array as index
      mx::array index = to_array(ScalarOrArray(indices_));
      return mx::scatter_prod(x_, {index}, mx::reciprocal(val), 0);
    }
    
    rb_raise(rb_eTypeError, "Invalid index type for at operation");
    return x_; // Unreachable
  }
  
  mx::array maximum(const ScalarOrArray& v) {
    auto val = to_array(v, x_.dtype());
    
    // Convert Ruby indices to array indices
    if (RB_TYPE_P(indices_, T_FIXNUM)) {
      // Single integer index
      int idx = NUM2INT(indices_);
      mx::array index = mx::array(idx);
      return mx::scatter_max(x_, {index}, val, 0);
    } else if (RB_TYPE_P(indices_, T_ARRAY)) {
      // Array of indices
      long len = RARRAY_LEN(indices_);
      // Create 1D array for indices
      mx::array index({len}, mx::int32);
      
      // Fill array element by element
      for (long i = 0; i < len; i++) {
        VALUE item = rb_ary_entry(indices_, i);
        int value = NUM2INT(item);
        
        // Set index element
        mx::array idx = mx::array(static_cast<int>(i));
        mx::array val = mx::array(value);
        index = mx::scatter(index, {idx}, val, 0);
      }
      return mx::scatter_max(x_, {index}, val, 0);
    } else if (rb_obj_is_kind_of(indices_, rb_path2class("MLX::Array"))) {
      // MLX array as index
      mx::array index = to_array(ScalarOrArray(indices_));
      return mx::scatter_max(x_, {index}, val, 0);
    }
    
    rb_raise(rb_eTypeError, "Invalid index type for at operation");
    return x_; // Unreachable
  }
  
  mx::array minimum(const ScalarOrArray& v) {
    auto val = to_array(v, x_.dtype());
    
    // Convert Ruby indices to array indices
    if (RB_TYPE_P(indices_, T_FIXNUM)) {
      // Single integer index
      int idx = NUM2INT(indices_);
      mx::array index = mx::array(idx);
      return mx::scatter_min(x_, {index}, val, 0);
    } else if (RB_TYPE_P(indices_, T_ARRAY)) {
      // Array of indices
      long len = RARRAY_LEN(indices_);
      // Create 1D array for indices
      mx::array index({len}, mx::int32);
      
      // Fill array element by element
      for (long i = 0; i < len; i++) {
        VALUE item = rb_ary_entry(indices_, i);
        int value = NUM2INT(item);
        
        // Set index element
        mx::array idx = mx::array(static_cast<int>(i));
        mx::array val = mx::array(value);
        index = mx::scatter(index, {idx}, val, 0);
      }
      return mx::scatter_min(x_, {index}, val, 0);
    } else if (rb_obj_is_kind_of(indices_, rb_path2class("MLX::Array"))) {
      // MLX array as index
      mx::array index = to_array(ScalarOrArray(indices_));
      return mx::scatter_min(x_, {index}, val, 0);
    }
    
    rb_raise(rb_eTypeError, "Invalid index type for at operation");
    return x_; // Unreachable
  }

private:
  mx::array x_;
  VALUE indices_;
};

// Ruby GC free function for mx::array
void rb_free_array(void* ptr) {
  mx::array* arr = static_cast<mx::array*>(ptr);
  delete arr;
}

// Ruby object allocation function
VALUE rb_alloc_array(VALUE klass) {
  mx::array* ptr = new mx::array({}, mx::float32); // Create with empty vector and float32 dtype
  return Data_Wrap_Struct(klass, NULL, rb_free_array, ptr);
}

// Helper to extract mx::array from Ruby VALUE
mx::array& get_array(VALUE self) {
  mx::array* ptr;
  Data_Get_Struct(self, mx::array, ptr);
  return *ptr;
}

// Array methods for Ruby
VALUE rb_array_initialize(int argc, VALUE* argv, VALUE self) {
  // Enhanced initialization with shape and dtype
  if (argc == 0) {
    // Create empty array
    get_array(self) = mx::array({}, mx::float32);
  } else if (argc >= 1) {
    VALUE val = argv[0];
    
    // Get optional dtype
    mx::Dtype dtype = mx::float32;
    if (argc >= 2 && !NIL_P(argv[1])) {
      int dtype_val = NUM2INT(argv[1]);
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
        default: dtype = mx::float32;
      }
    }
    
    // Convert based on type
    if (RB_TYPE_P(val, T_ARRAY)) {
      // Create from Ruby array
      ScalarOrArray soa(val);
      get_array(self) = to_array(soa, dtype);
    } else if (RB_TYPE_P(val, T_FLOAT) || RB_TYPE_P(val, T_FIXNUM)) {
      // Create from scalar
      double scalar_val = NUM2DBL(val);
      get_array(self) = mx::array(scalar_val, dtype);
    } else if (RB_TYPE_P(val, T_TRUE) || RB_TYPE_P(val, T_FALSE)) {
      // Create from boolean
      bool bool_val = RTEST(val);
      get_array(self) = mx::array(bool_val, dtype);
    } else if (rb_obj_is_kind_of(val, rb_path2class("MLX::Array"))) {
      // Copy from another array
      mx::array& other_arr = get_array(val);
      get_array(self) = mx::array(other_arr);
      if (other_arr.dtype() != dtype) {
        get_array(self) = mx::astype(get_array(self), dtype);
      }
    } else {
      rb_raise(rb_eTypeError, "Invalid input type for array initialization");
    }
  }
  
  return self;
}

// Basic array operations
VALUE rb_array_shape(VALUE self) {
  mx::array& arr = get_array(self);
  VALUE result = rb_ary_new();
  
  for (int i = 0; i < arr.ndim(); i++) {
    rb_ary_push(result, INT2NUM(arr.shape(i)));
  }
  
  return result;
}

VALUE rb_array_dtype(VALUE self) {
  mx::array& arr = get_array(self);
  return INT2NUM(static_cast<int>(arr.dtype().val()));
}

VALUE rb_array_ndim(VALUE self) {
  mx::array& arr = get_array(self);
  return INT2NUM(arr.ndim());
}

VALUE rb_array_size(VALUE self) {
  mx::array& arr = get_array(self);
  return INT2NUM(arr.size());
}

VALUE rb_array_itemsize(VALUE self) {
  mx::array& arr = get_array(self);
  return INT2NUM(arr.itemsize());
}

VALUE rb_array_nbytes(VALUE self) {
  mx::array& arr = get_array(self);
  return INT2NUM(arr.nbytes());
}

VALUE rb_array_to_s(VALUE self) {
  mx::array& arr = get_array(self);
  std::ostringstream os;
  os << arr;
  return rb_str_new_cstr(os.str().c_str());
}

VALUE rb_array_add(VALUE self, VALUE other) {
  mx::array& arr = get_array(self);
  ScalarOrArray soa(other);
  
  if (!is_comparable_with_array(soa)) {
    throw_invalid_operation("addition", soa);
  }
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::add(arr, to_array(soa, arr.dtype()))));
}

VALUE rb_array_sub(VALUE self, VALUE other) {
  mx::array& arr = get_array(self);
  ScalarOrArray soa(other);
  
  if (!is_comparable_with_array(soa)) {
    throw_invalid_operation("subtraction", soa);
  }
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::subtract(arr, to_array(soa, arr.dtype()))));
}

VALUE rb_array_mul(VALUE self, VALUE other) {
  mx::array& arr = get_array(self);
  ScalarOrArray soa(other);
  
  if (!is_comparable_with_array(soa)) {
    throw_invalid_operation("multiplication", soa);
  }
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::multiply(arr, to_array(soa, arr.dtype()))));
}

VALUE rb_array_div(VALUE self, VALUE other) {
  mx::array& arr = get_array(self);
  ScalarOrArray soa(other);
  
  if (!is_comparable_with_array(soa)) {
    throw_invalid_operation("division", soa);
  }
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::divide(arr, to_array(soa, arr.dtype()))));
}

VALUE rb_array_eq(VALUE self, VALUE other) {
  mx::array& arr = get_array(self);
  
  if (rb_obj_is_kind_of(other, rb_class_of(self))) {
    mx::array& other_arr = get_array(other);
    mx::array result = mx::equal(arr, other_arr);
    
    // If all elements are true, return true
    if (mx::all(result).item<bool>()) {
      return Qtrue;
    }
  }
  
  return Qfalse;
}

VALUE rb_array_neg(VALUE self) {
  mx::array& arr = get_array(self);
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(-arr));
}

VALUE rb_array_lt(VALUE self, VALUE other) {
  mx::array& arr = get_array(self);
  ScalarOrArray soa(other);
  
  if (!is_comparable_with_array(soa)) {
    throw_invalid_operation("less than", soa);
  }
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::less(arr, to_array(soa, arr.dtype()))));
}

VALUE rb_array_lte(VALUE self, VALUE other) {
  mx::array& arr = get_array(self);
  ScalarOrArray soa(other);
  
  if (!is_comparable_with_array(soa)) {
    throw_invalid_operation("less than or equal", soa);
  }
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::less_equal(arr, to_array(soa, arr.dtype()))));
}

VALUE rb_array_gt(VALUE self, VALUE other) {
  mx::array& arr = get_array(self);
  ScalarOrArray soa(other);
  
  if (!is_comparable_with_array(soa)) {
    throw_invalid_operation("greater than", soa);
  }
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::greater(arr, to_array(soa, arr.dtype()))));
}

VALUE rb_array_gte(VALUE self, VALUE other) {
  mx::array& arr = get_array(self);
  ScalarOrArray soa(other);
  
  if (!is_comparable_with_array(soa)) {
    throw_invalid_operation("greater than or equal", soa);
  }
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::greater_equal(arr, to_array(soa, arr.dtype()))));
}

VALUE rb_array_item(VALUE self) {
  mx::array& arr = get_array(self);
  
  if (arr.size() != 1) {
    rb_raise(rb_eRuntimeError, "item() can only be called on arrays with a single element");
  }
  
  return to_scalar(arr);
}

VALUE rb_array_tolist(VALUE self) {
  mx::array& arr = get_array(self);
  return tolist(arr);
}

// Mathematical operations
VALUE rb_array_abs(VALUE self) {
  mx::array& arr = get_array(self);
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::abs(arr)));
}

VALUE rb_array_square(VALUE self) {
  mx::array& arr = get_array(self);
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::square(arr)));
}

VALUE rb_array_sqrt(VALUE self) {
  mx::array& arr = get_array(self);
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::sqrt(arr)));
}

VALUE rb_array_rsqrt(VALUE self) {
  mx::array& arr = get_array(self);
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::rsqrt(arr)));
}

VALUE rb_array_reciprocal(VALUE self) {
  mx::array& arr = get_array(self);
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::reciprocal(arr)));
}

VALUE rb_array_exp(VALUE self) {
  mx::array& arr = get_array(self);
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::exp(arr)));
}

VALUE rb_array_log(VALUE self) {
  mx::array& arr = get_array(self);
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::log(arr)));
}

VALUE rb_array_log2(VALUE self) {
  mx::array& arr = get_array(self);
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::log2(arr)));
}

VALUE rb_array_log10(VALUE self) {
  mx::array& arr = get_array(self);
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::log10(arr)));
}

VALUE rb_array_log1p(VALUE self) {
  mx::array& arr = get_array(self);
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::log1p(arr)));
}

VALUE rb_array_sin(VALUE self) {
  mx::array& arr = get_array(self);
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::sin(arr)));
}

VALUE rb_array_cos(VALUE self) {
  mx::array& arr = get_array(self);
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::cos(arr)));
}

VALUE rb_array_power(VALUE self, VALUE other) {
  mx::array& arr = get_array(self);
  ScalarOrArray soa(other);
  
  if (!is_comparable_with_array(soa)) {
    throw_invalid_operation("power", soa);
  }
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::power(arr, to_array(soa, arr.dtype()))));
}

// Array manipulation methods
VALUE rb_array_reshape(int argc, VALUE* argv, VALUE self) {
  mx::array& arr = get_array(self);
  
  if (argc < 1) {
    rb_raise(rb_eArgError, "reshape requires shape arguments");
  }
  
  mx::Shape shape;
  if (RB_TYPE_P(argv[0], T_ARRAY) && argc == 1) {
    // Shape is passed as an array
    VALUE shape_array = argv[0];
    long len = RARRAY_LEN(shape_array);
    for (long i = 0; i < len; i++) {
      VALUE item = rb_ary_entry(shape_array, i);
      shape.push_back(NUM2INT(item));
    }
  } else {
    // Shape is passed as separate arguments
    for (int i = 0; i < argc; i++) {
      shape.push_back(NUM2INT(argv[i]));
    }
  }
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::reshape(arr, shape)));
}

VALUE rb_array_moveaxis(VALUE self, VALUE source, VALUE destination) {
  mx::array& arr = get_array(self);
  int src = NUM2INT(source);
  int dst = NUM2INT(destination);
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::moveaxis(arr, src, dst)));
}

VALUE rb_array_swapaxes(VALUE self, VALUE axis1, VALUE axis2) {
  mx::array& arr = get_array(self);
  int ax1 = NUM2INT(axis1);
  int ax2 = NUM2INT(axis2);
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::swapaxes(arr, ax1, ax2)));
}

VALUE rb_array_split(int argc, VALUE* argv, VALUE self) {
  mx::array& arr = get_array(self);
  
  if (argc < 1) {
    rb_raise(rb_eArgError, "split requires indices_or_sections argument");
  }
  
  VALUE indices_or_sections = argv[0];
  int axis = 0;
  
  if (argc >= 2) {
    axis = NUM2INT(argv[1]);
  }
  
  if (RB_TYPE_P(indices_or_sections, T_FIXNUM)) {
    // Split into equal sections
    int sections = NUM2INT(indices_or_sections);
    std::vector<mx::array> splits = mx::split(arr, sections, axis);
    
    VALUE result = rb_ary_new2(splits.size());
    for (size_t i = 0; i < splits.size(); i++) {
      rb_ary_push(result, Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                                          new mx::array(splits[i])));
    }
    return result;
  } else if (RB_TYPE_P(indices_or_sections, T_ARRAY)) {
    // Split at specific indices
    mx::Shape indices;
    long len = RARRAY_LEN(indices_or_sections);
    for (long i = 0; i < len; i++) {
      VALUE item = rb_ary_entry(indices_or_sections, i);
      indices.push_back(NUM2INT(item));
    }
    
    std::vector<mx::array> splits = mx::split(arr, indices, axis);
    
    VALUE result = rb_ary_new2(splits.size());
    for (size_t i = 0; i < splits.size(); i++) {
      rb_ary_push(result, Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                                          new mx::array(splits[i])));
    }
    return result;
  }
  
  rb_raise(rb_eArgError, "indices_or_sections must be an integer or array");
  return Qnil;
}

VALUE rb_array_flatten(int argc, VALUE* argv, VALUE self) {
  mx::array& arr = get_array(self);
  int start_axis = 0;
  int end_axis = -1;
  
  if (argc >= 1) {
    start_axis = NUM2INT(argv[0]);
  }
  
  if (argc >= 2) {
    end_axis = NUM2INT(argv[1]);
  }
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::flatten(arr, start_axis, end_axis)));
}

VALUE rb_array_squeeze(int argc, VALUE* argv, VALUE self) {
  mx::array& arr = get_array(self);
  
  if (argc == 0) {
    // Squeeze all dimensions
    return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                           new mx::array(mx::squeeze(arr)));
  } else if (argc == 1) {
    if (RB_TYPE_P(argv[0], T_ARRAY)) {
      // Array of axes to squeeze
      VALUE axes_array = argv[0];
      long len = RARRAY_LEN(axes_array);
      std::vector<int> axes;
      for (long i = 0; i < len; i++) {
        VALUE item = rb_ary_entry(axes_array, i);
        axes.push_back(NUM2INT(item));
      }
      return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                             new mx::array(mx::squeeze(arr, axes)));
    } else {
      // Single axis to squeeze
      int axis = NUM2INT(argv[0]);
      return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                             new mx::array(mx::squeeze(arr, axis)));
    }
  }
  
  rb_raise(rb_eArgError, "Wrong number of arguments for squeeze");
  return Qnil;
}

VALUE rb_array_transpose(int argc, VALUE* argv, VALUE self) {
  mx::array& arr = get_array(self);
  
  if (argc == 0) {
    // Full transpose
    return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                           new mx::array(mx::transpose(arr)));
  } else {
    // Transpose with specific axis ordering
    std::vector<int> axes;
    if (RB_TYPE_P(argv[0], T_ARRAY) && argc == 1) {
      // Axes passed as an array
      VALUE axes_array = argv[0];
      long len = RARRAY_LEN(axes_array);
      for (long i = 0; i < len; i++) {
        VALUE item = rb_ary_entry(axes_array, i);
        axes.push_back(NUM2INT(item));
      }
    } else {
      // Axes passed as separate arguments
      for (int i = 0; i < argc; i++) {
        axes.push_back(NUM2INT(argv[i]));
      }
    }
    
    return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                           new mx::array(mx::transpose(arr, axes)));
  }
}

VALUE rb_array_t(VALUE self) {
  mx::array& arr = get_array(self);
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::transpose(arr)));
}

// Helper for getting reduction axes
std::vector<int> get_reduce_axes(VALUE axis_value, int ndim) {
  std::vector<int> axes;
  
  if (NIL_P(axis_value)) {
    // If axis is nil, reduce over all dimensions
    return axes; // Empty means all axes
  } else if (RB_TYPE_P(axis_value, T_ARRAY)) {
    // Array of axes
    long len = RARRAY_LEN(axis_value);
    for (long i = 0; i < len; i++) {
      VALUE item = rb_ary_entry(axis_value, i);
      axes.push_back(NUM2INT(item));
    }
  } else {
    // Single axis
    axes.push_back(NUM2INT(axis_value));
  }
  
  return axes;
}

// Reduction operations
VALUE rb_array_all(int argc, VALUE* argv, VALUE self) {
  mx::array& arr = get_array(self);
  VALUE axis = Qnil;
  VALUE keepdims = Qfalse;
  
  // Parse arguments
  if (argc >= 1) axis = argv[0];
  if (argc >= 2) keepdims = argv[1];
  
  std::vector<int> axes = get_reduce_axes(axis, arr.ndim());
  bool keep_dims = RTEST(keepdims);
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::all(arr, axes, keep_dims)));
}

VALUE rb_array_any(int argc, VALUE* argv, VALUE self) {
  mx::array& arr = get_array(self);
  VALUE axis = Qnil;
  VALUE keepdims = Qfalse;
  
  // Parse arguments
  if (argc >= 1) axis = argv[0];
  if (argc >= 2) keepdims = argv[1];
  
  std::vector<int> axes = get_reduce_axes(axis, arr.ndim());
  bool keep_dims = RTEST(keepdims);
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::any(arr, axes, keep_dims)));
}

VALUE rb_array_sum(int argc, VALUE* argv, VALUE self) {
  mx::array& arr = get_array(self);
  VALUE axis = Qnil;
  VALUE keepdims = Qfalse;
  
  // Parse arguments
  if (argc >= 1) axis = argv[0];
  if (argc >= 2) keepdims = argv[1];
  
  std::vector<int> axes = get_reduce_axes(axis, arr.ndim());
  bool keep_dims = RTEST(keepdims);
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::sum(arr, axes, keep_dims)));
}

VALUE rb_array_prod(int argc, VALUE* argv, VALUE self) {
  mx::array& arr = get_array(self);
  VALUE axis = Qnil;
  VALUE keepdims = Qfalse;
  
  // Parse arguments
  if (argc >= 1) axis = argv[0];
  if (argc >= 2) keepdims = argv[1];
  
  std::vector<int> axes = get_reduce_axes(axis, arr.ndim());
  bool keep_dims = RTEST(keepdims);
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::prod(arr, axes, keep_dims)));
}

VALUE rb_array_min(int argc, VALUE* argv, VALUE self) {
  mx::array& arr = get_array(self);
  VALUE axis = Qnil;
  VALUE keepdims = Qfalse;
  
  // Parse arguments
  if (argc >= 1) axis = argv[0];
  if (argc >= 2) keepdims = argv[1];
  
  std::vector<int> axes = get_reduce_axes(axis, arr.ndim());
  bool keep_dims = RTEST(keepdims);
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::min(arr, axes, keep_dims)));
}

VALUE rb_array_max(int argc, VALUE* argv, VALUE self) {
  mx::array& arr = get_array(self);
  VALUE axis = Qnil;
  VALUE keepdims = Qfalse;
  
  // Parse arguments
  if (argc >= 1) axis = argv[0];
  if (argc >= 2) keepdims = argv[1];
  
  std::vector<int> axes = get_reduce_axes(axis, arr.ndim());
  bool keep_dims = RTEST(keepdims);
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::max(arr, axes, keep_dims)));
}

VALUE rb_array_mean(int argc, VALUE* argv, VALUE self) {
  mx::array& arr = get_array(self);
  VALUE axis = Qnil;
  VALUE keepdims = Qfalse;
  
  // Parse arguments
  if (argc >= 1) axis = argv[0];
  if (argc >= 2) keepdims = argv[1];
  
  std::vector<int> axes = get_reduce_axes(axis, arr.ndim());
  bool keep_dims = RTEST(keepdims);
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::mean(arr, axes, keep_dims)));
}

VALUE rb_array_logsumexp(int argc, VALUE* argv, VALUE self) {
  mx::array& arr = get_array(self);
  VALUE axis = Qnil;
  VALUE keepdims = Qfalse;
  
  // Parse arguments
  if (argc >= 1) axis = argv[0];
  if (argc >= 2) keepdims = argv[1];
  
  std::vector<int> axes = get_reduce_axes(axis, arr.ndim());
  bool keep_dims = RTEST(keepdims);
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::logsumexp(arr, axes, keep_dims)));
}

VALUE rb_array_std(int argc, VALUE* argv, VALUE self) {
  mx::array& arr = get_array(self);
  VALUE axis = Qnil;
  VALUE keepdims = Qfalse;
  VALUE ddof_val = INT2NUM(0);
  
  // Parse arguments
  if (argc >= 1) axis = argv[0];
  if (argc >= 2) keepdims = argv[1];
  if (argc >= 3) ddof_val = argv[2];
  
  std::vector<int> axes = get_reduce_axes(axis, arr.ndim());
  bool keep_dims = RTEST(keepdims);
  int ddof = NUM2INT(ddof_val);
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::std(arr, axes, keep_dims, ddof)));
}

VALUE rb_array_var(int argc, VALUE* argv, VALUE self) {
  mx::array& arr = get_array(self);
  VALUE axis = Qnil;
  VALUE keepdims = Qfalse;
  VALUE ddof_val = INT2NUM(0);
  
  // Parse arguments
  if (argc >= 1) axis = argv[0];
  if (argc >= 2) keepdims = argv[1];
  if (argc >= 3) ddof_val = argv[2];
  
  std::vector<int> axes = get_reduce_axes(axis, arr.ndim());
  bool keep_dims = RTEST(keepdims);
  int ddof = NUM2INT(ddof_val);
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::var(arr, axes, keep_dims, ddof)));
}

VALUE rb_array_argmin(int argc, VALUE* argv, VALUE self) {
  mx::array& arr = get_array(self);
  
  if (argc == 0) {
    // Global argmin
    return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                           new mx::array(mx::argmin(arr)));
  } else {
    int axis = NUM2INT(argv[0]);
    bool keepdims = argc >= 2 ? RTEST(argv[1]) : false;
    
    return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                           new mx::array(mx::argmin(arr, axis, keepdims)));
  }
}

VALUE rb_array_argmax(int argc, VALUE* argv, VALUE self) {
  mx::array& arr = get_array(self);
  
  if (argc == 0) {
    // Global argmax
    return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                           new mx::array(mx::argmax(arr)));
  } else {
    int axis = NUM2INT(argv[0]);
    bool keepdims = argc >= 2 ? RTEST(argv[1]) : false;
    
    return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                           new mx::array(mx::argmax(arr, axis, keepdims)));
  }
}

VALUE rb_array_cumsum(int argc, VALUE* argv, VALUE self) {
  mx::array& arr = get_array(self);
  
  // Default values
  std::optional<int> axis = std::nullopt;
  bool reverse = false;
  bool inclusive = true;
  
  // Parse arguments
  if (argc >= 1 && !NIL_P(argv[0])) {
    axis = NUM2INT(argv[0]);
  }
  
  if (argc >= 2) {
    reverse = RTEST(argv[1]);
  }
  
  if (argc >= 3) {
    inclusive = RTEST(argv[2]);
  }
  
  if (axis.has_value()) {
    return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                           new mx::array(mx::cumsum(arr, *axis, reverse, inclusive)));
  } else {
    // If no axis is specified, reshape to 1D and use axis 0
    return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                           new mx::array(mx::cumsum(mx::reshape(arr, {-1}), 0, reverse, inclusive)));
  }
}

VALUE rb_array_cumprod(int argc, VALUE* argv, VALUE self) {
  mx::array& arr = get_array(self);
  
  // Default values
  std::optional<int> axis = std::nullopt;
  bool reverse = false;
  bool inclusive = true;
  
  // Parse arguments
  if (argc >= 1 && !NIL_P(argv[0])) {
    axis = NUM2INT(argv[0]);
  }
  
  if (argc >= 2) {
    reverse = RTEST(argv[1]);
  }
  
  if (argc >= 3) {
    inclusive = RTEST(argv[2]);
  }
  
  if (axis.has_value()) {
    return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                           new mx::array(mx::cumprod(arr, *axis, reverse, inclusive)));
  } else {
    // If no axis is specified, reshape to 1D and use axis 0
    return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                           new mx::array(mx::cumprod(mx::reshape(arr, {-1}), 0, reverse, inclusive)));
  }
}

VALUE rb_array_cummax(int argc, VALUE* argv, VALUE self) {
  mx::array& arr = get_array(self);
  
  // Default values
  std::optional<int> axis = std::nullopt;
  bool reverse = false;
  bool inclusive = true;
  
  // Parse arguments
  if (argc >= 1 && !NIL_P(argv[0])) {
    axis = NUM2INT(argv[0]);
  }
  
  if (argc >= 2) {
    reverse = RTEST(argv[1]);
  }
  
  if (argc >= 3) {
    inclusive = RTEST(argv[2]);
  }
  
  if (axis.has_value()) {
    return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                           new mx::array(mx::cummax(arr, *axis, reverse, inclusive)));
  } else {
    // If no axis is specified, reshape to 1D and use axis 0
    return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                           new mx::array(mx::cummax(mx::reshape(arr, {-1}), 0, reverse, inclusive)));
  }
}

VALUE rb_array_cummin(int argc, VALUE* argv, VALUE self) {
  mx::array& arr = get_array(self);
  
  // Default values
  std::optional<int> axis = std::nullopt;
  bool reverse = false;
  bool inclusive = true;
  
  // Parse arguments
  if (argc >= 1 && !NIL_P(argv[0])) {
    axis = NUM2INT(argv[0]);
  }
  
  if (argc >= 2) {
    reverse = RTEST(argv[1]);
  }
  
  if (argc >= 3) {
    inclusive = RTEST(argv[2]);
  }
  
  if (axis.has_value()) {
    return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                           new mx::array(mx::cummin(arr, *axis, reverse, inclusive)));
  } else {
    // If no axis is specified, reshape to 1D and use axis 0
    return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                           new mx::array(mx::cummin(mx::reshape(arr, {-1}), 0, reverse, inclusive)));
  }
}

VALUE rb_array_round(int argc, VALUE* argv, VALUE self) {
  mx::array& arr = get_array(self);
  int decimals = 0;
  
  if (argc >= 1) {
    decimals = NUM2INT(argv[0]);
  }
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::round(arr, decimals)));
}

VALUE rb_array_conj(VALUE self) {
  mx::array& arr = get_array(self);
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::conjugate(arr)));
}

VALUE rb_array_diagonal(int argc, VALUE* argv, VALUE self) {
  mx::array& arr = get_array(self);
  int offset = 0;
  int axis1 = 0;
  int axis2 = 1;
  
  if (argc >= 1) offset = NUM2INT(argv[0]);
  if (argc >= 2) axis1 = NUM2INT(argv[1]);
  if (argc >= 3) axis2 = NUM2INT(argv[2]);
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::diagonal(arr, offset, axis1, axis2)));
}

VALUE rb_array_diag(int argc, VALUE* argv, VALUE self) {
  mx::array& arr = get_array(self);
  int k = 0;
  
  if (argc >= 1) k = NUM2INT(argv[0]);
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::diag(arr, k)));
}

VALUE rb_array_view(VALUE self, VALUE dtype_val) {
  mx::array& arr = get_array(self);
  int dtype_int = NUM2INT(dtype_val);
  // Initialize with a default value
  mx::Dtype dtype = mx::float32;
  switch (dtype_int) {
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
    default: dtype = mx::float32;
  }
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::view(arr, dtype)));
}

VALUE rb_array_matmul(VALUE self, VALUE other) {
  mx::array& arr = get_array(self);
  
  if (!rb_obj_is_kind_of(other, rb_class_of(self))) {
    rb_raise(rb_eTypeError, "Matrix multiplication requires an MLX array");
  }
  
  mx::array& other_arr = get_array(other);
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::matmul(arr, other_arr)));
}

VALUE rb_array_floor_div(VALUE self, VALUE other) {
  mx::array& arr = get_array(self);
  ScalarOrArray soa(other);
  
  if (!is_comparable_with_array(soa)) {
    throw_invalid_operation("floor division", soa);
  }
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::floor_divide(arr, to_array(soa, arr.dtype()))));
}

VALUE rb_array_mod(VALUE self, VALUE other) {
  mx::array& arr = get_array(self);
  ScalarOrArray soa(other);
  
  if (!is_comparable_with_array(soa)) {
    throw_invalid_operation("modulo", soa);
  }
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::remainder(arr, to_array(soa, arr.dtype()))));
}

VALUE rb_array_neq(VALUE self, VALUE other) {
  mx::array& arr = get_array(self);
  ScalarOrArray soa(other);
  
  if (!is_comparable_with_array(soa)) {
    return Qtrue; // Not equal to non-comparable objects
  }
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::not_equal(arr, to_array(soa, arr.dtype()))));
}

// Implement Enumerable support
VALUE rb_array_each(VALUE self) {
  mx::array& arr = get_array(self);
  
  if (!rb_block_given_p()) {
    rb_raise(rb_eArgError, "No block given");
  }
  
  // Different handling based on dimensions
  if (arr.ndim() == 1) {
    // For 1D arrays, iterate over each element
    for (int i = 0; i < arr.shape(0); i++) {
      mx::array idx = mx::array(i);
      mx::array val = mx::take(arr, idx, 0);
      rb_yield(Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, new mx::array(val)));
    }
  } else {
    // For multi-dimensional arrays, iterate over the first dimension
    for (int i = 0; i < arr.shape(0); i++) {
      mx::array idx = mx::array(i);
      mx::array slice = mx::take(arr, idx, 0);
      mx::array squeezed = mx::squeeze(slice, 0);
      rb_yield(Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, new mx::array(squeezed)));
    }
  }
  
  return self;
}

// Helper function to check if an object responds to to_mlx_array
bool responds_to_to_mlx_array(VALUE obj) {
  return rb_respond_to(obj, rb_intern("to_mlx_array"));
}

// Convert object using to_mlx_array if available
mx::array convert_using_to_mlx_array(VALUE obj) {
  VALUE mlx_array = rb_funcall(obj, rb_intern("to_mlx_array"), 0);
  
  if (!rb_obj_is_kind_of(mlx_array, rb_path2class("MLX::Array"))) {
    rb_raise(rb_eTypeError, "to_mlx_array must return an MLX::Array");
  }
  
  mx::array& arr = get_array(mlx_array);
  return arr;
}

// Bitwise operations
VALUE rb_array_invert(VALUE self) {
  mx::array& arr = get_array(self);
  
  if (mx::issubdtype(arr.dtype(), mx::inexact)) {
    rb_raise(rb_eTypeError, "Floating point types not allowed with bitwise inversion");
  }
  
  if (arr.dtype() == mx::bool_) {
    return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                           new mx::array(mx::logical_not(arr)));
  }
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::bitwise_invert(arr)));
}

VALUE rb_array_bitwise_and(VALUE self, VALUE other) {
  mx::array& arr = get_array(self);
  ScalarOrArray soa(other);
  
  if (!is_comparable_with_array(soa)) {
    throw_invalid_operation("bitwise and", soa);
  }
  
  auto b = to_array(soa, arr.dtype());
  if (mx::issubdtype(arr.dtype(), mx::inexact) || mx::issubdtype(b.dtype(), mx::inexact)) {
    rb_raise(rb_eTypeError, "Floating point types not allowed with bitwise and");
  }
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::bitwise_and(arr, b)));
}

VALUE rb_array_bitwise_or(VALUE self, VALUE other) {
  mx::array& arr = get_array(self);
  ScalarOrArray soa(other);
  
  if (!is_comparable_with_array(soa)) {
    throw_invalid_operation("bitwise or", soa);
  }
  
  auto b = to_array(soa, arr.dtype());
  if (mx::issubdtype(arr.dtype(), mx::inexact) || mx::issubdtype(b.dtype(), mx::inexact)) {
    rb_raise(rb_eTypeError, "Floating point types not allowed with bitwise or");
  }
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::bitwise_or(arr, b)));
}

VALUE rb_array_bitwise_xor(VALUE self, VALUE other) {
  mx::array& arr = get_array(self);
  ScalarOrArray soa(other);
  
  if (!is_comparable_with_array(soa)) {
    throw_invalid_operation("bitwise xor", soa);
  }
  
  auto b = to_array(soa, arr.dtype());
  if (mx::issubdtype(arr.dtype(), mx::inexact) || mx::issubdtype(b.dtype(), mx::inexact)) {
    rb_raise(rb_eTypeError, "Floating point types not allowed with bitwise xor");
  }
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::bitwise_xor(arr, b)));
}

VALUE rb_array_left_shift(VALUE self, VALUE other) {
  mx::array& arr = get_array(self);
  ScalarOrArray soa(other);
  
  if (!is_comparable_with_array(soa)) {
    throw_invalid_operation("left shift", soa);
  }
  
  auto b = to_array(soa, arr.dtype());
  if (mx::issubdtype(arr.dtype(), mx::inexact) || mx::issubdtype(b.dtype(), mx::inexact)) {
    rb_raise(rb_eTypeError, "Floating point types not allowed with left shift");
  }
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::left_shift(arr, b)));
}

VALUE rb_array_right_shift(VALUE self, VALUE other) {
  mx::array& arr = get_array(self);
  ScalarOrArray soa(other);
  
  if (!is_comparable_with_array(soa)) {
    throw_invalid_operation("right shift", soa);
  }
  
  auto b = to_array(soa, arr.dtype());
  if (mx::issubdtype(arr.dtype(), mx::inexact) || mx::issubdtype(b.dtype(), mx::inexact)) {
    rb_raise(rb_eTypeError, "Floating point types not allowed with right shift");
  }
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::right_shift(arr, b)));
}

VALUE rb_array_astype(int argc, VALUE* argv, VALUE self) {
  mx::array& arr = get_array(self);
  
  if (argc < 1) {
    rb_raise(rb_eArgError, "astype requires a dtype argument");
  }
  
  int dtype_val = NUM2INT(argv[0]);
  // Initialize with a default value
  mx::Dtype dtype = mx::float32;
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
    default: dtype = mx::float32;
  }
  
  return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                         new mx::array(mx::astype(arr, dtype)));
}

// Array indexing methods
VALUE rb_array_aref(VALUE self, VALUE index) {
  mx::array& arr = get_array(self);
  
  // Handle different index types
  if (RB_TYPE_P(index, T_FIXNUM)) {
    // Single integer index
    int idx = NUM2INT(index);
    mx::array idx_arr = mx::array(idx);
    mx::array result = mx::take(arr, idx_arr, 0);
    
    // If taking from first dimension, squeeze the result
    if (arr.ndim() > 1) {
      result = mx::squeeze(result, 0);
    }
    
    return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                           new mx::array(result));
  } else if (RB_TYPE_P(index, T_ARRAY)) {
    // Array of indices or multi-dimensional indexing
    long len = RARRAY_LEN(index);
    
    if (len == arr.ndim()) {
      // Multi-dimensional indexing like [1, 2, 3]
      std::vector<mx::array> indices;
      std::vector<int> axes;
      
      for (long i = 0; i < len; i++) {
        VALUE idx = rb_ary_entry(index, i);
        if (RB_TYPE_P(idx, T_FIXNUM)) {
          indices.push_back(mx::array(NUM2INT(idx)));
        } else if (rb_obj_is_kind_of(idx, rb_path2class("MLX::Array"))) {
          indices.push_back(get_array(idx));
        } else {
          rb_raise(rb_eTypeError, "Index elements must be integers or MLX arrays");
        }
        
        // Add axis value
        axes.push_back(i);
      }
      
      mx::Shape slice_sizes = arr.shape();
      for (size_t i = 0; i < axes.size(); i++) {
        slice_sizes[axes[i]] = 1;
      }
      mx::array result = mx::gather(arr, indices, axes, slice_sizes);
      return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                             new mx::array(result));
    } else {
      // Array of indices for first dimension like [1, 3, 5]
      // Convert Ruby array to a vector of integers first
      std::vector<int32_t> indices_vec;
      
      for (long i = 0; i < len; i++) {
        VALUE idx = rb_ary_entry(index, i);
        if (RB_TYPE_P(idx, T_FIXNUM)) {
          indices_vec.push_back(NUM2INT(idx));
        } else {
          rb_raise(rb_eTypeError, "Index elements must be integers");
        }
      }
      
      // Create MLX array properly with shape and data type
      mx::array idx_arr({static_cast<int>(indices_vec.size())}, mx::int32);
      for (size_t i = 0; i < indices_vec.size(); i++) {
        // Set each element individually
        idx_arr.data<int32_t>()[i] = indices_vec[i];
      }
      mx::array result = mx::take(arr, idx_arr, 0);
      
      return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                             new mx::array(result));
    }
  } else if (rb_obj_is_kind_of(index, rb_path2class("MLX::Array"))) {
    // MLX array as index
    mx::array& idx_arr = get_array(index);
    mx::array result = mx::take(arr, idx_arr, 0);
    
    return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                           new mx::array(result));
  } else if (rb_obj_is_kind_of(index, rb_path2class("Range"))) {
    // Ruby Range as index
    int start, stop;
    VALUE range_begin = rb_funcall(index, rb_intern("begin"), 0);
    VALUE range_end = rb_funcall(index, rb_intern("end"), 0);
    VALUE exclude_end = rb_funcall(index, rb_intern("exclude_end?"), 0);
    
    start = NIL_P(range_begin) ? 0 : NUM2INT(range_begin);
    stop = NIL_P(range_end) ? arr.shape(0) : NUM2INT(range_end);
    
    if (RTEST(exclude_end)) {
      stop -= 1;
    }
    
    // Calculate the size of the range
    int size = stop - start + 1;
    
    // Create a 1D array for indices
    mx::array idx_arr({size}, mx::int32);
    
    // Fill array with range values
    for (int i = 0; i < size; i++) {
      mx::array idx = mx::array(i);
      mx::array val = mx::array(start + i);
      idx_arr = mx::scatter(idx_arr, {idx}, val, 0);
    }
    mx::array result = mx::take(arr, idx_arr, 0);
    
    return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                           new mx::array(result));
  }
  
  rb_raise(rb_eTypeError, "Invalid index type");
  return Qnil; // Unreachable
}

VALUE rb_array_aset(VALUE self, VALUE index, VALUE val) {
  mx::array& arr = get_array(self);
  ScalarOrArray soa(val);
  
  if (!is_comparable_with_array(soa)) {
    throw_invalid_operation("assignment", soa);
  }
  
  auto value = to_array(soa, arr.dtype());
  
  // Handle different index types
  if (RB_TYPE_P(index, T_FIXNUM)) {
    // Single integer index
    int idx = NUM2INT(index);
    mx::array idx_arr = mx::array(idx);
    
    get_array(self) = mx::scatter(arr, {idx_arr}, value, 0);
    return val;
  } else if (RB_TYPE_P(index, T_ARRAY)) {
    // Array of indices or multi-dimensional indexing
    long len = RARRAY_LEN(index);
    
    if (len == arr.ndim()) {
      // Multi-dimensional indexing like [1, 2, 3]
      std::vector<mx::array> indices;
      std::vector<int> axes;
      
      for (long i = 0; i < len; i++) {
        VALUE idx = rb_ary_entry(index, i);
        if (RB_TYPE_P(idx, T_FIXNUM)) {
          indices.push_back(mx::array(NUM2INT(idx)));
        } else if (rb_obj_is_kind_of(idx, rb_path2class("MLX::Array"))) {
          indices.push_back(get_array(idx));
        } else {
          rb_raise(rb_eTypeError, "Index elements must be integers or MLX arrays");
        }
        
        // Add axis value
        axes.push_back(i);
      }
      
      get_array(self) = mx::scatter(arr, indices, value, axes);
      return val;
    } else {
      // Array of indices for first dimension like [1, 3, 5]
      // Convert Ruby array to a vector of integers first
      std::vector<int32_t> indices_vec;
      
      for (long i = 0; i < len; i++) {
        VALUE idx = rb_ary_entry(index, i);
        if (RB_TYPE_P(idx, T_FIXNUM)) {
          indices_vec.push_back(NUM2INT(idx));
        } else {
          rb_raise(rb_eTypeError, "Index elements must be integers");
        }
      }
      
      // Create MLX array properly with shape and data type
      mx::array idx_arr({static_cast<int>(indices_vec.size())}, mx::int32);
      for (size_t i = 0; i < indices_vec.size(); i++) {
        // Set each element individually
        idx_arr.data<int32_t>()[i] = indices_vec[i];
      }
      get_array(self) = mx::scatter(arr, {idx_arr}, value, 0);
      return val;
    }
  } else if (rb_obj_is_kind_of(index, rb_path2class("MLX::Array"))) {
    // MLX array as index
    mx::array& idx_arr = get_array(index);
    get_array(self) = mx::scatter(arr, {idx_arr}, value, 0);
    return val;
  } else if (rb_obj_is_kind_of(index, rb_path2class("Range"))) {
    // Ruby Range as index
    int start, stop;
    VALUE range_begin = rb_funcall(index, rb_intern("begin"), 0);
    VALUE range_end = rb_funcall(index, rb_intern("end"), 0);
    VALUE exclude_end = rb_funcall(index, rb_intern("exclude_end?"), 0);
    
    start = NIL_P(range_begin) ? 0 : NUM2INT(range_begin);
    stop = NIL_P(range_end) ? arr.shape(0) : NUM2INT(range_end);
    
    if (RTEST(exclude_end)) {
      stop -= 1;
    }
    
    // Calculate the size of the range
    int size = stop - start + 1;
    
    // Create a 1D array for indices
    mx::array idx_arr({size}, mx::int32);
    
    // Fill array with range values
    for (int i = 0; i < size; i++) {
      mx::array idx = mx::array(i);
      mx::array val = mx::array(start + i);
      idx_arr = mx::scatter(idx_arr, {idx}, val, 0);
    }
    get_array(self) = mx::scatter(arr, {idx_arr}, value, 0);
    return val;
  }
  
  rb_raise(rb_eTypeError, "Invalid index type");
  return Qnil; // Unreachable
}

// Helper for array.at[idx]
VALUE rb_array_at_aref(VALUE self, VALUE index) {
  ArrayAt* at_ptr;
  Data_Get_Struct(self, ArrayAt, at_ptr);
  
  at_ptr->set_indices(index);
  return self;
}

void init_array(VALUE module) {
  // Define the Array class
  VALUE array_class = rb_define_class_under(module, "Array", rb_cObject);
  rb_define_alloc_func(array_class, rb_alloc_array);
  
  // Include Enumerable for iteration support
  rb_include_module(array_class, rb_const_get(rb_cObject, rb_intern("Enumerable")));
  
  // Instance methods
  rb_define_method(array_class, "initialize", RUBY_METHOD_FUNC(rb_array_initialize), -1);
  rb_define_method(array_class, "shape", RUBY_METHOD_FUNC(rb_array_shape), 0);
  rb_define_method(array_class, "dtype", RUBY_METHOD_FUNC(rb_array_dtype), 0);
  rb_define_method(array_class, "ndim", RUBY_METHOD_FUNC(rb_array_ndim), 0);
  rb_define_method(array_class, "size", RUBY_METHOD_FUNC(rb_array_size), 0);
  rb_define_method(array_class, "itemsize", RUBY_METHOD_FUNC(rb_array_itemsize), 0);
  rb_define_method(array_class, "nbytes", RUBY_METHOD_FUNC(rb_array_nbytes), 0);
  rb_define_method(array_class, "to_s", RUBY_METHOD_FUNC(rb_array_to_s), 0);
  rb_define_method(array_class, "item", RUBY_METHOD_FUNC(rb_array_item), 0);
  rb_define_method(array_class, "tolist", RUBY_METHOD_FUNC(rb_array_tolist), 0);
  rb_define_method(array_class, "astype", RUBY_METHOD_FUNC(rb_array_astype), -1);
  
  // Array indexing
  rb_define_method(array_class, "[]", RUBY_METHOD_FUNC(rb_array_aref), 1);
  rb_define_method(array_class, "[]=", RUBY_METHOD_FUNC(rb_array_aset), 2);
  
  // Enumerable support
  rb_define_method(array_class, "each", RUBY_METHOD_FUNC(rb_array_each), 0);
  
  // Mathematical operations
  rb_define_method(array_class, "abs", RUBY_METHOD_FUNC(rb_array_abs), 0);
  rb_define_method(array_class, "square", RUBY_METHOD_FUNC(rb_array_square), 0);
  rb_define_method(array_class, "sqrt", RUBY_METHOD_FUNC(rb_array_sqrt), 0);
  rb_define_method(array_class, "rsqrt", RUBY_METHOD_FUNC(rb_array_rsqrt), 0);
  rb_define_method(array_class, "reciprocal", RUBY_METHOD_FUNC(rb_array_reciprocal), 0);
  rb_define_method(array_class, "exp", RUBY_METHOD_FUNC(rb_array_exp), 0);
  rb_define_method(array_class, "log", RUBY_METHOD_FUNC(rb_array_log), 0);
  rb_define_method(array_class, "log2", RUBY_METHOD_FUNC(rb_array_log2), 0);
  rb_define_method(array_class, "log10", RUBY_METHOD_FUNC(rb_array_log10), 0);
  rb_define_method(array_class, "log1p", RUBY_METHOD_FUNC(rb_array_log1p), 0);
  rb_define_method(array_class, "sin", RUBY_METHOD_FUNC(rb_array_sin), 0);
  rb_define_method(array_class, "cos", RUBY_METHOD_FUNC(rb_array_cos), 0);
  rb_define_method(array_class, "**", RUBY_METHOD_FUNC(rb_array_power), 1);
  
  // Array manipulation methods
  rb_define_method(array_class, "reshape", RUBY_METHOD_FUNC(rb_array_reshape), -1);
  rb_define_method(array_class, "flatten", RUBY_METHOD_FUNC(rb_array_flatten), -1);
  rb_define_method(array_class, "squeeze", RUBY_METHOD_FUNC(rb_array_squeeze), -1);
  rb_define_method(array_class, "transpose", RUBY_METHOD_FUNC(rb_array_transpose), -1);
  rb_define_method(array_class, "t", RUBY_METHOD_FUNC(rb_array_t), 0);
  rb_define_method(array_class, "moveaxis", RUBY_METHOD_FUNC(rb_array_moveaxis), 2);
  rb_define_method(array_class, "swapaxes", RUBY_METHOD_FUNC(rb_array_swapaxes), 2);
  rb_define_method(array_class, "split", RUBY_METHOD_FUNC(rb_array_split), -1);
  rb_define_method(array_class, "diagonal", RUBY_METHOD_FUNC(rb_array_diagonal), -1);
  rb_define_method(array_class, "diag", RUBY_METHOD_FUNC(rb_array_diag), -1);
  
  // Reduction operations
  rb_define_method(array_class, "all", RUBY_METHOD_FUNC(rb_array_all), -1);
  rb_define_method(array_class, "any", RUBY_METHOD_FUNC(rb_array_any), -1);
  rb_define_method(array_class, "sum", RUBY_METHOD_FUNC(rb_array_sum), -1);
  rb_define_method(array_class, "prod", RUBY_METHOD_FUNC(rb_array_prod), -1);
  rb_define_method(array_class, "min", RUBY_METHOD_FUNC(rb_array_min), -1);
  rb_define_method(array_class, "max", RUBY_METHOD_FUNC(rb_array_max), -1);
  rb_define_method(array_class, "mean", RUBY_METHOD_FUNC(rb_array_mean), -1);
  rb_define_method(array_class, "logsumexp", RUBY_METHOD_FUNC(rb_array_logsumexp), -1);
  rb_define_method(array_class, "std", RUBY_METHOD_FUNC(rb_array_std), -1);
  rb_define_method(array_class, "var", RUBY_METHOD_FUNC(rb_array_var), -1);
  
  // Additional operations
  rb_define_method(array_class, "argmin", RUBY_METHOD_FUNC(rb_array_argmin), -1);
  rb_define_method(array_class, "argmax", RUBY_METHOD_FUNC(rb_array_argmax), -1);
  rb_define_method(array_class, "cumsum", RUBY_METHOD_FUNC(rb_array_cumsum), -1);
  rb_define_method(array_class, "cumprod", RUBY_METHOD_FUNC(rb_array_cumprod), -1);
  rb_define_method(array_class, "cummax", RUBY_METHOD_FUNC(rb_array_cummax), -1);
  rb_define_method(array_class, "cummin", RUBY_METHOD_FUNC(rb_array_cummin), -1);
  rb_define_method(array_class, "round", RUBY_METHOD_FUNC(rb_array_round), -1);
  rb_define_method(array_class, "conj", RUBY_METHOD_FUNC(rb_array_conj), 0);
  rb_define_method(array_class, "view", RUBY_METHOD_FUNC(rb_array_view), 1);
  
  // Bitwise operations
  rb_define_method(array_class, "~@", RUBY_METHOD_FUNC(rb_array_invert), 0);
  rb_define_method(array_class, "&", RUBY_METHOD_FUNC(rb_array_bitwise_and), 1);
  rb_define_method(array_class, "|", RUBY_METHOD_FUNC(rb_array_bitwise_or), 1);
  rb_define_method(array_class, "^", RUBY_METHOD_FUNC(rb_array_bitwise_xor), 1);
  rb_define_method(array_class, "<<", RUBY_METHOD_FUNC(rb_array_left_shift), 1);
  rb_define_method(array_class, ">>", RUBY_METHOD_FUNC(rb_array_right_shift), 1);
  
  // Arithmetic operators
  rb_define_method(array_class, "+", RUBY_METHOD_FUNC(rb_array_add), 1);
  rb_define_method(array_class, "-", RUBY_METHOD_FUNC(rb_array_sub), 1);
  rb_define_method(array_class, "*", RUBY_METHOD_FUNC(rb_array_mul), 1);
  rb_define_method(array_class, "/", RUBY_METHOD_FUNC(rb_array_div), 1);
  rb_define_method(array_class, "-@", RUBY_METHOD_FUNC(rb_array_neg), 0);
  rb_define_method(array_class, "matmul", RUBY_METHOD_FUNC(rb_array_matmul), 1);
  rb_define_method(array_class, "@", RUBY_METHOD_FUNC(rb_array_matmul), 1); // Ruby 2.5+ operator
  rb_define_method(array_class, "floor_div", RUBY_METHOD_FUNC(rb_array_floor_div), 1);
  rb_define_method(array_class, "%", RUBY_METHOD_FUNC(rb_array_mod), 1);
  
  // Comparison operators
  rb_define_method(array_class, "==", RUBY_METHOD_FUNC(rb_array_eq), 1);
  rb_define_method(array_class, "!=", RUBY_METHOD_FUNC(rb_array_neq), 1);
  rb_define_method(array_class, "<", RUBY_METHOD_FUNC(rb_array_lt), 1);
  rb_define_method(array_class, "<=", RUBY_METHOD_FUNC(rb_array_lte), 1);
  rb_define_method(array_class, ">", RUBY_METHOD_FUNC(rb_array_gt), 1);
  rb_define_method(array_class, ">=", RUBY_METHOD_FUNC(rb_array_gte), 1);
  
  // Define the Dtype subclass and constants
  VALUE dtype_class = rb_define_class_under(module, "Dtype", rb_cObject);
  
  // Define dtype constants
  rb_define_const(module, "BOOL", INT2NUM(static_cast<int>(mx::bool_.val())));
  rb_define_const(module, "UINT8", INT2NUM(static_cast<int>(mx::uint8.val())));
  rb_define_const(module, "UINT16", INT2NUM(static_cast<int>(mx::uint16.val())));
  rb_define_const(module, "UINT32", INT2NUM(static_cast<int>(mx::uint32.val())));
  rb_define_const(module, "UINT64", INT2NUM(static_cast<int>(mx::uint64.val())));
  rb_define_const(module, "INT8", INT2NUM(static_cast<int>(mx::int8.val())));
  rb_define_const(module, "INT16", INT2NUM(static_cast<int>(mx::int16.val())));
  rb_define_const(module, "INT32", INT2NUM(static_cast<int>(mx::int32.val())));
  rb_define_const(module, "INT64", INT2NUM(static_cast<int>(mx::int64.val())));
  rb_define_const(module, "FLOAT16", INT2NUM(static_cast<int>(mx::float16.val())));
  rb_define_const(module, "FLOAT32", INT2NUM(static_cast<int>(mx::float32.val())));
  rb_define_const(module, "FLOAT64", INT2NUM(static_cast<int>(mx::float64.val())));
  rb_define_const(module, "BFLOAT16", INT2NUM(static_cast<int>(mx::bfloat16.val())));
  rb_define_const(module, "COMPLEX64", INT2NUM(static_cast<int>(mx::complex64.val())));
  
  // Dtype category constants
  rb_define_const(module, "COMPLEXFLOATING", INT2NUM(static_cast<int>(mx::complexfloating)));
  rb_define_const(module, "FLOATING", INT2NUM(static_cast<int>(mx::floating)));
  rb_define_const(module, "INEXACT", INT2NUM(static_cast<int>(mx::inexact)));
  rb_define_const(module, "SIGNEDINTEGER", INT2NUM(static_cast<int>(mx::signedinteger)));
  rb_define_const(module, "UNSIGNEDINTEGER", INT2NUM(static_cast<int>(mx::unsignedinteger)));
  rb_define_const(module, "INTEGER", INT2NUM(static_cast<int>(mx::integer)));
  rb_define_const(module, "NUMBER", INT2NUM(static_cast<int>(mx::number)));
  rb_define_const(module, "GENERIC", INT2NUM(static_cast<int>(mx::generic)));
} 