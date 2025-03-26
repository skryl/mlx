#include <ruby.h>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <vector>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

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
      return mx::array(); // Return empty array when done
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

// Ruby GC free function for mx::array
void rb_free_array(void* ptr) {
  mx::array* arr = static_cast<mx::array*>(ptr);
  delete arr;
}

// Ruby object allocation function
VALUE rb_alloc_array(VALUE klass) {
  mx::array* ptr = new mx::array();
  return Data_Wrap_Struct(klass, 0, rb_free_array, ptr);
}

// Helper to extract mx::array from Ruby VALUE
mx::array& get_array(VALUE self) {
  mx::array* ptr;
  Data_Get_Struct(self, mx::array, ptr);
  return *ptr;
}

// Array methods for Ruby
VALUE rb_array_initialize(int argc, VALUE* argv, VALUE self) {
  // Simple initialization with shape and dtype
  if (argc == 0) {
    // Create empty array
    get_array(self) = mx::array();
  } else if (argc == 1 && RB_TYPE_P(argv[0], T_ARRAY)) {
    // Create from Ruby array
    VALUE rb_ary = argv[0];
    std::vector<double> data;
    
    for (long i = 0; i < RARRAY_LEN(rb_ary); i++) {
      VALUE item = rb_ary_entry(rb_ary, i);
      data.push_back(NUM2DBL(item));
    }
    
    get_array(self) = mx::array(data);
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

VALUE rb_array_to_s(VALUE self) {
  mx::array& arr = get_array(self);
  std::ostringstream os;
  os << arr;
  return rb_str_new_cstr(os.str().c_str());
}

VALUE rb_array_add(VALUE self, VALUE other) {
  mx::array& arr = get_array(self);
  
  if (rb_obj_is_kind_of(other, rb_cNumeric)) {
    double val = NUM2DBL(other);
    return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                           new mx::array(arr + val));
  } else {
    mx::array& other_arr = get_array(other);
    return Data_Wrap_Struct(rb_class_of(self), 0, rb_free_array, 
                           new mx::array(arr + other_arr));
  }
}

void init_array(VALUE module) {
  // Define the Array class
  VALUE array_class = rb_define_class_under(module, "Array", rb_cObject);
  rb_define_alloc_func(array_class, rb_alloc_array);
  
  // Instance methods
  rb_define_method(array_class, "initialize", RUBY_METHOD_FUNC(rb_array_initialize), -1);
  rb_define_method(array_class, "shape", RUBY_METHOD_FUNC(rb_array_shape), 0);
  rb_define_method(array_class, "dtype", RUBY_METHOD_FUNC(rb_array_dtype), 0);
  rb_define_method(array_class, "ndim", RUBY_METHOD_FUNC(rb_array_ndim), 0);
  rb_define_method(array_class, "size", RUBY_METHOD_FUNC(rb_array_size), 0);
  rb_define_method(array_class, "to_s", RUBY_METHOD_FUNC(rb_array_to_s), 0);
  rb_define_method(array_class, "+", RUBY_METHOD_FUNC(rb_array_add), 1);
  
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
  rb_define_const(module, "BFLOAT16", INT2NUM(static_cast<int>(mx::bfloat16.val())));
  rb_define_const(module, "COMPLEX64", INT2NUM(static_cast<int>(mx::complex64.val())));
} 