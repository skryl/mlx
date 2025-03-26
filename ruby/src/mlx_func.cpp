#include <ruby.h>
#include "mlx/array.h"
#include "mlx/ops.h"
#include "mlx_func.h"

namespace mx = mlx::core;

// Convert Ruby value to ScalarOrArray
ScalarOrArray value_to_scalar_or_array(VALUE value) {
  if (rb_obj_is_kind_of(value, rb_cNumeric)) {
    return ScalarOrArray(NUM2DBL(value));
  } else {
    // Get the underlying MLX array
    mx::array* arr_ptr;
    Data_Get_Struct(value, mx::array, arr_ptr);
    return ScalarOrArray(*arr_ptr);
  }
}

// MLX functions
static VALUE mlx_eval(VALUE self, VALUE obj) {
  // Extract the MLX array
  mx::array* arr_ptr;
  Data_Get_Struct(obj, mx::array, arr_ptr);
  
  // Evaluate the array
  mx::array result = mx::eval(*arr_ptr);
  
  // Wrap and return
  return Data_Wrap_Struct(rb_class_of(obj), 0, nullptr, new mx::array(result));
}

static VALUE mlx_eval_batch(VALUE self, VALUE array_list) {
  Check_Type(array_list, T_ARRAY);
  
  // Convert Ruby array to vector of MLX arrays
  std::vector<mx::array> arrays;
  for (long i = 0; i < RARRAY_LEN(array_list); i++) {
    VALUE item = rb_ary_entry(array_list, i);
    mx::array* arr_ptr;
    Data_Get_Struct(item, mx::array, arr_ptr);
    arrays.push_back(*arr_ptr);
  }
  
  // Evaluate the batch
  std::vector<mx::array> results = mx::eval(arrays);
  
  // Convert back to Ruby array
  VALUE result_list = rb_ary_new();
  for (size_t i = 0; i < results.size(); i++) {
    VALUE rb_arr = Data_Wrap_Struct(rb_class_of(rb_ary_entry(array_list, i)), 0, nullptr, 
                                     new mx::array(results[i]));
    rb_ary_push(result_list, rb_arr);
  }
  
  return result_list;
}

static VALUE mlx_value_and_grad(VALUE self, VALUE func, VALUE args, VALUE has_aux) {
  // This is a complex function that requires a Ruby closure
  // For simplicity, this is a placeholder - full implementation would need
  // to handle Ruby callbacks properly
  rb_raise(rb_eNotImpError, "value_and_grad not fully implemented yet");
  return Qnil;
}

static VALUE mlx_grad(VALUE self, VALUE func, VALUE args, VALUE has_aux) {
  // Similar to value_and_grad, this needs careful handling of Ruby callbacks
  rb_raise(rb_eNotImpError, "grad not fully implemented yet");
  return Qnil;
}

// Initialize the mlx function module
void init_mlx_func(VALUE module) {
  // Define module functions
  rb_define_module_function(module, "eval", RUBY_METHOD_FUNC(mlx_eval), 1);
  rb_define_module_function(module, "eval_batch", RUBY_METHOD_FUNC(mlx_eval_batch), 1);
  rb_define_module_function(module, "value_and_grad", RUBY_METHOD_FUNC(mlx_value_and_grad), 3);
  rb_define_module_function(module, "grad", RUBY_METHOD_FUNC(mlx_grad), 3);
}
 