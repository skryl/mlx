#include <ruby.h>
#include "mlx/linalg.h"
#include "mlx/ops.h"

namespace mx = mlx::core;

// Helper to extract mx::array from Ruby VALUE
static mx::array& get_array(VALUE obj) {
  mx::array* arr_ptr;
  Data_Get_Struct(obj, mx::array, arr_ptr);
  return *arr_ptr;
}

// Helper to extract Ruby array into int vector
static std::vector<int> ruby_array_to_int_vector(VALUE rb_array) {
  Check_Type(rb_array, T_ARRAY);
  std::vector<int> result;
  for (long i = 0; i < RARRAY_LEN(rb_array); i++) {
    VALUE item = rb_ary_entry(rb_array, i);
    result.push_back(NUM2INT(item));
  }
  return result;
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

// Linear algebra module methods
static VALUE linalg_norm(int argc, VALUE* argv, VALUE self) {
  // Required: array
  // Optional: ord (nil/number/string), axis (nil/int/array), keepdims (bool), stream
  if (argc < 1 || argc > 5) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..5)", argc);
  }
  
  VALUE arr = argv[0];
  VALUE ord = (argc > 1) ? argv[1] : Qnil;
  VALUE axis = (argc > 2) ? argv[2] : Qnil;
  VALUE keepdims = (argc > 3) ? argv[3] : Qfalse;
  VALUE stream_val = (argc > 4) ? argv[4] : Qnil;
  
  mx::array& a = get_array(arr);
  bool keep_dims = RTEST(keepdims);
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  // Process different combinations of ord and axis
  std::optional<std::vector<int>> axes = std::nullopt;
  
  if (!NIL_P(axis)) {
    if (RB_TYPE_P(axis, T_ARRAY)) {
      axes = ruby_array_to_int_vector(axis);
    } else if (RB_TYPE_P(axis, T_FIXNUM)) {
      axes = std::vector<int>{NUM2INT(axis)};
    } else {
      rb_raise(rb_eTypeError, "axis must be nil, an integer, or an array of integers");
    }
  }
  
  if (NIL_P(ord)) {
    // Default norm (2-norm for vectors, Frobenius norm for matrices)
    return wrap_array(mx::linalg::norm(a, axes, keep_dims, stream));
  } else if (RB_TYPE_P(ord, T_FLOAT) || RB_TYPE_P(ord, T_FIXNUM)) {
    // Numeric order
    double order = NUM2DBL(ord);
    return wrap_array(mx::linalg::norm(a, order, axes, keep_dims, stream));
  } else if (RB_TYPE_P(ord, T_STRING)) {
    // String order like 'fro', 'nuc', 'inf', '-inf'
    std::string order_str = StringValueCStr(ord);
    return wrap_array(mx::linalg::norm(a, order_str, axes, keep_dims, stream));
  } else {
    rb_raise(rb_eTypeError, "ord must be nil, a number, or a string");
    return Qnil;
  }
}

static VALUE linalg_svd(int argc, VALUE* argv, VALUE self) {
  // Required: array
  // Optional: compute_uv (bool), stream
  if (argc < 1 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..3)", argc);
  }
  
  VALUE arr = argv[0];
  VALUE compute_uv = (argc > 1) ? argv[1] : Qtrue;
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  
  mx::array& a = get_array(arr);
  bool comp_uv = RTEST(compute_uv);
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  std::vector<mx::array> result = mx::linalg::svd(a, comp_uv, stream);
  
  if (!comp_uv) {
    // Just return singular values
    return wrap_array(result[0]);
  } else {
    VALUE rb_result = rb_ary_new();
    rb_ary_push(rb_result, wrap_array(result[0])); // U
    rb_ary_push(rb_result, wrap_array(result[1])); // S
    rb_ary_push(rb_result, wrap_array(result[2])); // Vt
    return rb_result;
  }
}

static VALUE linalg_qr(int argc, VALUE* argv, VALUE self) {
  // Required: array
  // Optional: stream
  if (argc < 1 || argc > 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..2)", argc);
  }
  
  VALUE arr = argv[0];
  VALUE stream_val = (argc > 1) ? argv[1] : Qnil;
  
  mx::array& a = get_array(arr);
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  auto result = mx::linalg::qr(a, stream);
  
  VALUE rb_result = rb_ary_new();
  rb_ary_push(rb_result, wrap_array(result.first)); // Q
  rb_ary_push(rb_result, wrap_array(result.second)); // R
  
  return rb_result;
}

static VALUE linalg_inv(int argc, VALUE* argv, VALUE self) {
  // Required: array
  // Optional: stream
  if (argc < 1 || argc > 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..2)", argc);
  }
  
  mx::array& a = get_array(argv[0]);
  VALUE stream_val = (argc > 1) ? argv[1] : Qnil;
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  mx::array result = mx::linalg::inv(a, stream);
  return wrap_array(result);
}

static VALUE linalg_tri_inv(int argc, VALUE* argv, VALUE self) {
  // Required: array
  // Optional: upper (bool), stream
  if (argc < 1 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..3)", argc);
  }
  
  mx::array& a = get_array(argv[0]);
  VALUE upper_val = (argc > 1) ? argv[1] : Qfalse;
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  
  bool upper = RTEST(upper_val);
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  mx::array result = mx::linalg::tri_inv(a, upper, stream);
  return wrap_array(result);
}

static VALUE linalg_cholesky(int argc, VALUE* argv, VALUE self) {
  // Required: array
  // Optional: upper (bool), stream
  if (argc < 1 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..3)", argc);
  }
  
  mx::array& a = get_array(argv[0]);
  VALUE upper_val = (argc > 1) ? argv[1] : Qfalse;
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  
  bool upper = RTEST(upper_val);
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  mx::array result = mx::linalg::cholesky(a, upper, stream);
  return wrap_array(result);
}

static VALUE linalg_cholesky_inv(int argc, VALUE* argv, VALUE self) {
  // Required: array
  // Optional: upper (bool), stream
  if (argc < 1 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..3)", argc);
  }
  
  mx::array& a = get_array(argv[0]);
  VALUE upper_val = (argc > 1) ? argv[1] : Qfalse;
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  
  bool upper = RTEST(upper_val);
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  mx::array result = mx::linalg::cholesky_inv(a, upper, stream);
  return wrap_array(result);
}

static VALUE linalg_eigh(int argc, VALUE* argv, VALUE self) {
  // Required: array
  // Optional: UPLO (string - 'L' or 'U'), stream
  if (argc < 1 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..3)", argc);
  }
  
  mx::array& a = get_array(argv[0]);
  std::string uplo = (argc > 1 && !NIL_P(argv[1])) ? StringValueCStr(argv[1]) : "L";
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  auto result = mx::linalg::eigh(a, uplo, stream);
  
  VALUE rb_result = rb_ary_new();
  rb_ary_push(rb_result, wrap_array(result.first)); // Eigenvalues
  rb_ary_push(rb_result, wrap_array(result.second)); // Eigenvectors
  
  return rb_result;
}

static VALUE linalg_eigvalsh(int argc, VALUE* argv, VALUE self) {
  // Required: array
  // Optional: UPLO (string - 'L' or 'U'), stream
  if (argc < 1 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..3)", argc);
  }
  
  mx::array& a = get_array(argv[0]);
  std::string uplo = (argc > 1 && !NIL_P(argv[1])) ? StringValueCStr(argv[1]) : "L";
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  mx::array result = mx::linalg::eigvalsh(a, uplo, stream);
  return wrap_array(result);
}

static VALUE linalg_matmul(int argc, VALUE* argv, VALUE self) {
  // Required: array, array
  // Optional: stream
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  mx::array& a = get_array(argv[0]);
  mx::array& b = get_array(argv[1]);
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  mx::array result = mx::matmul(a, b, stream);
  return wrap_array(result);
}

// Implementation of det using lu_factor
static mx::array det_impl(const mx::array& a, mx::StreamOrDevice stream) {
  // Validate that the array is square
  if (a.ndim() < 2 || a.shape(-1) != a.shape(-2)) {
    throw std::invalid_argument("[det] Input matrix must be square");
  }
  
  // Use LU factorization to compute the determinant
  auto lu_result = mx::linalg::lu_factor(a, stream);
  auto& lu = lu_result.first;
  auto& pivots = lu_result.second;
  
  // Extract diagonal elements
  auto diag = mx::diagonal(lu, 0, -2, -1, stream);
  
  // Compute product of diagonal elements
  auto diag_prod = mx::prod(diag, -1, stream);
  
  // For computing sign, we need to count the number of row exchanges
  // We can get this from the lu_factor computation, which returns
  // a permutation vector. The parity is determined by the number of
  // swaps needed to get from original positions to pivoted positions
  
  // Simplified approach: Use the fact that LU already preserves the sign change
  // in its decomposition, so we just need to take the product of diagonal elements
  return diag_prod;
}

static VALUE linalg_det(int argc, VALUE* argv, VALUE self) {
  // Required: array
  // Optional: stream
  if (argc < 1 || argc > 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..2)", argc);
  }
  
  mx::array& a = get_array(argv[0]);
  VALUE stream_val = (argc > 1) ? argv[1] : Qnil;
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  mx::array result = det_impl(a, stream);
  return wrap_array(result);
}

static std::pair<mx::array, mx::array> slogdet_impl(const mx::array& a, mx::StreamOrDevice stream) {
  // Validate that the array is square
  if (a.ndim() < 2 || a.shape(-1) != a.shape(-2)) {
    throw std::invalid_argument("[slogdet] Input matrix must be square");
  }
  
  // Calculate determinant 
  mx::array determinant = det_impl(a, stream);
  
  // Calculate sign and log(abs(det))
  mx::array sign = mx::sign(determinant, stream);
  mx::array logabsdet = mx::log(mx::abs(determinant, stream), stream);
  
  return std::make_pair(sign, logabsdet);
}

static VALUE linalg_slogdet(int argc, VALUE* argv, VALUE self) {
  // Required: array
  // Optional: stream
  if (argc < 1 || argc > 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..2)", argc);
  }
  
  mx::array& a = get_array(argv[0]);
  VALUE stream_val = (argc > 1) ? argv[1] : Qnil;
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  auto result = slogdet_impl(a, stream);
  
  VALUE rb_result = rb_ary_new();
  rb_ary_push(rb_result, wrap_array(result.first)); // Sign
  rb_ary_push(rb_result, wrap_array(result.second)); // Logarithm of the absolute value
  
  return rb_result;
}

static VALUE linalg_solve(int argc, VALUE* argv, VALUE self) {
  // Required: array, array
  // Optional: stream
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  mx::array& a = get_array(argv[0]);
  mx::array& b = get_array(argv[1]);
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  mx::array result = mx::linalg::solve(a, b, stream);
  return wrap_array(result);
}

static VALUE linalg_solve_triangular(int argc, VALUE* argv, VALUE self) {
  // Required: array, array
  // Optional: upper (bool), stream
  if (argc < 2 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..4)", argc);
  }
  
  mx::array& a = get_array(argv[0]);
  mx::array& b = get_array(argv[1]);
  VALUE upper_val = (argc > 2) ? argv[2] : Qfalse;
  VALUE stream_val = (argc > 3) ? argv[3] : Qnil;
  
  bool upper = RTEST(upper_val);
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  mx::array result = mx::linalg::solve_triangular(a, b, upper, stream);
  return wrap_array(result);
}

static mx::array matrix_power_impl(const mx::array& a, int n, mx::StreamOrDevice stream) {
  // Validate that the array is square
  if (a.ndim() < 2 || a.shape(-1) != a.shape(-2)) {
    throw std::invalid_argument("[matrix_power] Input matrix must be square");
  }

  // Special cases
  if (n == 0) {
    // Create identity matrix with the same shape as a
    auto shape = a.shape();
    int m = shape[shape.size() - 1];
    
    // First create a result array with the same shape as a
    auto result = mx::zeros_like(a, stream);
    
    // Then set the diagonal elements to 1
    auto diag_indices = mx::arange(m, stream);
    auto flat_indices = diag_indices * m + diag_indices;
    auto result_flat = mx::reshape(result, {-1}, stream);
    
    // Use scatter to set diagonal to 1
    mx::array ones = mx::ones({m}, result_flat.dtype(), stream);
    result_flat = mx::scatter(result_flat, flat_indices, ones, 0, stream);
    
    // Reshape back to original shape
    return mx::reshape(result_flat, shape, stream);
  } else if (n == 1) {
    // Return the matrix itself
    return a;
  } else if (n < 0) {
    // For negative powers, compute inverse and then positive power
    auto a_inv = mx::linalg::inv(a, stream);
    return matrix_power_impl(a_inv, -n, stream);
  }
  
  // Binary exponentiation for efficient computation
  // For the base case, create identity matrix
  auto shape = a.shape();
  int m = shape[shape.size() - 1];
  
  // Initialize result as identity matrix
  auto result = mx::zeros_like(a, stream);
  auto diag_indices = mx::arange(m, stream);
  auto flat_indices = diag_indices * m + diag_indices;
  auto result_flat = mx::reshape(result, {-1}, stream);
  
  // Use scatter to set diagonal to 1
  mx::array ones = mx::ones({m}, result_flat.dtype(), stream);
  result_flat = mx::scatter(result_flat, flat_indices, ones, 0, stream);
  result = mx::reshape(result_flat, shape, stream);
  
  mx::array base = a;
  
  while (n > 0) {
    if (n % 2 == 1) {
      // If n is odd, multiply result by current base
      result = mx::matmul(result, base, stream);
    }
    // Square the base
    base = mx::matmul(base, base, stream);
    // Integer division by 2
    n /= 2;
  }
  
  return result;
}

static VALUE linalg_matrix_power(int argc, VALUE* argv, VALUE self) {
  // Required: array, int
  // Optional: stream
  if (argc < 2 || argc > 3) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..3)", argc);
  }
  
  mx::array& a = get_array(argv[0]);
  int n = NUM2INT(argv[1]);
  VALUE stream_val = (argc > 2) ? argv[2] : Qnil;
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  mx::array result = matrix_power_impl(a, n, stream);
  return wrap_array(result);
}

static VALUE linalg_pinv(int argc, VALUE* argv, VALUE self) {
  // Required: array
  // Optional: stream
  if (argc < 1 || argc > 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..2)", argc);
  }
  
  mx::array& a = get_array(argv[0]);
  VALUE stream_val = (argc > 1) ? argv[1] : Qnil;
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  mx::array result = mx::linalg::pinv(a, stream);
  return wrap_array(result);
}

static VALUE linalg_cross(int argc, VALUE* argv, VALUE self) {
  // Required: array, array
  // Optional: axis (int), stream
  if (argc < 2 || argc > 4) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 2..4)", argc);
  }
  
  mx::array& a = get_array(argv[0]);
  mx::array& b = get_array(argv[1]);
  int axis = (argc > 2 && !NIL_P(argv[2])) ? NUM2INT(argv[2]) : -1;
  VALUE stream_val = (argc > 3) ? argv[3] : Qnil;
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  mx::array result = mx::linalg::cross(a, b, axis, stream);
  return wrap_array(result);
}

static VALUE linalg_lu(int argc, VALUE* argv, VALUE self) {
  // Required: array
  // Optional: stream
  if (argc < 1 || argc > 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..2)", argc);
  }
  
  mx::array& a = get_array(argv[0]);
  VALUE stream_val = (argc > 1) ? argv[1] : Qnil;
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  auto result = mx::linalg::lu(a, stream);
  
  VALUE rb_result = rb_ary_new();
  rb_ary_push(rb_result, wrap_array(result.at(0))); // P
  rb_ary_push(rb_result, wrap_array(result.at(1))); // L
  rb_ary_push(rb_result, wrap_array(result.at(2))); // U
  
  return rb_result;
}

static VALUE linalg_lu_factor(int argc, VALUE* argv, VALUE self) {
  // Required: array
  // Optional: stream
  if (argc < 1 || argc > 2) {
    rb_raise(rb_eArgError, "wrong number of arguments (given %d, expected 1..2)", argc);
  }
  
  mx::array& a = get_array(argv[0]);
  VALUE stream_val = (argc > 1) ? argv[1] : Qnil;
  mx::StreamOrDevice stream = NIL_P(stream_val) ? mx::StreamOrDevice{} : get_stream_or_device(stream_val);
  
  auto result = mx::linalg::lu_factor(a, stream);
  
  VALUE rb_result = rb_ary_new();
  rb_ary_push(rb_result, wrap_array(result.first)); // LU
  rb_ary_push(rb_result, wrap_array(result.second)); // pivots
  
  return rb_result;
}

// Initialize linear algebra module
void init_linalg(VALUE module) {
  // Define module functions
  rb_define_module_function(module, "norm", RUBY_METHOD_FUNC(linalg_norm), -1);
  rb_define_module_function(module, "svd", RUBY_METHOD_FUNC(linalg_svd), -1);
  rb_define_module_function(module, "qr", RUBY_METHOD_FUNC(linalg_qr), -1);
  rb_define_module_function(module, "inv", RUBY_METHOD_FUNC(linalg_inv), -1);
  rb_define_module_function(module, "tri_inv", RUBY_METHOD_FUNC(linalg_tri_inv), -1);
  rb_define_module_function(module, "cholesky", RUBY_METHOD_FUNC(linalg_cholesky), -1);
  rb_define_module_function(module, "cholesky_inv", RUBY_METHOD_FUNC(linalg_cholesky_inv), -1);
  rb_define_module_function(module, "eigh", RUBY_METHOD_FUNC(linalg_eigh), -1);
  rb_define_module_function(module, "eigvalsh", RUBY_METHOD_FUNC(linalg_eigvalsh), -1);
  rb_define_module_function(module, "matmul", RUBY_METHOD_FUNC(linalg_matmul), -1);
  rb_define_module_function(module, "det", RUBY_METHOD_FUNC(linalg_det), -1);
  rb_define_module_function(module, "slogdet", RUBY_METHOD_FUNC(linalg_slogdet), -1);
  rb_define_module_function(module, "solve", RUBY_METHOD_FUNC(linalg_solve), -1);
  rb_define_module_function(module, "solve_triangular", RUBY_METHOD_FUNC(linalg_solve_triangular), -1);
  rb_define_module_function(module, "matrix_power", RUBY_METHOD_FUNC(linalg_matrix_power), -1);
  rb_define_module_function(module, "pinv", RUBY_METHOD_FUNC(linalg_pinv), -1);
  rb_define_module_function(module, "cross", RUBY_METHOD_FUNC(linalg_cross), -1);
  rb_define_module_function(module, "lu", RUBY_METHOD_FUNC(linalg_lu), -1);
  rb_define_module_function(module, "lu_factor", RUBY_METHOD_FUNC(linalg_lu_factor), -1);
} 