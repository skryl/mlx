#include <ruby.h>
#include <numeric>
#include "mlx/ops.h"
#include "indexing.h"

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

// Helper to convert Ruby value to mx::array
static mx::array value_to_array(VALUE val, mx::Dtype dtype = mx::float32) {
  if (rb_obj_is_kind_of(val, rb_path2class("MLX::Core::Array"))) {
    return get_array(val);
  } else if (RB_TYPE_P(val, T_FLOAT) || RB_TYPE_P(val, T_FIXNUM)) {
    return mx::array(NUM2DBL(val), dtype);
  } else if (RB_TYPE_P(val, T_TRUE) || RB_TYPE_P(val, T_FALSE)) {
    return mx::array(RTEST(val), mx::bool_);
  } else if (RB_TYPE_P(val, T_ARRAY)) {
    // Convert Ruby array to a C array
    int size = RARRAY_LEN(val);
    std::vector<double> c_array(size);
    
    for (int i = 0; i < size; i++) {
      VALUE item = rb_ary_entry(val, i);
      if (RB_TYPE_P(item, T_FLOAT) || RB_TYPE_P(item, T_FIXNUM)) {
        c_array[i] = NUM2DBL(item);
      } else {
        rb_raise(rb_eTypeError, "Array elements must be numeric");
      }
    }
    
    // Create MLX array directly from C array
    return mx::array(c_array.data(), {size}, dtype);
  } else {
    rb_raise(rb_eTypeError, "Cannot convert value to MLX array");
    return mx::array(0); // Never reached, but need a valid construction
  }
}

// Utilities for advanced indexing
static bool is_none_value(VALUE obj) {
  return NIL_P(obj);
}

static int get_slice_int(VALUE obj, int default_val) {
  if (!NIL_P(obj)) {
    if (!RB_TYPE_P(obj, T_FIXNUM)) {
      rb_raise(rb_eArgError, "Slice indices must be integers or nil.");
    }
    return NUM2INT(obj);
  }
  return default_val;
}

static void get_slice_params(
    mx::ShapeElem& starts,
    mx::ShapeElem& ends,
    mx::ShapeElem& strides,
    VALUE rb_start, VALUE rb_end, VALUE rb_step,
    int axis_size) {
  
  strides = get_slice_int(rb_step, 1);
  starts = get_slice_int(rb_start, strides < 0 ? axis_size - 1 : 0);
  ends = get_slice_int(rb_end, strides < 0 ? -axis_size - 1 : axis_size);
}

static mx::array get_int_index(VALUE idx, int axis_size) {
  int idx_ = NUM2INT(idx);
  idx_ = (idx_ < 0) ? idx_ + axis_size : idx_;
  return mx::array(idx_, mx::uint32);
}

// Indexing operations
static mx::array mlx_get_item_slice(
    const mx::array& src, 
    VALUE rb_start, VALUE rb_end, VALUE rb_step) {
  
  // Check input and raise error if 0 dim for parity with numpy
  if (src.ndim() == 0) {
    rb_raise(rb_eArgError, "too many indices for array: array is 0-dimensional");
  }

  // Return a copy of the array if none slice is requested
  if (NIL_P(rb_start) && NIL_P(rb_end) && NIL_P(rb_step)) {
    return src;
  }

  mx::Shape starts(src.ndim(), 0);
  auto ends = src.shape();
  mx::Shape strides(src.ndim(), 1);

  // Check and update slice params
  get_slice_params(starts[0], ends[0], strides[0], rb_start, rb_end, rb_step, ends[0]);
  return mx::slice(src, starts, ends, strides);
}

static mx::array mlx_get_item_array(const mx::array& src, const mx::array& indices) {
  // Check input and raise error if 0 dim for parity with numpy
  if (src.ndim() == 0) {
    rb_raise(rb_eArgError, "too many indices for array: array is 0-dimensional");
  }

  if (indices.dtype() == mx::bool_) {
    rb_raise(rb_eArgError, "boolean indices are not yet supported");
  }

  // If only one input array is mentioned, we set axis=0 in take
  return mx::take(src, indices, 0);
}

static mx::array mlx_get_item_int(const mx::array& src, VALUE idx) {
  // Check input and raise error if 0 dim for parity with numpy
  if (src.ndim() == 0) {
    rb_raise(rb_eArgError, "too many indices for array: array is 0-dimensional");
  }

  // If only one input idx is mentioned, we set axis=0 in take
  return mx::take(src, get_int_index(idx, src.shape(0)), 0);
}

static mx::array mlx_gather_nd(
    mx::array src,
    std::vector<VALUE>& indices,
    bool gather_first,
    int& max_dims) {
  
  max_dims = 0;
  std::vector<mx::array> gather_indices;
  std::vector<bool> is_slice(indices.size(), false);
  int num_slices = 0;
  
  // Gather all the arrays
  for (int i = 0; i < indices.size(); i++) {
    VALUE idx = indices[i];

    if (rb_obj_is_kind_of(idx, rb_cRange)) {
      // Convert Ruby Range to start, end
      VALUE range_begin = rb_funcall(idx, rb_intern("begin"), 0);
      VALUE range_end = rb_funcall(idx, rb_intern("end"), 0);
      VALUE exclude_end = rb_funcall(idx, rb_intern("exclude_end?"), 0);
      
      int start = NUM2INT(range_begin);
      int end = NUM2INT(range_end);
      if (RTEST(exclude_end)) {
        end = end - 1;
      }
      
      // Handle negative indices
      start = (start < 0) ? start + src.shape(i) : start;
      end = (end < 0) ? end + src.shape(i) : end;
      int stride = 1;

      gather_indices.push_back(mx::arange(start, end + 1, stride, mx::uint32));
      num_slices++;
      is_slice[i] = true;
    } else if (RB_TYPE_P(idx, T_FIXNUM)) {
      gather_indices.push_back(get_int_index(idx, src.shape(i)));
    } else if (rb_obj_is_kind_of(idx, rb_path2class("MLX::Core::Array"))) {
      mx::array& arr = get_array(idx);
      max_dims = std::max(static_cast<int>(arr.ndim()), max_dims);
      gather_indices.push_back(arr);
    }
  }

  // Reshape them so that the int/array indices are first
  if (gather_first) {
    int slice_index = 0;
    for (int i = 0; i < gather_indices.size(); i++) {
      if (is_slice[i]) {
        mx::Shape index_shape(max_dims + num_slices, 1);
        index_shape[max_dims + slice_index] = gather_indices[i].shape(0);
        gather_indices[i] = mx::reshape(gather_indices[i], std::move(index_shape));
        slice_index++;
      } else {
        auto index_shape = gather_indices[i].shape();
        index_shape.insert(index_shape.end(), num_slices, 1);
        gather_indices[i] = mx::reshape(gather_indices[i], std::move(index_shape));
      }
    }
  } else {
    // Reshape them so that the int/array indices are last
    for (int i = 0; i < gather_indices.size(); i++) {
      if (i < num_slices) {
        mx::Shape index_shape(max_dims + num_slices, 1);
        index_shape[i] = gather_indices[i].shape(0);
        gather_indices[i] = mx::reshape(gather_indices[i], std::move(index_shape));
      }
    }
  }

  // Do the gather
  std::vector<int> axes(indices.size());
  std::iota(axes.begin(), axes.end(), 0);
  auto slice_sizes = src.shape();
  std::fill(slice_sizes.begin(), slice_sizes.begin() + indices.size(), 1);
  src = mx::gather(src, gather_indices, axes, slice_sizes);

  // Squeeze the array index dims
  for (auto& ax : axes) {
    ax += max_dims + num_slices;
  }
  return mx::squeeze(src, axes);
}

// Helper for scatter operations - similar to mlx_compute_scatter_args in Python
std::tuple<std::vector<mx::array>, mx::array, std::vector<int>> 
compute_scatter_args(const mx::array& src, VALUE obj, VALUE v) {
  mx::array update = value_to_array(v, src.dtype());
  
  // Handle integer index case
  if (RB_TYPE_P(obj, T_FIXNUM)) {
    if (src.ndim() == 0) {
      rb_raise(rb_eArgError, "too many indices for array: array is 0-dimensional");
    }
    
    // Remove any leading singleton dimensions from the update
    // and then broadcast update to shape of src[0, ...]
    int s = 0;
    for (; s < update.ndim() && update.shape(s) == 1; s++);
    auto up_shape = mx::Shape(update.shape().begin() + s, update.shape().end());
    auto shape = src.shape();
    shape[0] = 1;
    
    return {
      {get_int_index(obj, src.shape(0))},
      mx::broadcast_to(mx::reshape(update, up_shape), shape),
      {0}
    };
  }
  
  // Handle array index case
  if (rb_obj_is_kind_of(obj, rb_path2class("MLX::Core::Array"))) {
    mx::array& indices = get_array(obj);
    if (src.ndim() == 0) {
      rb_raise(rb_eArgError, "too many indices for array: array is 0-dimensional");
    }

    // Remove leading singletons
    int s = 0;
    for (; s < update.ndim() && update.shape(s) == 1; s++);
    auto up_shape = mx::Shape(update.shape().begin() + s, update.shape().end());
    update = mx::reshape(update, up_shape);
    
    // The update shape must broadcast with indices.shape + src.shape[1:]
    auto broadcast_shape = indices.shape();
    broadcast_shape.insert(broadcast_shape.end(), src.shape().begin() + 1, src.shape().end());
    update = mx::broadcast_to(update, broadcast_shape);
    
    // Reshape with a 1 in indices.ndim position
    broadcast_shape.insert(broadcast_shape.begin() + indices.ndim(), 1);
    update = mx::reshape(update, broadcast_shape);
    
    return {{indices}, update, {0}};
  }
  
  // Handle Range (slice) case
  if (rb_obj_is_kind_of(obj, rb_cRange)) {
    if (src.ndim() == 0) {
      rb_raise(rb_eArgError, "too many indices for array: array is 0-dimensional");
    }
    
    VALUE range_begin = rb_funcall(obj, rb_intern("begin"), 0);
    VALUE range_end = rb_funcall(obj, rb_intern("end"), 0);
    VALUE exclude_end = rb_funcall(obj, rb_intern("exclude_end?"), 0);
    
    int start = range_begin == Qnil ? 0 : NUM2INT(range_begin);
    int end = range_end == Qnil ? src.shape(0) : NUM2INT(range_end);
    if (RTEST(exclude_end)) {
      end = end - 1;
    }
    
    // Handle negative indices
    start = (start < 0) ? start + src.shape(0) : start;
    end = (end < 0) ? end + src.shape(0) : end;
    
    // If simple stride
    int stride = 1; // Ruby ranges don't have explicit stride
    
    if (stride == 1) {
      // Remove leading singletons
      int s = 0;
      for (; s < update.ndim() && update.shape(s) == 1; s++);
      auto up_shape = mx::Shape(update.shape().begin() + s, update.shape().end());
      update = mx::reshape(update, up_shape);
      
      // Build array to mark start of slice
      auto idx = mx::array({start}, {1}, mx::uint32);
      
      // Get slice size
      int slice_size = (end - start + 1);
      
      // Broadcast update to slice size
      mx::Shape up_shape_broadcast = {1, slice_size};
      up_shape_broadcast.insert(
          up_shape_broadcast.end(), src.shape().begin() + 1, src.shape().end());
      
      update = mx::broadcast_to(update, up_shape_broadcast);
      
      return {{idx}, update, {0}};
    }
    
    // For non-simple stride, handle as array indexing with arange
    auto indices = mx::arange(start, end + 1, stride, mx::uint32);
    
    // Remove leading singletons
    int s = 0;
    for (; s < update.ndim() && update.shape(s) == 1; s++);
    auto up_shape = mx::Shape(update.shape().begin() + s, update.shape().end());
    update = mx::reshape(update, up_shape);
    
    // Broadcast update to match indices shape + remaining src shape
    auto broadcast_shape = indices.shape();
    broadcast_shape.insert(broadcast_shape.end(), src.shape().begin() + 1, src.shape().end());
    update = mx::broadcast_to(update, broadcast_shape);
    
    // Add dimension for the index
    broadcast_shape.insert(broadcast_shape.begin() + indices.ndim(), 1);
    update = mx::reshape(update, broadcast_shape);
    
    return {{indices}, update, {0}};
  }
  
  // Handle array of indices case
  if (RB_TYPE_P(obj, T_ARRAY)) {
    std::vector<mx::array> index_arrays;
    std::vector<int> axes;
    int axis = 0;
    
    for (long i = 0; i < RARRAY_LEN(obj); i++) {
      VALUE idx = rb_ary_entry(obj, i);
      
      if (RB_TYPE_P(idx, T_FIXNUM)) {
        index_arrays.push_back(get_int_index(idx, src.shape(axis)));
        axes.push_back(axis);
        axis++;
      } else if (rb_obj_is_kind_of(idx, rb_path2class("MLX::Core::Array"))) {
        index_arrays.push_back(get_array(idx));
        axes.push_back(axis);
        axis++;
      } else if (rb_obj_is_kind_of(idx, rb_cRange)) {
        // Handle range similar to above, but for multi-dimensional case
        VALUE range_begin = rb_funcall(idx, rb_intern("begin"), 0);
        VALUE range_end = rb_funcall(idx, rb_intern("end"), 0);
        VALUE exclude_end = rb_funcall(idx, rb_intern("exclude_end?"), 0);
        
        int start = range_begin == Qnil ? 0 : NUM2INT(range_begin);
        int end = range_end == Qnil ? src.shape(axis) : NUM2INT(range_end);
        if (RTEST(exclude_end)) {
          end = end - 1;
        }
        
        // Handle negative indices
        start = (start < 0) ? start + src.shape(axis) : start;
        end = (end < 0) ? end + src.shape(axis) : end;
        
        index_arrays.push_back(mx::arange(start, end + 1, 1, mx::uint32));
        axes.push_back(axis);
        axis++;
      }
    }
    
    if (index_arrays.empty()) {
      return {{}, update, {}};
    }
    
    // Broadcast all index arrays together
    index_arrays = mx::broadcast_arrays(index_arrays);
    
    // Create shape for broadcast
    auto up_shape = index_arrays[0].shape();
    // Add dimensions for remaining axes of src
    up_shape.insert(up_shape.end(), src.shape().begin() + axis, src.shape().end());
    
    // Broadcast update to match
    update = mx::broadcast_to(update, up_shape);
    
    return {index_arrays, update, axes};
  }
  
  // Default case: no indices, just broadcast update to src shape
  update = mx::broadcast_to(update, src.shape());
  return {{}, update, {}};
}

// Implementation of scatter operations
void ruby_set_item(mx::array& src, VALUE obj, VALUE v) {
  auto [indices, updates, axes] = compute_scatter_args(src, obj, v);
  if (indices.empty()) {
    src.overwrite_descriptor(updates);
  } else {
    auto result = mx::scatter(src, indices, updates, axes);
    src.overwrite_descriptor(result);
  }
}

mx::array ruby_add_item(const mx::array& src, VALUE obj, VALUE v) {
  auto [indices, updates, axes] = compute_scatter_args(src, obj, v);
  if (indices.empty()) {
    return src + updates;
  } else {
    return mx::scatter_add(src, indices, updates, axes);
  }
}

mx::array ruby_subtract_item(const mx::array& src, VALUE obj, VALUE v) {
  auto [indices, updates, axes] = compute_scatter_args(src, obj, v);
  if (indices.empty()) {
    return src - updates;
  } else {
    return mx::scatter_add(src, indices, -updates, axes);
  }
}

mx::array ruby_multiply_item(const mx::array& src, VALUE obj, VALUE v) {
  auto [indices, updates, axes] = compute_scatter_args(src, obj, v);
  if (indices.empty()) {
    return src * updates;
  } else {
    return mx::scatter_prod(src, indices, updates, axes);
  }
}

mx::array ruby_divide_item(const mx::array& src, VALUE obj, VALUE v) {
  auto [indices, updates, axes] = compute_scatter_args(src, obj, v);
  if (indices.empty()) {
    return src / updates;
  } else {
    return mx::scatter_prod(src, indices, mx::reciprocal(updates), axes);
  }
}

mx::array ruby_maximum_item(const mx::array& src, VALUE obj, VALUE v) {
  auto [indices, updates, axes] = compute_scatter_args(src, obj, v);
  if (indices.empty()) {
    return mx::maximum(src, updates);
  } else {
    return mx::scatter_max(src, indices, updates, axes);
  }
}

mx::array ruby_minimum_item(const mx::array& src, VALUE obj, VALUE v) {
  auto [indices, updates, axes] = compute_scatter_args(src, obj, v);
  if (indices.empty()) {
    return mx::minimum(src, updates);
  } else {
    return mx::scatter_min(src, indices, updates, axes);
  }
}

// Indexing module methods
static VALUE indexing_take(VALUE self, VALUE arr, VALUE indices, VALUE axis) {
  mx::array& a = get_array(arr);
  mx::array& idx = get_array(indices);
  int ax = NIL_P(axis) ? 0 : NUM2INT(axis);
  
  mx::array result = mx::take(a, idx, ax);
  return wrap_array(result);
}

static VALUE indexing_take_along_axis(VALUE self, VALUE arr, VALUE indices, VALUE axis) {
  mx::array& a = get_array(arr);
  mx::array& idx = get_array(indices);
  int ax = NIL_P(axis) ? 0 : NUM2INT(axis);
  
  mx::array result = mx::take_along_axis(a, idx, ax);
  return wrap_array(result);
}

static VALUE indexing_slice(VALUE self, VALUE arr, VALUE start, VALUE stop, VALUE step) {
  mx::array& a = get_array(arr);
  mx::array result = mlx_get_item_slice(a, start, stop, step);
  return wrap_array(result);
}

static VALUE indexing_index(VALUE self, VALUE arr, VALUE indices) {
  mx::array& a = get_array(arr);
  
  // Check if indices is an array of arrays
  if (RB_TYPE_P(indices, T_ARRAY)) {
    std::vector<mx::array> idx_arrays;
    for (long i = 0; i < RARRAY_LEN(indices); i++) {
      VALUE item = rb_ary_entry(indices, i);
      if (rb_obj_is_kind_of(item, rb_path2class("MLX::Core::Array"))) {
        mx::array& idx = get_array(item);
        idx_arrays.push_back(idx);
      } else {
        rb_raise(rb_eTypeError, "Indices must be MLX arrays");
      }
    }
    
    mx::array result = mx::gather(a, idx_arrays, std::vector<int>{0}, a.shape());
    return wrap_array(result);
  } else if (rb_obj_is_kind_of(indices, rb_path2class("MLX::Core::Array"))) {
    // Single index array
    mx::array& idx = get_array(indices);
    std::vector<mx::array> idx_arrays = {idx};
    
    mx::array result = mx::gather(a, idx_arrays, std::vector<int>{0}, a.shape());
    return wrap_array(result);
  } else {
    rb_raise(rb_eTypeError, "Indices must be an MLX array or an array of MLX arrays");
    return Qnil;
  }
}

static VALUE indexing_dynamic_slice(VALUE self, VALUE arr, VALUE start_indices, VALUE slice_sizes) {
  mx::array& a = get_array(arr);
  
  // Check if start_indices is an array of arrays
  if (RB_TYPE_P(start_indices, T_ARRAY)) {
    std::vector<mx::array> start_idx_arrays;
    for (long i = 0; i < RARRAY_LEN(start_indices); i++) {
      VALUE item = rb_ary_entry(start_indices, i);
      if (rb_obj_is_kind_of(item, rb_path2class("MLX::Core::Array"))) {
        mx::array& idx = get_array(item);
        start_idx_arrays.push_back(idx);
      } else {
        rb_raise(rb_eTypeError, "Start indices must be MLX arrays");
      }
    }
    
    // Check slice_sizes
    if (RB_TYPE_P(slice_sizes, T_ARRAY)) {
      std::vector<int> slice_sizes_vec;
      for (long i = 0; i < RARRAY_LEN(slice_sizes); i++) {
        VALUE item = rb_ary_entry(slice_sizes, i);
        slice_sizes_vec.push_back(NUM2INT(item));
      }
      
      // Use mx::gather instead of dynamic_slice
      std::vector<int> axes(start_idx_arrays.size());
      std::iota(axes.begin(), axes.end(), 0);
      mx::array result = mx::gather(a, start_idx_arrays, axes, slice_sizes_vec);
      return wrap_array(result);
    } else {
      rb_raise(rb_eArgError, "slice_sizes must be an array");
      return Qnil;
    }
  } else {
    rb_raise(rb_eArgError, "start_indices must be an array");
    return Qnil;
  }
}

static VALUE indexing_scatter(VALUE self, VALUE arr, VALUE indices, VALUE updates, VALUE axis) {
  mx::array& a = get_array(arr);
  mx::array& idx = get_array(indices);
  mx::array& updates_arr = get_array(updates);
  int ax = NIL_P(axis) ? 0 : NUM2INT(axis);
  
  mx::array result = mx::scatter(a, idx, updates_arr, ax);
  return wrap_array(result);
}

static VALUE indexing_scatter_add(VALUE self, VALUE arr, VALUE indices, VALUE updates, VALUE axis) {
  mx::array& a = get_array(arr);
  mx::array& idx = get_array(indices);
  mx::array& updates_arr = get_array(updates);
  int ax = NIL_P(axis) ? 0 : NUM2INT(axis);
  
  mx::array result = mx::scatter_add(a, idx, updates_arr, ax);
  return wrap_array(result);
}

static VALUE indexing_scatter_prod(VALUE self, VALUE arr, VALUE indices, VALUE updates, VALUE axis) {
  mx::array& a = get_array(arr);
  mx::array& idx = get_array(indices);
  mx::array& updates_arr = get_array(updates);
  int ax = NIL_P(axis) ? 0 : NUM2INT(axis);
  
  mx::array result = mx::scatter_prod(a, idx, updates_arr, ax);
  return wrap_array(result);
}

static VALUE indexing_scatter_max(VALUE self, VALUE arr, VALUE indices, VALUE updates, VALUE axis) {
  mx::array& a = get_array(arr);
  mx::array& idx = get_array(indices);
  mx::array& updates_arr = get_array(updates);
  int ax = NIL_P(axis) ? 0 : NUM2INT(axis);
  
  mx::array result = mx::scatter_max(a, idx, updates_arr, ax);
  return wrap_array(result);
}

static VALUE indexing_scatter_min(VALUE self, VALUE arr, VALUE indices, VALUE updates, VALUE axis) {
  mx::array& a = get_array(arr);
  mx::array& idx = get_array(indices);
  mx::array& updates_arr = get_array(updates);
  int ax = NIL_P(axis) ? 0 : NUM2INT(axis);
  
  mx::array result = mx::scatter_min(a, idx, updates_arr, ax);
  return wrap_array(result);
}

static VALUE indexing_gather(VALUE self, VALUE arr, VALUE indices, VALUE axes, VALUE slice_sizes) {
  mx::array& a = get_array(arr);
  
  // Check indices
  std::vector<mx::array> idx_arrays;
  if (RB_TYPE_P(indices, T_ARRAY)) {
    for (long i = 0; i < RARRAY_LEN(indices); i++) {
      VALUE item = rb_ary_entry(indices, i);
      if (rb_obj_is_kind_of(item, rb_path2class("MLX::Core::Array"))) {
        mx::array& idx = get_array(item);
        idx_arrays.push_back(idx);
      } else {
        rb_raise(rb_eTypeError, "Indices must be MLX arrays");
      }
    }
  } else {
    rb_raise(rb_eArgError, "indices must be an array of MLX arrays");
    return Qnil;
  }
  
  // Check axes
  std::vector<int> axes_vec;
  if (RB_TYPE_P(axes, T_ARRAY)) {
    for (long i = 0; i < RARRAY_LEN(axes); i++) {
      VALUE item = rb_ary_entry(axes, i);
      axes_vec.push_back(NUM2INT(item));
    }
  } else {
    rb_raise(rb_eArgError, "axes must be an array of integers");
    return Qnil;
  }
  
  // Check slice_sizes
  std::vector<int> slice_sizes_vec;
  if (RB_TYPE_P(slice_sizes, T_ARRAY)) {
    for (long i = 0; i < RARRAY_LEN(slice_sizes); i++) {
      VALUE item = rb_ary_entry(slice_sizes, i);
      slice_sizes_vec.push_back(NUM2INT(item));
    }
  } else {
    rb_raise(rb_eArgError, "slice_sizes must be an array of integers");
    return Qnil;
  }
  
  mx::array result = mx::gather(a, idx_arrays, axes_vec, slice_sizes_vec);
  return wrap_array(result);
}

static VALUE indexing_put_along_axis(VALUE self, VALUE arr, VALUE indices, VALUE values, VALUE axis) {
  mx::array& a = get_array(arr);
  mx::array& idx = get_array(indices);
  mx::array& vals = get_array(values);
  int ax = NIL_P(axis) ? 0 : NUM2INT(axis);
  
  mx::array result = mx::put_along_axis(a, idx, vals, ax);
  return wrap_array(result);
}

// Initialize indexing module
void init_indexing(VALUE module) {
  // Define module functions
  rb_define_module_function(module, "take", RUBY_METHOD_FUNC(indexing_take), 3);
  rb_define_module_function(module, "take_along_axis", RUBY_METHOD_FUNC(indexing_take_along_axis), 3);
  rb_define_module_function(module, "slice", RUBY_METHOD_FUNC(indexing_slice), 4);
  rb_define_module_function(module, "index", RUBY_METHOD_FUNC(indexing_index), 2);
  rb_define_module_function(module, "dynamic_slice", RUBY_METHOD_FUNC(indexing_dynamic_slice), 3);
  rb_define_module_function(module, "scatter", RUBY_METHOD_FUNC(indexing_scatter), 4);
  rb_define_module_function(module, "scatter_add", RUBY_METHOD_FUNC(indexing_scatter_add), 4);
  rb_define_module_function(module, "scatter_prod", RUBY_METHOD_FUNC(indexing_scatter_prod), 4);
  rb_define_module_function(module, "scatter_max", RUBY_METHOD_FUNC(indexing_scatter_max), 4);
  rb_define_module_function(module, "scatter_min", RUBY_METHOD_FUNC(indexing_scatter_min), 4);
  rb_define_module_function(module, "gather", RUBY_METHOD_FUNC(indexing_gather), 4);
  rb_define_module_function(module, "put_along_axis", RUBY_METHOD_FUNC(indexing_put_along_axis), 4);
}

// Exported functions for use in array.cpp for advanced indexing
mx::array ruby_get_item(const mx::array& src, VALUE obj) {
  if (NIL_P(obj)) {
    return src;
  } else if (RB_TYPE_P(obj, T_FIXNUM)) {
    return mlx_get_item_int(src, obj);
  } else if (rb_obj_is_kind_of(obj, rb_path2class("MLX::Core::Array"))) {
    return mlx_get_item_array(src, get_array(obj));
  } else if (rb_obj_is_kind_of(obj, rb_cRange)) {
    VALUE range_begin = rb_funcall(obj, rb_intern("begin"), 0);
    VALUE range_end = rb_funcall(obj, rb_intern("end"), 0);
    VALUE exclude_end = rb_funcall(obj, rb_intern("exclude_end?"), 0);
    
    int start = NUM2INT(range_begin);
    int end = NUM2INT(range_end);
    if (RTEST(exclude_end)) {
      end = end - 1;
    }
    
    return mlx_get_item_slice(src, INT2NUM(start), INT2NUM(end), INT2NUM(1));
  } else if (RB_TYPE_P(obj, T_ARRAY)) {
    std::vector<VALUE> indices;
    for (long i = 0; i < RARRAY_LEN(obj); i++) {
      indices.push_back(rb_ary_entry(obj, i));
    }
    int max_dims = 0;
    return mlx_gather_nd(src, indices, true, max_dims);
  } else {
    rb_raise(rb_eTypeError, "Unsupported indexing type");
    return mx::array(0); // Use scalar value instead of default construction
  }
} 