#pragma once

#include <ruby.h>
#include <variant>
#include <optional>
#include <functional>
#include <numeric>
#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/device.h"

namespace mx = mlx::core;

// Forward declaration of internal Ruby class
extern VALUE rb_cMLXArray;

// Type alias for stream or device
using StreamOrDevice = std::variant<std::monostate, mx::Stream, mx::Device>;

// StreamContext class
class StreamContext {
public:
  // Remove default constructor as Stream requires initialization
  explicit StreamContext(const StreamOrDevice& stream_or_device) : 
      device_(mx::Device::cpu), // Initialize with default CPU device
      stream_(mx::default_stream(mx::Device::cpu)) // Initialize with default stream
  {
    if (auto pv = std::get_if<mx::Stream>(&stream_or_device); pv) {
      stream_ = *pv;
      device_ = pv->device;
    } else if (auto pv = std::get_if<mx::Device>(&stream_or_device); pv) {
      device_ = *pv;
      stream_ = mx::default_stream(device_);
    }
    // If monostate, keep the default CPU device and stream
  }
  
  ~StreamContext() = default;

  // Getters
  mx::Stream& stream() { return stream_; }
  mx::Device& device() { return device_; }

private:
  mx::Device device_;  // Device must be defined before stream_
  mx::Stream stream_;  // Stream depends on device
};

// Helper functions for tree operations
std::vector<int> get_reduce_axes(const std::variant<std::monostate, int, std::vector<int>>& v, int dims);

// Common utility functions used by multiple files - implemented inline to avoid duplicate definitions
inline mx::array& get_array(VALUE obj) {
  if (rb_obj_is_kind_of(obj, rb_cMLXArray)) {
    VALUE iv_array = rb_ivar_get(obj, rb_intern("@array"));
    mx::array* arr_ptr = static_cast<mx::array*>(DATA_PTR(iv_array));
    return *arr_ptr;
  } else {
    mx::array* arr_ptr;
    Data_Get_Struct(obj, mx::array, arr_ptr);
    return *arr_ptr;
  }
}

inline bool responds_to_to_mlx_array(VALUE obj) {
  return rb_respond_to(obj, rb_intern("to_mlx_array"));
}

inline mx::array convert_using_to_mlx_array(VALUE obj) {
  VALUE result = rb_funcall(obj, rb_intern("to_mlx_array"), 0);
  if (!rb_obj_is_kind_of(result, rb_cMLXArray)) {
    rb_raise(rb_eTypeError, "to_mlx_array did not return an MLX::Array");
    return mx::array(0.0f); // Default placeholder value
  }
  return get_array(result);
}

inline VALUE wrap_array(const mx::array& arr) {
  // Create a new MLX::Array Ruby object
  VALUE mlx_array = rb_class_new_instance(0, nullptr, rb_cMLXArray);
  
  // Create a new mlx::core::array and copy the data
  mx::array* arr_copy = new mx::array(arr);
  
  // Wrap the array in a Ruby DATA object
  VALUE data = Data_Wrap_Struct(rb_cObject, nullptr, 
      [](void* p) { delete static_cast<mx::array*>(p); }, 
      arr_copy);
  
  // Set the @array instance variable
  rb_ivar_set(mlx_array, rb_intern("@array"), data);
  
  return mlx_array;
}

// Helper functions for converting between Ruby and MLX types
mx::array to_array(VALUE v, std::optional<mx::Dtype> dtype = std::nullopt);
std::pair<mx::array, mx::array> to_arrays(VALUE a, VALUE b);

// Helper to extract stream or device from Ruby object
StreamOrDevice get_stream_or_device(VALUE obj);

// Ruby wrapper function for creating StreamContext
VALUE rb_create_stream_context(int argc, VALUE* argv, VALUE self);

// Initialize the utils module
void init_utils(VALUE module); 