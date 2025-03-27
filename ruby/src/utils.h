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

// Helper functions for converting between Ruby and MLX types
mx::array to_array(VALUE v, std::optional<mx::Dtype> dtype = std::nullopt);
std::pair<mx::array, mx::array> to_arrays(VALUE a, VALUE b);

// Helper to extract stream or device from Ruby object
StreamOrDevice get_stream_or_device(VALUE obj);

// Common utility functions used by multiple files
extern mx::array& get_array(VALUE obj);
extern bool responds_to_to_mlx_array(VALUE obj);
extern mx::array convert_using_to_mlx_array(VALUE obj);
extern VALUE wrap_array(const mx::array& arr);

// Ruby wrapper function for creating StreamContext
VALUE rb_create_stream_context(int argc, VALUE* argv, VALUE self);

// Initialize the utils module
void init_utils(VALUE module); 