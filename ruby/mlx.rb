# MLX Ruby - Ruby bindings for Apple's MLX machine learning framework
# This is the main entry point for the MLX Ruby library.
#
# The library is structured as follows:
# - mlx/ext/mlxc - C extension for core functionality
# - mlx/core.rb - Ruby wrapper for core functionality
# - mlx/nn/* - Neural network modules and layers
# - mlx/optimizers/* - Optimizers and learning rate schedulers

require 'mlx/version'
require 'mlx/core'

# Neural network core
require 'mlx/nn/base'
require 'mlx/nn/init'
require 'mlx/nn/loss'

# Optimizers and schedulers
require 'mlx/optimizers/optimizers'
require 'mlx/optimizers/schedulers'

# Neural network layers
require 'mlx/nn/layers/linear'
require 'mlx/nn/layers/activations'
require 'mlx/nn/layers/containers'
require 'mlx/nn/layers/normalization'
require 'mlx/nn/layers/convolution'
require 'mlx/nn/layers/pooling'
require 'mlx/nn/layers/recurrent'
require 'mlx/nn/layers/transformer'
require 'mlx/nn/layers/dropout'
require 'mlx/nn/layers/embedding'
require 'mlx/nn/layers/positional_encoding'
require 'mlx/nn/layers/upsample'

# MLX main module
module MLX
  # VERSION is defined in version.rb
  
  # Data types
  INT8 = :int8
  INT16 = :int16
  INT32 = :int32
  INT64 = :int64
  UINT8 = :uint8
  UINT16 = :uint16
  UINT32 = :uint32
  UINT64 = :uint64
  FLOAT16 = :float16
  FLOAT32 = :float32
  BFLOAT16 = :bfloat16
  BOOL = :bool
  
  # Helper method to get device information
  def self.device
    Device.default
  end
  
  # Helper method to set default device
  def self.set_default_device(device)
    Device.set_default(device)
  end
  
  # Helper method to synchronize operations
  def self.sync
    Stream.synchronize
  end
  
  # Helper method to log current memory usage
  def self.memory_stats
    Memory.info
  end
  
  # Print array with formatting options
  def self.print(array, precision: 4, max_width: 80)
    formatted = array.to_s(precision: precision, max_width: max_width)
    puts formatted
  end
  
  # Helper method to convert tensors to Ruby arrays
  def self.to_ruby(array)
    array.to_a
  end
  
  # Check if CUDA is available
  def self.cuda_available?
    device_count = Device.count('gpu')
    device_count > 0
  end
  
  # Forward common math operations directly to MLX core
  class << self
    # Array creation functions
    def array(data, dtype: nil)
      MLX::Array.new(data, dtype: dtype)
    end
    
    def zeros(shape, dtype: FLOAT32)
      MLX::Array.zeros(shape, dtype: dtype)
    end
    
    def ones(shape, dtype: FLOAT32)
      MLX::Array.ones(shape, dtype: dtype)
    end
    
    def full(shape, value, dtype: nil)
      MLX::Array.full(shape, value, dtype: dtype)
    end
    
    def arange(start, stop = nil, step = 1, dtype: nil)
      MLX::Array.arange(start, stop, step, dtype: dtype)
    end
    
    def linspace(start, stop, num = 50, dtype: nil)
      MLX::Array.linspace(start, stop, num, dtype: dtype)
    end
    
    def eye(n, m = nil, dtype: FLOAT32)
      MLX::Array.eye(n, m, dtype: dtype)
    end
    
    def identity(n, dtype: FLOAT32)
      MLX::Array.identity(n, dtype: dtype)
    end
    
    def zeros_like(array)
      MLX::Array.zeros_like(array)
    end
    
    def ones_like(array)
      MLX::Array.ones_like(array)
    end
    
    def full_like(array, value)
      MLX::Array.full_like(array, value)
    end
    
    # Random functions
    def random_uniform(low = 0.0, high = 1.0, shape = [], dtype: FLOAT32)
      Random.uniform(low, high, shape, dtype: dtype)
    end
    
    def random_normal(mean = 0.0, std = 1.0, shape = [], dtype: FLOAT32)
      Random.normal(mean, std, shape, dtype: dtype)
    end
    
    # Basic operations
    def add(a, b)
      a + b
    end
    
    def subtract(a, b)
      a - b
    end
    
    def multiply(a, b)
      a * b
    end
    
    def divide(a, b)
      a / b
    end
    
    def power(a, b)
      a ** b
    end
    
    def negative(a)
      -a
    end
    
    # Math functions
    def exp(x)
      MLX::Math.exp(x)
    end
    
    def log(x)
      MLX::Math.log(x)
    end
    
    def log10(x)
      MLX::Math.log10(x)
    end
    
    def log2(x)
      MLX::Math.log2(x)
    end
    
    def log1p(x)
      MLX::Math.log1p(x)
    end
    
    def sigmoid(x)
      MLX::Math.sigmoid(x)
    end
    
    def tanh(x)
      MLX::Math.tanh(x)
    end
    
    def sin(x)
      MLX::Math.sin(x)
    end
    
    def cos(x)
      MLX::Math.cos(x)
    end
    
    def tan(x)
      MLX::Math.tan(x)
    end
    
    def square(x)
      MLX::Math.square(x)
    end
    
    def sqrt(x)
      MLX::Math.sqrt(x)
    end
    
    def abs(x)
      MLX::Math.abs(x)
    end
    
    def sign(x)
      MLX::Math.sign(x)
    end
    
    def clip(x, min_val, max_val)
      MLX::Math.clip(x, min_val, max_val)
    end
    
    # Reduction operations
    def sum(x, axis: nil, keepdim: false)
      MLX::Ops.sum(x, axis: axis, keepdim: keepdim)
    end
    
    def prod(x, axis: nil, keepdim: false)
      MLX::Ops.prod(x, axis: axis, keepdim: keepdim)
    end
    
    def mean(x, axis: nil, keepdim: false)
      MLX::Ops.mean(x, axis: axis, keepdim: keepdim)
    end
    
    def max(x, axis: nil, keepdim: false)
      MLX::Ops.max(x, axis: axis, keepdim: keepdim)
    end
    
    def min(x, axis: nil, keepdim: false)
      MLX::Ops.min(x, axis: axis, keepdim: keepdim)
    end
    
    def argmax(x, axis: nil, keepdim: false)
      MLX::Ops.argmax(x, axis: axis, keepdim: keepdim)
    end
    
    def argmin(x, axis: nil, keepdim: false)
      MLX::Ops.argmin(x, axis: axis, keepdim: keepdim)
    end
    
    # Comparison operations
    def equal(a, b)
      a == b
    end
    
    def not_equal(a, b)
      a != b
    end
    
    def greater(a, b)
      a > b
    end
    
    def greater_equal(a, b)
      a >= b
    end
    
    def less(a, b)
      a < b
    end
    
    def less_equal(a, b)
      a <= b
    end
    
    def maximum(a, b)
      MLX::Ops.maximum(a, b)
    end
    
    def minimum(a, b)
      MLX::Ops.minimum(a, b)
    end
    
    # Logic operations
    def logical_and(a, b)
      MLX::Ops.logical_and(a, b)
    end
    
    def logical_or(a, b)
      MLX::Ops.logical_or(a, b)
    end
    
    def logical_not(a)
      MLX::Ops.logical_not(a)
    end
    
    def where(condition, a, b)
      MLX::Ops.where(condition, a, b)
    end
    
    # Linear algebra operations
    def matmul(a, b)
      MLX::Linalg.matmul(a, b)
    end
    
    def dot(a, b)
      MLX::Linalg.dot(a, b)
    end
    
    def vdot(a, b)
      MLX::Linalg.vdot(a, b)
    end
    
    # Shape operations
    def reshape(a, shape)
      MLX::Ops.reshape(a, shape)
    end
    
    def transpose(a, axes = nil)
      MLX::Ops.transpose(a, axes)
    end
    
    def concatenate(arrays, axis = 0)
      MLX::Ops.concatenate(arrays, axis)
    end
    
    def stack(arrays, axis = 0)
      MLX::Ops.stack(arrays, axis)
    end
    
    def split(a, indices_or_sections, axis = 0)
      MLX::Ops.split(a, indices_or_sections, axis)
    end
    
    def pad(a, pad_width, mode = 'constant', constant_value = 0)
      MLX::Ops.pad(a, pad_width, mode, constant_value)
    end
    
    def squeeze(a, axis = nil)
      MLX::Ops.squeeze(a, axis)
    end
    
    def expand_dims(a, axis)
      MLX::Ops.expand_dims(a, axis)
    end
    
    # Indexing operations
    def slice(a, start, size = nil)
      MLX::Ops.slice(a, start, size)
    end
    
    def gather(a, indices, axis = 0)
      MLX::Ops.gather(a, indices, axis)
    end
    
    def update_slice(a, b, start)
      MLX::Ops.update_slice(a, b, start)
    end
    
    def take(a, indices, axis = nil)
      MLX::Ops.take(a, indices, axis)
    end
    
    # Special functions
    def softmax(x, axis = -1)
      MLX::Ops.softmax(x, axis)
    end
    
    def log_softmax(x, axis = -1)
      MLX::Ops.log_softmax(x, axis)
    end
    
    def one_hot(indices, num_classes, on_value = 1, off_value = 0)
      MLX::Ops.one_hot(indices, num_classes, on_value, off_value)
    end
    
    def dropout(x, p: 0.5, training: true)
      MLX::Ops.dropout(x, p: p, training: training)
    end
    
    # Neural network specific operations
    def conv1d(x, weight, stride: 1, padding: 0, dilation: 1, groups: 1)
      MLX::NN::Ops.conv1d(x, weight, stride: stride, padding: padding, dilation: dilation, groups: groups)
    end
    
    def conv2d(x, weight, stride: 1, padding: 0, dilation: 1, groups: 1)
      MLX::NN::Ops.conv2d(x, weight, stride: stride, padding: padding, dilation: dilation, groups: groups)
    end
    
    def conv3d(x, weight, stride: 1, padding: 0, dilation: 1, groups: 1)
      MLX::NN::Ops.conv3d(x, weight, stride: stride, padding: padding, dilation: dilation, groups: groups)
    end
    
    def max_pool1d(x, kernel_size, stride: nil, padding: 0, ceil_mode: false)
      MLX::NN::Ops.max_pool1d(x, kernel_size, stride: stride, padding: padding, ceil_mode: ceil_mode)
    end
    
    def max_pool2d(x, kernel_size, stride: nil, padding: 0, ceil_mode: false)
      MLX::NN::Ops.max_pool2d(x, kernel_size, stride: stride, padding: padding, ceil_mode: ceil_mode)
    end
    
    def max_pool3d(x, kernel_size, stride: nil, padding: 0, ceil_mode: false)
      MLX::NN::Ops.max_pool3d(x, kernel_size, stride: stride, padding: padding, ceil_mode: ceil_mode)
    end
    
    def avg_pool1d(x, kernel_size, stride: nil, padding: 0, ceil_mode: false, count_include_pad: true)
      MLX::NN::Ops.avg_pool1d(x, kernel_size, stride: stride, padding: padding, 
                            ceil_mode: ceil_mode, count_include_pad: count_include_pad)
    end
    
    def avg_pool2d(x, kernel_size, stride: nil, padding: 0, ceil_mode: false, count_include_pad: true)
      MLX::NN::Ops.avg_pool2d(x, kernel_size, stride: stride, padding: padding, 
                            ceil_mode: ceil_mode, count_include_pad: count_include_pad)
    end
    
    def avg_pool3d(x, kernel_size, stride: nil, padding: 0, ceil_mode: false, count_include_pad: true)
      MLX::NN::Ops.avg_pool3d(x, kernel_size, stride: stride, padding: padding, 
                            ceil_mode: ceil_mode, count_include_pad: count_include_pad)
    end
    
    # Matrix operations
    def triu(a, k = 0)
      MLX::Linalg.triu(a, k)
    end
    
    def tril(a, k = 0)
      MLX::Linalg.tril(a, k)
    end
    
    def diag(a, k = 0)
      MLX::Linalg.diag(a, k)
    end
    
    def diag_part(a)
      MLX::Linalg.diag_part(a)
    end
    
    # Model I/O
    def save(model, path)
      MLX::IO.save(model, path)
    end
    
    def load(path)
      MLX::IO.load(path)
    end
  end
end 