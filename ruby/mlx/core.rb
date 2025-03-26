# This file contains core functionality for the MLX Ruby bindings
# It provides direct access to the low-level C++ MLX API
require 'mlx/ext/core'

# Require the distributed_run module
require_relative 'distributed_run'

module MLX
  # Easy access to Core classes
  Array = Core::Array
  Stream = Core::Stream
  
  # Define dtype constants at the top level
  BOOL = Core::BOOL
  UINT8 = Core::UINT8
  UINT16 = Core::UINT16
  UINT32 = Core::UINT32
  UINT64 = Core::UINT64
  INT8 = Core::INT8
  INT16 = Core::INT16
  INT32 = Core::INT32
  INT64 = Core::INT64
  FLOAT16 = Core::FLOAT16
  FLOAT32 = Core::FLOAT32
  BFLOAT16 = Core::BFLOAT16
  COMPLEX64 = Core::COMPLEX64
  
  # Device constants
  CPU = Core::CPU
  GPU = Core::GPU
  
  # Factory method for creating arrays
  def self.array(data, dtype = nil)
    arr = Array.new(data)
    # Apply dtype conversion if specified
    if dtype
      arr = convert(arr, dtype)
    end
    arr
  end
  
  # Zero arrays
  def self.zeros(shape, dtype = FLOAT32)
    Core.zeros(shape, dtype)
  end
  
  # One arrays
  def self.ones(shape, dtype = FLOAT32)
    Core.ones(shape, dtype)
  end
  
  # Full arrays
  def self.full(shape, fill_value, dtype = FLOAT32)
    Core.full(shape, fill_value, dtype)
  end
  
  # Type conversion
  def self.convert(arr, dtype)
    Core.to_type(arr, dtype)
  end
  
  def self.to_float16(arr)
    Core.to_float16(arr)
  end
  
  def self.to_float32(arr)
    Core.to_float32(arr)
  end
  
  def self.to_int32(arr)
    Core.to_int32(arr)
  end
  
  def self.to_bool(arr)
    Core.to_bool(arr)
  end
  
  # Constants
  def self.pi
    Core.pi
  end
  
  def self.e
    Core.e
  end
  
  def self.inf
    Core.inf
  end
  
  def self.nan
    Core.nan
  end
  
  # Random module methods
  module Random
    def self.key(seed)
      Core.key(seed)
    end
    
    def self.split(key, num)
      Core.split(key, num)
    end
    
    def self.uniform(key, shape, dtype = FLOAT32)
      Core.uniform(key, shape, dtype)
    end
    
    def self.normal(key, shape, dtype = FLOAT32)
      Core.normal(key, shape, dtype)
    end
    
    def self.randint(key, low, high, shape, dtype = INT32)
      Core.randint(key, low, high, shape, dtype)
    end
    
    def self.bernoulli(key, p, shape)
      Core.bernoulli(key, p, shape)
    end
  end
  
  # Linear algebra module methods
  module Linalg
    def self.norm(arr, ord = nil, axis = nil)
      Core.norm(arr, ord, axis)
    end
    
    def self.svd(arr, full_matrices = false)
      Core.svd(arr, full_matrices)
    end
    
    def self.qr(arr, mode = "reduced")
      Core.qr(arr, mode)
    end
    
    def self.inv(arr)
      Core.inv(arr)
    end
    
    def self.matmul(a, b)
      Core.matmul(a, b)
    end
    
    def self.det(arr)
      Core.det(arr)
    end
  end
  
  # Fast Fourier Transform module
  module FFT
    def self.fft(arr, n = nil, axis = -1)
      Core.fft(arr, n, axis)
    end
    
    def self.ifft(arr, n = nil, axis = -1)
      Core.ifft(arr, n, axis)
    end
    
    def self.fft2(arr, s = nil, axes = nil)
      Core.fft2(arr, s, axes)
    end
    
    def self.ifft2(arr, s = nil, axes = nil)
      Core.ifft2(arr, s, axes)
    end
    
    def self.fftn(arr, s = nil, axes = nil)
      Core.fftn(arr, s, axes)
    end
    
    def self.ifftn(arr, s = nil, axes = nil)
      Core.ifftn(arr, s, axes)
    end
  end
  
  # Fast operations for neural networks
  module Fast
    def self.gemm(a, b, c, transpose_a = false, transpose_b = false)
      Core.gemm(a, b, c, transpose_a, transpose_b)
    end
    
    def self.scaled_dot_product_attention(queries, keys, values, scale, mask = nil)
      Core.scaled_dot_product_attention(queries, keys, values, scale, mask)
    end
    
    def self.multi_head_attention(queries, keys, values, num_heads)
      Core.multi_head_attention(queries, keys, values, num_heads)
    end
    
    def self.rms_norm(x, weight, eps = 1e-5)
      Core.rms_norm(x, weight, eps)
    end
    
    def self.layer_norm(x, weight, bias = nil, eps = 1e-5)
      Core.layer_norm(x, weight, bias, eps)
    end
    
    def self.rope(x, dims, traditional = false, base = 10000.0, scale = 1.0)
      Core.rope(x, dims, traditional, base, scale)
    end
    
    def self.rope_inplace(x, dims, traditional = false, base = 10000.0, scale = 1.0)
      Core.rope_inplace(x, dims, traditional, base, scale)
    end
  end
  
  # Distributed computing module
  module Distributed
    def self.initialize(communication_key, world_size, rank)
      Core.initialize(communication_key, world_size, rank)
    end
    
    def self.is_initialized?
      Core.is_initialized
    end
    
    def self.world_size
      Core.world_size
    end
    
    def self.rank
      Core.rank
    end
    
    def self.local_world_size
      Core.local_world_size
    end
    
    def self.local_rank
      Core.local_rank
    end
    
    def self.local_group_size
      Core.local_group_size
    end
    
    def self.local_group_rank
      Core.local_group_rank
    end
    
    def self.shutdown
      Core.shutdown
    end
    
    def self.barrier
      Core.barrier
    end
    
    def self.all_reduce(input, reduction)
      Core.all_reduce(input, reduction)
    end
    
    def self.all_gather(input)
      Core.all_gather(input)
    end
  end
end 