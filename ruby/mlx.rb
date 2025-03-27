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

module MLX
  
    def self.array(data, dtype = nil); Core::Array.new(data, dtype: dtype); end

    def self.core
        Core
    end

    def self.nn
        NN
    end

    def self.optimizers
        Optimizers
    end

    def self.schedulers
        Schedulers
    end

    def self.fast
        Core::Fast
    end

    def self.fft
        Core::FFT
    end

    def self.linalg
        Core::Linalg
    end

    def self.metal
        Core::Metal
    end

    def self.random
        Core::Random
    end

    def self.device
        Core::Device
    end

    def self.stream
        Core::Stream
    end

    def self.metal
        Core::Metal
    end

    def self.memory
        Core::Memory
    end

    def self.indexing
        Core::Indexing
    end

    def self.ops
        Core::Ops
    end

    def self.transforms
        Core::Transforms
    end

    def self.trees
        Core::Trees
    end

    def self.convert
        Core::Convert
    end

    def self.load
        Core::Load
    end

    def self.math
        Core::Math
    end

    def self.linalg
        Core::Linalg
    end

    # Methods delegating to Core::Ops 
    def self.zeros(shape, dtype = nil, stream = nil); Core::Ops.zeros(shape, dtype, stream); end
    def self.ones(shape, dtype = nil, stream = nil); Core::Ops.ones(shape, dtype, stream); end
    def self.full(shape, fill_value, dtype = nil, stream = nil); Core::Ops.full(shape, fill_value, dtype, stream); end
    def self.arange(start, stop, step = 1, dtype = nil, stream = nil); Core::Ops.arange(start, stop, step, dtype, stream); end
    def self.identity(n, dtype = nil, stream = nil); Core::Ops.identity(n, dtype, stream); end
    def self.eye(n, m = nil, k = 0, dtype = nil, stream = nil); Core::Ops.eye(n, m, k, dtype, stream); end

    def self.reshape(arr, shape, stream = nil); Core::Ops.reshape(arr, shape); end
    def self.flatten(arr, start_axis = 0, end_axis = -1, stream = nil); Core::Ops.flatten(arr, start_axis, end_axis); end
    def self.squeeze(arr, axis = nil, stream = nil); Core::Ops.squeeze(arr, axis); end
    def self.expand_dims(arr, axis, stream = nil); Core::Ops.expand_dims(arr, axis); end
    
    def self.abs(x); Core::Ops.abs(x); end
    def self.sign(x); Core::Ops.sign(x); end
    def self.negative(x); Core::Ops.negative(x); end

    def self.add(x, y); Core::Ops.add(x, y); end
    def self.subtract(x, y); Core::Ops.subtract(x, y); end
    def self.multiply(x, y); Core::Ops.multiply(x, y); end
    def self.divide(x, y); Core::Ops.divide(x, y); end
    
    def self.equal(x, y); Core::Ops.equal(x, y); end
    def self.not_equal(x, y); Core::Ops.not_equal(x, y); end
    def self.greater(x, y); Core::Ops.greater(x, y); end
    def self.greater_equal(x, y); Core::Ops.greater_equal(x, y); end
    def self.less(x, y); Core::Ops.less(x, y); end
    def self.less_equal(x, y); Core::Ops.less_equal(x, y); end

    # Methods delegating to Core::Array (Element-wise operations)
    def self.square(x); x.square; end
    def self.sqrt(x); x.sqrt; end
    def self.rsqrt(x); x.rsqrt; end
    
    # Methods delegating to Core::Array (Reduction operations)
    def self.all(x, axis = nil, keepdims = false); x.all(axis, keepdims); end
    def self.any(x, axis = nil, keepdims = false); x.any(axis, keepdims); end
    def self.sum(x, axis = nil, keepdims = false); x.sum(axis, keepdims); end
    def self.prod(x, axis = nil, keepdims = false); x.prod(axis, keepdims); end
    def self.mean(x, axis = nil, keepdims = false); x.mean(axis, keepdims); end
    def self.max(x, axis = nil, keepdims = false); x.max(axis, keepdims); end
    def self.min(x, axis = nil, keepdims = false); x.min(axis, keepdims); end
    def self.argmax(x, axis = nil, keepdims = false); x.argmax(axis, keepdims); end
    def self.argmin(x, axis = nil, keepdims = false); x.argmin(axis, keepdims); end
    
    # Methods delegating to x (Logical operations)
    def self.logical_not(x); x.logical_not; end
    def self.logical_and(x, y); x.logical_and(y); end
    def self.logical_or(x, y); x.logical_or(y); end
    def self.logical_xor(x, y); x.logical_xor(y); end

    # Methods delegating to x (Array creation helpers)
    def self.zeros_like(x); x.zeros_like; end
    def self.ones_like(x); x.ones_like; end
    def self.full_like(x, value); x.full_like(value); end
    
    # Methods delegating to x (Array operations)
    def self.diag(x, k = 0); x.diag(k); end
    def self.diag_part(x); x.diag_part; end
    def self.tril(x, k = 0); x.tril(k); end
    def self.triu(x, k = 0); x.triu(k); end
    def self.flip(x, axis); x.flip(axis); end
    def self.outer(x, y); x.outer(y); end
    def self.vdot(x, y); x.vdot(y); end
    
    # Methods delegating to x (Advanced operations)
    def self.einsum(subscripts, *operands); x.einsum(subscripts, *operands); end
    def self.einsum_path(subscripts, *operands); x.einsum_path(subscripts, *operands); end
    def self.export_function(fn, *args); x.export_function(fn, *args); end
    def self.export_to_dot(fn, *args); x.export_to_dot(fn, *args); end
    def self.exporter(fn); x.exporter(fn); end
    def self.import_function(path); x.import_function(path); end
    def self.jvp(fn, primals, tangents); x.jvp(fn, primals, tangents); end
    def self.vjp(fn, primals); x.vjp(fn, primals); end
    def self.vmap(fn); x.vmap(fn); end
    
    # Methods delegating to x (Array properties)
    def self.isfinite(x); x.isfinite; end
    def self.isinf(x); x.isinf; end
    def self.isnan(x); x.isnan; end
    def self.allclose(x, y, rtol = 1e-5, atol = 1e-8); x.allclose(y, rtol, atol); end
    def self.array_equal(x, y); x.array_equal(y); end
    
    # Methods delegating to x (Neural network operations)
    def self.softmax(x, axis = -1); x.softmax(axis); end
    def self.softplus(x); x.softplus; end
    def self.dropout(x, p, training = true); x.dropout(p, training); end
    def self.one_hot(x, num_classes); x.one_hot(num_classes); end
    
    # Methods delegating to x (Convolution operations)
    def self.conv1d(x, weight, bias = nil, stride = 1, padding = 0, dilation = 1, groups = 1); x.conv1d(weight, bias, stride, padding, dilation, groups); end
    def self.conv2d(x, weight, bias = nil, stride = 1, padding = 0, dilation = 1, groups = 1); x.conv2d(weight, bias, stride, padding, dilation, groups); end
    def self.conv3d(x, weight, bias = nil, stride = 1, padding = 0, dilation = 1, groups = 1); x.conv3d(weight, bias, stride, padding, dilation, groups); end
    def self.conv_general(x, weight, bias = nil, stride = 1, padding = 0, dilation = 1, groups = 1); x.conv_general(weight, bias, stride, padding, dilation, groups); end
    def self.conv_transpose1d(x, weight, bias = nil, stride = 1, padding = 0, output_padding = 0, groups = 1, dilation = 1); x.conv_transpose1d(weight, bias, stride, padding, output_padding, groups, dilation); end
    def self.conv_transpose2d(x, weight, bias = nil, stride = 1, padding = 0, output_padding = 0, groups = 1, dilation = 1); x.conv_transpose2d(weight, bias, stride, padding, output_padding, groups, dilation); end
    def self.conv_transpose3d(x, weight, bias = nil, stride = 1, padding = 0, output_padding = 0, groups = 1, dilation = 1); x.conv_transpose3d(weight, bias, stride, padding, output_padding, groups, dilation); end
    def self.convolve(a, v, mode = :full); x.convolve(a, v, mode); end
    
    # Methods delegating to x (Pooling operations)
    def self.max_pool1d(x, kernel_size, stride = nil, padding = 0, dilation = 1); x.max_pool1d(kernel_size, stride, padding, dilation); end
    def self.max_pool2d(x, kernel_size, stride = nil, padding = 0, dilation = 1); x.max_pool2d(kernel_size, stride, padding, dilation); end
    def self.max_pool3d(x, kernel_size, stride = nil, padding = 0, dilation = 1); x.max_pool3d(kernel_size, stride, padding, dilation); end
    def self.avg_pool1d(x, kernel_size, stride = nil, padding = 0); x.avg_pool1d(kernel_size, stride, padding); end
    def self.avg_pool2d(x, kernel_size, stride = nil, padding = 0); x.avg_pool2d(kernel_size, stride, padding); end
    def self.avg_pool3d(x, kernel_size, stride = nil, padding = 0); x.avg_pool3d(kernel_size, stride, padding); end

    
    # Methods delegating to Core::Device (Device operations)
    def self.to_cpu(x); Core::Device.to_cpu(x); end
    def self.to_gpu(x); Core::Device.to_gpu(x); end
    def self.to_ruby(x); Core::Device.to_ruby(x); end
    def self.set_default_device(device); Core::Device.set_default_device(device); end
    
    # Methods delegating to Core::Memory (Memory management)
    def self.set_cache_limit(limit); Core::Memory.set_cache_limit(limit); end
    def self.set_memory_limit(limit); Core::Memory.set_memory_limit(limit); end
    def self.set_wired_limit(limit); Core::Memory.set_wired_limit(limit); end
    
    # Methods delegating to Core::Transforms
    def self.async_eval(fn, *args); Core::Transforms.eval(fn, *args); end
    def self.broadcast_to(x, shape); Core::Transforms.broadcast_to(x, shape); end
    def self.compile(fn); Core::Transforms.compile(fn); end
    def self.eval(fn, *args); Core::Transforms.eval(fn, *args); end
    def self.expand_dims(x, axis); Core::Transforms.expand_dims(x, axis); end
    def self.grad(fn, *args); Core::Transforms.grad(fn, *args); end
    def self.moveaxis(x, source, destination); Core::Transforms.moveaxis(x, source, destination); end
    def self.pad(x, pad_width, mode = :constant, constant_values = 0); Core::Transforms.pad(x, pad_width, mode, constant_values); end
    def self.permute_dims(x, axes); Core::Transforms.transpose(x, axes); end
    def self.repeat(x, repeats, axis = nil); Core::Transforms.repeat(x, repeats, axis); end
    def self.split(x, indices_or_sections, axis = 0); Core::Transforms.split(x, indices_or_sections, axis); end
    def self.stack(arrays, axis = 0); Core::Transforms.stack(arrays, axis); end
    def self.stop_gradient(x); Core::Transforms.stop_gradient(x); end
    def self.tile(x, reps); Core::Transforms.tile(x, reps); end
    def self.transpose(x, axes = nil); Core::Transforms.transpose(x, axes); end
    def self.value_and_grad(fn, *args); Core::Transforms.value_and_grad(fn, *args); end
    def self.concatenate(arrays, axis = 0); Core::Transforms.concatenate(arrays, axis); end
    
    # Methods delegating to Core::Indexing
    def self.gather(x, indices, axis = nil); Core::Indexing.gather(x, indices, axis); end
    def self.index_select(x, dim, index); Core::Indexing.index_select(x, dim, index); end
    def self.scatter(x, indices, updates, axis = nil); Core::Indexing.scatter(x, indices, updates, axis); end
    def self.scatter_add(x, indices, updates, axis = nil); Core::Indexing.scatter_add(x, indices, updates, axis); end
    def self.slice(x, start_indices, lengths, strides = nil); Core::Indexing.slice(x, start_indices, lengths, strides); end
    def self.take(x, indices, axis = nil); Core::Indexing.take(x, indices, axis); end
    def self.take_along_axis(x, indices, axis); Core::Indexing.take_along_axis(x, indices, axis); end
    def self.update_slice(x, start_indices, lengths, values); Core::Indexing.update_slice(x, start_indices, lengths, values); end
    
    # Methods delegating to Core::Math
    def self.cos(x); Core::Math.cos(x); end
    def self.exp(x); Core::Math.exp(x); end
    def self.log(x); Core::Math.log(x); end
    def self.sigmoid(x); Core::Math.sigmoid(x); end
    def self.sin(x); Core::Math.sin(x); end
    def self.tanh(x); Core::Math.tanh(x); end
    
    # Methods delegating to Core::Constants
    def self.e; Core::Constants.e; end
    def self.pi; Core::Constants.pi; end
    
    # Methods delegating to Core::Stream
    def self.default_stream; Core::Stream.default_stream; end
    def self.new_stream; Core::Stream.new_stream; end
    def self.stream; Core::Stream.stream; end
    
    # Methods delegating to Core::Load
    def self.load(path, arrays = nil, device = nil); Core::Load.load(path, arrays, device); end
    def self.loadz(path, arrays = nil, device = nil); Core::Load.load_npz(path, arrays, device); end
    def self.save(path, arrays, compression = false, enable_async = false); Core::Load.save(path, arrays, compression, enable_async); end
    def self.save_gguf(path, arrays, enable_async = false); Core::Load.save_gguf(path, arrays, enable_async); end
    def self.save_safetensors(path, arrays, enable_async = false); Core::Load.save_safetensors(path, arrays, enable_async); end
    def self.savez(path, arrays, enable_async = false); Core::Load.savez(path, arrays, enable_async); end
    
    # Methods delegating to Core::Random
    def self.random_normal(shape, mean = 0.0, std = 1.0, dtype = nil); Core::Random.normal(Core::Random.key, shape, dtype); end
    def self.random_uniform(shape, low = 0.0, high = 1.0, dtype = nil); Core::Random.uniform(Core::Random.key, shape, dtype); end
    
    # Methods delegating to Core::Convert
    def self.convert(x, dtype); Core::Convert.to_type(x, dtype); end
    def self.double(x); Core::Convert.to_type(x, Core::FLOAT64); end
    def self.float32(x); Core::Convert.to_type(x, Core::FLOAT32); end
    def self.int16(x); Core::Convert.to_type(x, Core::INT16); end
    
    # Methods delegating to Core::Array
    def self.astype(x, dtype); x.astype(dtype); end

end