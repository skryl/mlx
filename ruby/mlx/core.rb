# Core implementation for MLX Ruby

# Set the path to the MLX library in the build directory
build_dir = File.expand_path('../../../build', __dir__)
ENV['DYLD_LIBRARY_PATH'] = "#{build_dir}:#{ENV['DYLD_LIBRARY_PATH']}"

# Define the MLX module
module MLX
  # Define the Core module
  module Core
    # Define data type constants in the Core module
    BOOL = :bool
    UINT8 = :uint8
    UINT16 = :uint16
    UINT32 = :uint32
    UINT64 = :uint64
    INT8 = :int8
    INT16 = :int16
    INT32 = :int32
    INT64 = :int64
    FLOAT16 = :float16
    FLOAT32 = :float32
    FLOAT64 = :float64
    BFLOAT16 = :bfloat16
    COMPLEX64 = :complex64
    
    # Type hierarchy constants
    COMPLEXFLOATING = :complexfloating
    FLOATING = :floating
    INEXACT = :inexact
    SIGNEDINTEGER = :signedinteger
    UNSIGNEDINTEGER = :unsignedinteger
    INTEGER = :integer
    NUMBER = :number
    GENERIC = :generic
    
    # Device constants
    CPU = :cpu
    GPU = :gpu

    VERSION = "0.1.0"    # (Matches MLX_VERSION from C++)
    DESCRIPTION = "MLX framework for machine learning on Apple silicon"
  
    # Full Class List
    class Array; end
    class ArrayAt; end
    class Device; end
    class Stream; end
    class StreamContext; end
    class Dtype; end
    class GcFunc; end

    # Full Module List
    module Constants; end
    module Convert; end
    module Distributed; end
    module Export; end
    module FFT; end
    module Fast; end
    module Indexing; end
    module Linalg; end
    module Load; end
    module Memory; end
    module Metal; end
    module Ops; end
    module Random; end
    module Transforms; end
    module Trees; end
    module Utils; end

    def self.modules
      [Constants, Convert, Distributed, Export, FFT, Fast, Indexing, Linalg, Load, Memory, Metal, Ops, Random, Transforms, Trees, Utils]
    end

    def self.classes
      [Array, ArrayAt, Device, Stream, StreamContext, Dtype, GcFunc]
    end
    
    # IMPORTANT: All the module methods below are just placeholders.
    # The real implementations are provided by the native extension.
    # Do not modify the module structure without updating the C++ code.

    # Define a simple version method
    def self.version
      VERSION
    end

    def self.platform
      "darwin"
    end

    # Constants module - implemented in constants.cpp
    module Constants
      # These will be overridden by the native extension
      def self.pi; 3.141592653589793; end
      def self.e; 2.718281828459045; end
      def self.euler_gamma; 0.57721566490153286; end
      def self.inf; Float::INFINITY; end
      def self.nan; Float::NAN; end
      def self.newaxis; nil; end
    end
    
    # Make constants available at module level 
    def self.pi; Constants.pi; end
    def self.e; Constants.e; end
    def self.euler_gamma; Constants.euler_gamma; end
    def self.inf; Constants.inf; end
    def self.nan; Constants.nan; end
    def self.newaxis; Constants.newaxis; end

    # (Optional) If you expose mlx_func_create at the Ruby level:
    def self.mlx_func_create(func, deps)
      # Implementation in C that calls 'mlx_func_create'
      # and returns a typed data object that behaves like GcFunc
    end

    class Dtype
      # Size (in bytes)
      def size; end
  
      # Equality check for two dtype objects
      def ==(other); end
  
      # Ruby requires a hash if you define ==, so we provide that
      def hash; end
  
      # String representation
      def to_s; end
    end

    class GcFunc
        # You normally do NOT call this directly; you get an object by calling
        # mlx_func_create(func, deps). The constructor is implied inside that C function.
        def initialize(func, deps)
          # C side stores these in the gc_func struct, 
          # but there is no public 'initialize' in real usage.
        end
    
        # Public: call the underlying function, forwarding all arguments.
        def call(*args)
          # Implementation in C calls rb_funcall2(func, :call, argc, argv).
        end
    
        # Public: return doc string if any.
        # Python side calls this __doc__ property. We mirror it in Ruby.
        def __doc__
          # Implementation in C returns Qnil or some doc from 'func' or stored metadata.
        end
    
        # Public: return signature data if any.
        # Python side calls this __nb_signature__. We mirror it in Ruby.
        def __nb_signature__
          # Implementation in C returns Qnil or some signature from 'func'.
        end
    
        # Public: return the vectorcall offset pointer (Python internal). 
        # Here just returns nil by default.
        def __vectorcalloffset__
          # Implementation in C returns Qnil (placeholder).
        end
    
        # Public: fallback for unknown methods/attributes.
        # This delegates to the underlying 'func' if it responds.
        def method_missing(m, *args, &block)
          # Implementation in C checks rb_respond_to(func, m)
          #   -> if true, calls rb_funcall2(func, m, args.size, args)
          #   -> else calls super
        end
    end

    class Device

      # Integer constants representing device types
      CPU = :cpu
      GPU = :gpu 
    
      # -- Module Methods --
    
      # Returns the current default device.
      # @return [MLX::Device]
      def self.default_device
        # ...
      end
    
      # Sets the default device. Accepts either an MLX::Device or an integer (CPU/GPU).
      # @param device [MLX::Device, Integer]
      # @return [nil]
      def self.set_default_device(device)
        # ...
      end
    
      # Placeholder for synchronization; no-op in MLX.
      # @return [nil]
      def self.sync_device
        # ...
      end
    
      # Returns an array of available devices (at least CPU, possibly GPU if available).
      # @return [Array<MLX::Device>]
      def self.devices
        # ...
      end
    
      # Create a new MLX::Device with a given device type (CPU/GPU) and index.
      # @param type [Integer] CPU or GPU
      # @param index [Integer] optional, defaults to 0
      def initialize(type, index = 0)
        # ...
      end
  
      # Returns the integer representation of this device’s type (CPU/GPU).
      # @return [Integer]
      def type
        # ...
      end
  
      # Returns a string representation of the device: "Device(type=cpu, index=0)", etc.
      # @return [String]
      def to_s
        # ...
      end
  
      # Checks equality with another MLX::Device or an integer device type.
      # @param other [MLX::Device, Integer]
      # @return [Boolean]
      def ==(other)
        # ...
      end
    
    end



    
    
    # Random module - implemented in random.cpp
    # 
    module Random

      # Returns the current seed "state" (the default_key().state in C++).
      # In Python, this is exposed as an attribute: random.state
      # In Ruby, we add a module function "state" to return the global key.
      def self.state
        # Stub
        # Return MLX::Core::Array containing the global PRNG state
      end

      #-------------------------------------------------------------------
      # Seeds the global PRNG (C++: default_key().seed(seed_val))
      #
      # @param [Integer] seed The seed value
      # @return [nil]
      #-------------------------------------------------------------------
      def self.seed(seed)
      end

      #-------------------------------------------------------------------
      # Returns a fresh PRNG key from a given seed
      #
      # @param [Integer] seed The seed value
      # @return [MLX::Core::Array] The PRNG key
      #-------------------------------------------------------------------
      def self.key(seed)
      end

      #-------------------------------------------------------------------
      # Splits a PRNG key into multiple subkeys
      #
      # @param [MLX::Core::Array] key
      # @param [Integer] num number of subkeys (default: 2)
      # @param [MLX::Stream, MLX::Device, nil] stream optional
      # @return [MLX::Core::Array] subkeys of shape [num, ...]
      #-------------------------------------------------------------------
      def self.split(key, num = 2, stream = nil)
      end

      #-------------------------------------------------------------------
      # Generates uniformly distributed random values: [low, high)
      #
      # @param [Numeric or MLX::Core::Array] low (default: 0)
      # @param [Numeric or MLX::Core::Array] high (default: 1)
      # @param [Array<Integer>] shape shape of the result (default: [])
      # @param [Integer, nil] dtype numeric code for the dtype (default: float32)
      # @param [MLX::Core::Array, nil] key optional custom PRNG key
      # @param [MLX::Stream, MLX::Device, nil] stream optional
      # @return [MLX::Core::Array]
      #-------------------------------------------------------------------
      def self.uniform(low = 0, high = 1, shape = [], dtype = nil, key = nil, stream = nil)
      end

      #-------------------------------------------------------------------
      # Generates normally distributed random values: mean=loc, std=scale
      #
      # @param [Array<Integer>] shape (default: [])
      # @param [Integer, nil] dtype (default: float32)
      # @param [Float] loc (default: 0.0)
      # @param [Float] scale (default: 1.0)
      # @param [MLX::Core::Array, nil] key
      # @param [MLX::Stream, MLX::Device, nil] stream
      # @return [MLX::Core::Array]
      #-------------------------------------------------------------------
      def self.normal(shape = [], dtype = nil, loc = 0.0, scale = 1.0, key = nil, stream = nil)
      end

      #-------------------------------------------------------------------
      # Generates multivariate normal random values given mean & cov
      #
      # @param [MLX::Core::Array] mean
      # @param [MLX::Core::Array] cov
      # @param [Array<Integer>] shape (default: [])
      # @param [Integer, nil] dtype (default: float32)
      # @param [MLX::Core::Array, nil] key
      # @param [MLX::Stream, MLX::Device, nil] stream
      # @return [MLX::Core::Array]
      #-------------------------------------------------------------------
      def self.multivariate_normal(mean, cov, shape = [], dtype = nil, key = nil, stream = nil)
      end

      #-------------------------------------------------------------------
      # Generates random integers in [low, high)
      #
      # @param [Numeric or MLX::Core::Array] low
      # @param [Numeric or MLX::Core::Array] high
      # @param [Array<Integer>] shape (default: [])
      # @param [Integer, nil] dtype (default: int32)
      # @param [MLX::Core::Array, nil] key
      # @param [MLX::Stream, MLX::Device, nil] stream
      # @return [MLX::Core::Array]
      #-------------------------------------------------------------------
      def self.randint(low, high, shape = [], dtype = nil, key = nil, stream = nil)
      end

      #-------------------------------------------------------------------
      # Generates Bernoulli random values in {0, 1}
      #
      # @param [Numeric or MLX::Core::Array] p (default: 0.5)
      # @param [Array<Integer>, nil] shape optional shape
      # @param [MLX::Core::Array, nil] key
      # @param [MLX::Stream, MLX::Device, nil] stream
      # @return [MLX::Core::Array]
      #-------------------------------------------------------------------
      def self.bernoulli(p = 0.5, shape = nil, key = nil, stream = nil)
      end

      #-------------------------------------------------------------------
      # Generates values from a truncated normal distribution
      #
      # @param [Numeric or MLX::Core::Array] lower
      # @param [Numeric or MLX::Core::Array] upper
      # @param [Array<Integer>, nil] shape optional
      # @param [Integer, nil] dtype (default: float32)
      # @param [MLX::Core::Array, nil] key
      # @param [MLX::Stream, MLX::Device, nil] stream
      # @return [MLX::Core::Array]
      #-------------------------------------------------------------------
      def self.truncated_normal(lower, upper, shape = nil, dtype = nil, key = nil, stream = nil)
      end

      #-------------------------------------------------------------------
      # Samples from a categorical distribution
      # (Accepts either shape or num_samples, but not both)
      #
      # @param [MLX::Core::Array] logits
      # @param [Integer] axis (default: -1)
      # @param [Array<Integer>, nil] shape optional
      # @param [Integer, nil] num_samples optional
      # @param [MLX::Core::Array, nil] key
      # @param [MLX::Stream, MLX::Device, nil] stream
      # @return [MLX::Core::Array]
      #-------------------------------------------------------------------
      def self.categorical(logits, axis = -1, shape = nil, num_samples = nil, key = nil, stream = nil)
      end

      #-------------------------------------------------------------------
      # Samples from the standard Gumbel distribution
      #
      # @param [Array<Integer>] shape (default: [])
      # @param [Integer, nil] dtype (default: float32)
      # @param [MLX::Core::Array, nil] key
      # @param [MLX::Stream, MLX::Device, nil] stream
      # @return [MLX::Core::Array]
      #-------------------------------------------------------------------
      def self.gumbel(shape = [], dtype = nil, key = nil, stream = nil)
      end

      #-------------------------------------------------------------------
      # Samples numbers from a Laplace distribution
      #
      # @param [Array<Integer>] shape (default: [])
      # @param [Integer, nil] dtype (default: float32)
      # @param [Float] loc (default: 0.0)
      # @param [Float] scale (default: 1.0)
      # @param [MLX::Core::Array, nil] key
      # @param [MLX::Stream, MLX::Device, nil] stream
      # @return [MLX::Core::Array]
      #-------------------------------------------------------------------
      def self.laplace(shape = [], dtype = nil, loc = 0.0, scale = 1.0, key = nil, stream = nil)
      end

      #-------------------------------------------------------------------
      # Generates a random permutation or permutes the entries of an array
      #
      # @param [Integer or MLX::Core::Array] x
      #   If x is an integer, returns a permutation of range(0, x).
      #   If x is an array, permutes it in-place (or returns new, see C++).
      # @param [Integer] axis (default: 0)
      # @param [MLX::Core::Array, nil] key
      # @param [MLX::Stream, MLX::Device, nil] stream
      # @return [MLX::Core::Array]
      #-------------------------------------------------------------------
      def self.permutation(x, axis = 0, key = nil, stream = nil)
      end
    end
    
    module Metal

      # :call-seq:
      #   Mlx::Metal.metal_is_available => true/false
      #
      # Check if the Metal back-end is available.
      def self.metal_is_available
        # ...
      end

      # :call-seq:
      #   Mlx::Metal.metal_get_active_memory => Integer
      #
      # (Deprecated in the Python bindings.)
      # Returns the current active memory usage.
      def self.metal_get_active_memory
        # ...
      end

      # :call-seq:
      #   Mlx::Metal.metal_get_peak_memory => Integer
      #
      # (Deprecated in the Python bindings.)
      # Returns the peak memory usage observed.
      def self.metal_get_peak_memory
        # ...
      end

      # :call-seq:
      #   Mlx::Metal.metal_reset_peak_memory => nil
      #
      # (Deprecated in the Python bindings.)
      # Resets the tracked peak memory usage to zero.
      def self.metal_reset_peak_memory
        # ...
      end

      # :call-seq:
      #   Mlx::Metal.metal_get_cache_memory => Integer
      #
      # (Deprecated in the Python bindings.)
      # Returns the GPU cache memory usage.
      def self.metal_get_cache_memory
        # ...
      end

      # :call-seq:
      #   Mlx::Metal.metal_set_memory_limit(limit) => Integer
      #
      # (Deprecated in the Python bindings.)
      # Sets the active memory limit in bytes. Returns the updated limit (or old limit, depending on internal logic).
      def self.metal_set_memory_limit(limit)
        # ...
      end

      # :call-seq:
      #   Mlx::Metal.metal_set_cache_limit(limit) => Integer
      #
      # (Deprecated in the Python bindings.)
      # Sets the GPU cache memory limit in bytes. Returns the updated limit.
      def self.metal_set_cache_limit(limit)
        # ...
      end

      # :call-seq:
      #   Mlx::Metal.metal_set_wired_limit(limit) => Integer
      #
      # (Deprecated in the Python bindings.)
      # Sets the “wired” memory limit in bytes. Returns the updated limit.
      def self.metal_set_wired_limit(limit)
        # ...
      end

      # :call-seq:
      #   Mlx::Metal.metal_clear_cache => nil
      #
      # (Deprecated in the Python bindings.)
      # Clears GPU memory caches.
      def self.metal_clear_cache
        # ...
      end

      # :call-seq:
      #   Mlx::Metal.start_metal_capture(path) => nil
      #
      # Start a Metal capture.
      #
      #   path - String pointing to a .gputrace file
      #
      # Used to capture GPU activity for debugging.
      def self.start_metal_capture(path)
        # ...
      end

      # :call-seq:
      #   Mlx::Metal.stop_metal_capture => nil
      #
      # Stop a Metal capture previously started by +start_metal_capture+.
      def self.stop_metal_capture
        # ...
      end

      # :call-seq:
      #   Mlx::Metal.metal_device_info => Hash
      #
      # Returns metadata about the GPU device and system settings.
      # The returned hash may contain:
      #   "architecture",
      #   "max_buffer_size",
      #   "max_recommended_working_set_size",
      #   "memory_size",
      #   "resource_limit".
      def self.metal_device_info
        # ...
      end
    end
    

    module Memory
      # Get the actively used memory in bytes.
      #
      # Note: This will not always match memory use reported by the system
      # because it excludes cached memory buffers.
      #
      # @return [Integer] number of bytes of actively used memory.
      def self.get_active_memory
        # Implemented in C++ via rb_define_module_function
      end
  
      # Get the peak amount of used memory in bytes.
      #
      # This is the maximum memory usage recorded since program start or since
      # reset_peak_memory was last called.
      #
      # @return [Integer] peak memory usage in bytes.
      def self.get_peak_memory
        # Implemented in C++ via rb_define_module_function
      end
  
      # Reset the tracked peak memory usage to zero.
      #
      # @return [nil]
      def self.reset_peak_memory
        # Implemented in C++ via rb_define_module_function
      end
  
      # Get the memory cache size in bytes.
      #
      # The cache is memory not currently used, but not yet returned to
      # the system allocator.
      #
      # @return [Integer] cached memory in bytes.
      def self.get_cache_memory
        # Implemented in C++ via rb_define_module_function
      end
  
      # Set the memory limit (in bytes) that the system should use as a guideline
      # for maximum memory usage during graph evaluation.
      #
      # If the memory limit is exceeded and the system (RAM+swap) cannot handle further
      # allocations, an exception will be raised.
      #
      # @param limit [Integer] the new memory limit in bytes
      # @return [Integer] the previous memory limit in bytes
      def self.set_memory_limit(limit)
        # Implemented in C++ via rb_define_module_function
      end
  
      # Set the free cache limit (in bytes).
      #
      # If the cache size exceeds this limit, free memory will be reclaimed
      # automatically. To disable caching altogether, set this to 0.
      #
      # @param limit [Integer] the new cache limit in bytes
      # @return [Integer] the previous cache limit in bytes
      def self.set_cache_limit(limit)
        # Implemented in C++ via rb_define_module_function
      end
  
      # Set the wired size limit (in bytes).
      #
      # This is only meaningful on macOS 15.0 or higher. The wired limit must
      # remain strictly below total system memory. Exceeding the system's own
      # wired limit is an error unless you have raised the system's limit via
      # e.g. `sudo sysctl iogpu.wired_limit_mb=<size>`.
      #
      # @param limit [Integer] the new wired limit in bytes
      # @return [Integer] the previous wired limit in bytes
      def self.set_wired_limit(limit)
        # Implemented in C++ via rb_define_module_function
      end
  
      # Clear the memory cache. After this call, get_cache_memory should return 0.
      #
      # @return [nil]
      def self.clear_cache
        # Implemented in C++ via rb_define_module_function
      end
    end

    module FFT
      # Perform a one-dimensional discrete Fourier Transform.
      #
      # @param a [MLX::Core::Array] The input array.
      # @param n [Integer, nil] Optional size of the transformed axis.
      # @param axis [Integer] Axis along which to perform the FFT.
      # @param stream [MLX::Stream, MLX::Device, nil] Optional stream/device.
      # @return [MLX::Core::Array]
      def self.fft(a, n = nil, axis = -1, stream = nil)
        # stub
      end

      # Perform a one-dimensional inverse discrete Fourier Transform.
      #
      # @param a [MLX::Core::Array]
      # @param n [Integer, nil]
      # @param axis [Integer]
      # @param stream [MLX::Stream, MLX::Device, nil]
      # @return [MLX::Core::Array]
      def self.ifft(a, n = nil, axis = -1, stream = nil)
        # stub
      end

      # Perform a one-dimensional real discrete Fourier Transform.
      #
      # @param a [MLX::Core::Array] Real input array (complex is cast to real).
      # @param n [Integer, nil]
      # @param axis [Integer]
      # @param stream [MLX::Stream, MLX::Device, nil]
      # @return [MLX::Core::Array] Complex-valued output.
      def self.rfft(a, n = nil, axis = -1, stream = nil)
        # stub
      end

      # Perform the inverse of rfft.
      #
      # @param a [MLX::Core::Array] Usually complex input array.
      # @param n [Integer, nil]
      # @param axis [Integer]
      # @param stream [MLX::Stream, MLX::Device, nil]
      # @return [MLX::Core::Array] Real-valued result.
      def self.irfft(a, n = nil, axis = -1, stream = nil)
        # stub
      end

      # Perform a two-dimensional discrete Fourier Transform.
      #
      # @param a [MLX::Core::Array]
      # @param s [Array<Integer>, nil] The shape of each transform dimension.
      # @param axes [Array<Integer>, nil] The axes along which to compute the transform.
      # @param stream [MLX::Stream, MLX::Device, nil]
      # @return [MLX::Core::Array]
      def self.fft2(a, s = nil, axes = nil, stream = nil)
        # stub
      end

      # Perform a two-dimensional inverse discrete Fourier Transform.
      #
      # @param a [MLX::Core::Array]
      # @param s [Array<Integer>, nil]
      # @param axes [Array<Integer>, nil]
      # @param stream [MLX::Stream, MLX::Device, nil]
      # @return [MLX::Core::Array]
      def self.ifft2(a, s = nil, axes = nil, stream = nil)
        # stub
      end

      # Perform a two-dimensional real discrete Fourier Transform.
      #
      # @param a [MLX::Core::Array]
      # @param s [Array<Integer>, nil]
      # @param axes [Array<Integer>, nil]
      # @param stream [MLX::Stream, MLX::Device, nil]
      # @return [MLX::Core::Array] Complex output
      def self.rfft2(a, s = nil, axes = nil, stream = nil)
        # stub
      end

      # Perform the inverse of rfft2.
      #
      # @param a [MLX::Core::Array]
      # @param s [Array<Integer>, nil]
      # @param axes [Array<Integer>, nil]
      # @param stream [MLX::Stream, MLX::Device, nil]
      # @return [MLX::Core::Array] Real output
      def self.irfft2(a, s = nil, axes = nil, stream = nil)
        # stub
      end

      # Perform an n-dimensional discrete Fourier Transform.
      #
      # @param a [MLX::Core::Array]
      # @param s [Array<Integer>, nil]
      # @param axes [Array<Integer>, nil]
      # @param stream [MLX::Stream, MLX::Device, nil]
      # @return [MLX::Core::Array]
      def self.fftn(a, s = nil, axes = nil, stream = nil)
        # stub
      end

      # Perform an n-dimensional inverse discrete Fourier Transform.
      #
      # @param a [MLX::Core::Array]
      # @param s [Array<Integer>, nil]
      # @param axes [Array<Integer>, nil]
      # @param stream [MLX::Stream, MLX::Device, nil]
      # @return [MLX::Core::Array]
      def self.ifftn(a, s = nil, axes = nil, stream = nil)
        # stub
      end

      # Perform an n-dimensional real discrete Fourier Transform.
      #
      # @param a [MLX::Core::Array]
      # @param s [Array<Integer>, nil]
      # @param axes [Array<Integer>, nil]
      # @param stream [MLX::Stream, MLX::Device, nil]
      # @return [MLX::Core::Array] Complex output
      def self.rfftn(a, s = nil, axes = nil, stream = nil)
        # stub
      end

      # Perform the inverse of rfftn.
      #
      # @param a [MLX::Core::Array]
      # @param s [Array<Integer>, nil]
      # @param axes [Array<Integer>, nil]
      # @param stream [MLX::Stream, MLX::Device, nil]
      # @return [MLX::Core::Array] Real output
      def self.irfftn(a, s = nil, axes = nil, stream = nil)
        # stub
      end
    end
    
    # Fast module - implemented in fast.cpp
    # 
    module Fast
      # Stubs for public module functions

      # Not implemented: raises NotImpError
      def self.gemm(*args)
        # ...
      end

      # scaled_dot_product_attention(q, k, v, scale, mask=nil, memory_efficient_threshold=nil, stream=nil)
      def self.scaled_dot_product_attention(q, k, v, scale, mask=nil, memory_efficient_threshold=nil, stream=nil)
        # ...
      end

      # Not implemented: raises NotImpError
      def self.multi_head_attention(*args)
        # ...
      end

      # rms_norm(x, weight=nil, eps, stream=nil)
      def self.rms_norm(x, weight=nil, eps=nil, stream=nil)
        # ...
      end

      # layer_norm(x, weight=nil, bias=nil, eps, stream=nil)
      def self.layer_norm(x, weight=nil, bias=nil, eps=nil, stream=nil)
        # ...
      end

      # rope(a, dims, traditional, base, scale, offset, freqs=nil, stream=nil)
      def self.rope(a, dims, traditional, base, scale, offset, freqs=nil, stream=nil)
        # ...
      end

      # Not implemented: raises NotImpError
      def self.rope_inplace(*args)
        # ...
      end

      # metal_kernel(...) => returns a Proc
      def self.metal_kernel(name, input_names, output_names, source, header="", ensure_row_contiguous=true, atomic_outputs=false)
        # This returns a Proc (the "kernel"). You call it like:
        #
        #   kernel.call(
        #     inputs: ..., output_shapes: ..., output_dtypes: ...,
        #     grid: ..., threadgroup: ...,
        #     template: ..., init_value: ..., verbose: ...,
        #     stream: ...
        #   )
        #
      end
    end
    
    module Indexing 
      # Takes elements along a given axis using integer or array indices.
      # @param arr [MLX::Core::Array]
      # @param indices [MLX::Core::Array]
      # @param axis [Integer, nil]
      # @return [MLX::Core::Array]
      def self.take(arr, indices, axis); end
  
      # Takes elements along a given axis with broadcasting logic for indices.
      # @param arr [MLX::Core::Array]
      # @param indices [MLX::Core::Array]
      # @param axis [Integer, nil]
      # @return [MLX::Core::Array]
      def self.take_along_axis(arr, indices, axis); end
  
      # Retrieves a slice [start:stop:step] along the first dimension.
      # @param arr [MLX::Core::Array]
      # @param start [Integer, nil]
      # @param stop [Integer, nil]
      # @param step [Integer, nil]
      # @return [MLX::Core::Array]
      def self.slice(arr, start, stop, step); end
  
      # Gathers elements by array-of-indices or single index array.
      # @param arr [MLX::Core::Array]
      # @param indices [MLX::Core::Array or Array of MLX::Core::Array]
      # @return [MLX::Core::Array]
      def self.index(arr, indices); end
  
      # Performs a dynamic slice: gather using an array of start indices + slice sizes.
      # @param arr [MLX::Core::Array]
      # @param start_indices [Array of MLX::Core::Array]
      # @param slice_sizes [Array<Integer>]
      # @return [MLX::Core::Array]
      def self.dynamic_slice(arr, start_indices, slice_sizes); end
  
      # Scatter: sets elements at given indices to the 'updates' array's values.
      # @param arr [MLX::Core::Array]
      # @param indices [MLX::Core::Array]
      # @param updates [MLX::Core::Array]
      # @param axis [Integer, nil]
      # @return [MLX::Core::Array] (the updated array)
      def self.scatter(arr, indices, updates, axis); end
  
      # Scatter-add: adds the 'updates' array's values into arr at given indices.
      # @param arr [MLX::Core::Array]
      # @param indices [MLX::Core::Array]
      # @param updates [MLX::Core::Array]
      # @param axis [Integer, nil]
      # @return [MLX::Core::Array]
      def self.scatter_add(arr, indices, updates, axis); end
  
      # Scatter-prod: multiplies the 'updates' array's values into arr at indices.
      # @param arr [MLX::Core::Array]
      # @param indices [MLX::Core::Array]
      # @param updates [MLX::Core::Array]
      # @param axis [Integer, nil]
      # @return [MLX::Core::Array]
      def self.scatter_prod(arr, indices, updates, axis); end
  
      # Scatter-max: elementwise maximum with updates at given indices.
      # @param arr [MLX::Core::Array]
      # @param indices [MLX::Core::Array]
      # @param updates [MLX::Core::Array]
      # @param axis [Integer, nil]
      # @return [MLX::Core::Array]
      def self.scatter_max(arr, indices, updates, axis); end
  
      # Scatter-min: elementwise minimum with updates at given indices.
      # @param arr [MLX::Core::Array]
      # @param indices [MLX::Core::Array]
      # @param updates [MLX::Core::Array]
      # @param axis [Integer, nil]
      # @return [MLX::Core::Array]
      def self.scatter_min(arr, indices, updates, axis); end
  
      # Gathers from arr at multiple index arrays and slice sizes, returning a new array.
      # @param arr [MLX::Core::Array]
      # @param indices [Array of MLX::Core::Array]
      # @param axes [Array<Integer>]
      # @param slice_sizes [Array<Integer>]
      # @return [MLX::Core::Array]
      def self.gather(arr, indices, axes, slice_sizes); end
  
      # put_along_axis: in-place put at positions defined by indices along a given axis.
      # @param arr [MLX::Core::Array]
      # @param indices [MLX::Core::Array]
      # @param values [MLX::Core::Array]
      # @param axis [Integer, nil]
      # @return [MLX::Core::Array]
      def self.put_along_axis(arr, indices, values, axis); end
    end
    

    module Transforms
  
      # reshape(a, shape, stream=nil) -> MLX::Core::Array
      #   Reshapes array 'a' to 'shape', optionally using a given stream.
      def self.reshape(a, shape, stream=nil)
        # ...
      end
  
      # transpose(a, axes=nil, stream=nil) -> MLX::Core::Array
      #   Transposes array 'a' (optionally with a permutation 'axes').
      def self.transpose(a, axes=nil, stream=nil)
        # ...
      end
  
      # squeeze(a, axes=nil, stream=nil) -> MLX::Core::Array
      #   Removes single-dimensional entries from 'a' (optionally a subset of axes).
      def self.squeeze(a, axes=nil, stream=nil)
        # ...
      end
  
      # expand_dims(a, axis, stream=nil) -> MLX::Core::Array
      #   Expands the shape of 'a' by inserting a dimension at 'axis'.
      def self.expand_dims(a, axis, stream=nil)
        # ...
      end
  
      # broadcast_to(a, shape, stream=nil) -> MLX::Core::Array
      #   Broadcasts 'a' to the given 'shape'.
      def self.broadcast_to(a, shape, stream=nil)
        # ...
      end
  
      # pad(a, pad_width, value=nil, stream=nil) -> MLX::Core::Array
      #   Pads array 'a' with 'value' according to the list of (low, high) pairs in 'pad_width'.
      def self.pad(a, pad_width, value=nil, stream=nil)
        # ...
      end
  
      # split(a, num_or_indices, axis=0, stream=nil) -> Array<MLX::Core::Array>
      #   Splits 'a' into multiple sub-arrays along 'axis'.
      #   If num_or_indices is an int, splits into that many equal parts.
      #   Otherwise num_or_indices is an array of split indices.
      def self.split(a, num_or_indices, axis=0, stream=nil)
        # ...
      end
  
      # concatenate(arrays, axis=0, stream=nil) -> MLX::Core::Array
      #   Concatenates a list of arrays along 'axis'.
      def self.concatenate(arrays, axis=0, stream=nil)
        # ...
      end
  
      # stack(arrays, axis=0, stream=nil) -> MLX::Core::Array
      #   Stacks a list of arrays along a new dimension 'axis'.
      def self.stack(arrays, axis=0, stream=nil)
        # ...
      end
  
      # tile(a, repeats, stream=nil) -> MLX::Core::Array
      #   Constructs an array by tiling 'a' according to 'repeats'.
      def self.tile(a, repeats, stream=nil)
        # ...
      end
  
      # repeat(a, repeats, axis=nil, stream=nil) -> MLX::Core::Array
      #   Repeats elements of 'a' 'repeats' times along 'axis'.
      def self.repeat(a, repeats, axis=nil, stream=nil)
        # ...
      end
  
      # moveaxis(a, source, destination, stream=nil) -> MLX::Core::Array
      #   Moves axis (or axes) in 'a' from 'source' to 'destination'.
      #   Either both single ints or both arrays of ints.
      def self.moveaxis(a, source, destination, stream=nil)
        # ...
      end
  
      # checkpoint(fun) -> ?
      #   Creates a checkpointed version of 'fun' (stub).
      def self.checkpoint(fun)
        # ...
      end
  
      # value_and_grad(fun, argnums=nil, argnames=nil) -> callable
      #   Returns a function that computes [fun(...) result, gradient].
      def self.value_and_grad(fun, argnums=nil, argnames=nil)
        # ...
      end
  
      # grad(fun, argnums=nil, argnames=nil) -> callable
      #   Returns a function that computes only the gradient of fun(...).
      def self.grad(fun, argnums=nil, argnames=nil)
        # ...
      end
  
      # jvp(fun, primals, tangents) -> [primal_out, tangent_out]
      #   Jacobian-vector product stub.
      def self.jvp(fun, primals, tangents)
        # ...
      end
  
      # vjp(fun, primals, cotangents) -> [primal_out, cotangent_out]
      #   Vector-Jacobian product stub.
      def self.vjp(fun, primals, cotangents)
        # ...
      end
  
      # vmap(fun, in_axes=0, out_axes=0) -> callable
      #   Vectorizes fun over specified axes.
      def self.vmap(fun, in_axes=0, out_axes=0)
        # ...
      end
  
      # compile(fun, inputs=nil, outputs=nil, shapeless=false) -> callable
      #   Compiles 'fun' with optional captured inputs/outputs.
      def self.compile(fun, inputs=nil, outputs=nil, shapeless=false)
        # ...
      end
  
      # disable_compile() -> nil
      #   Globally disable compilation.
      def self.disable_compile
        # ...
      end
  
      # enable_compile() -> nil
      #   Globally enable compilation.
      def self.enable_compile
        # ...
      end
  
      # stop_gradient(a, stream=nil) -> MLX::Core::Array
      #   Marks array 'a' so that it no longer accumulates gradients.
      def self.stop_gradient(a, stream=nil)
        # ...
      end
  
      # eval(*arrays) -> The same array(s)
      #   Eagerly evaluate one or more MLX::Core::Array objects in place.
      def self.eval(*arrays)
        # ...
      end
  
      # eval_batch(arrays) -> arrays
      #   Like eval() but takes an array of arrays.
      def self.eval_batch(arrays)
        # ...
      end
  
      # async_eval(*arrays) -> The same array(s)
      #   Asynchronously evaluate one or more MLX::Core::Array objects.
      def self.async_eval(*arrays)
        # ...
      end
    end
    
    module Trees
      # Flattens all arrays (MLX::Core::Array objects) out of 'tree' into an Array,
      # raising an error if it encounters non-array leaves and 'strict' is true.
      # @param tree [Array, Hash, MLX::Core::Array, Object] Any nested combination
      # @param strict [Boolean] if true, non-MLX::Core::Array leaves raise an error.
      # @return [Array<MLX::Core::Array>] arrays extracted from 'tree'
      def self.tree_flatten_arrays(tree, strict = true)
        # stub
      end
    
      # Same as #tree_flatten_arrays but with a required signature:
      # This is just an alias (under the hood) but can be restricted to a single param.
      # @param tree [Array, Hash, MLX::Core::Array, Object]
      # @return [Array<MLX::Core::Array>]
      def self.tree_flatten(tree)
        # stub
      end
    
      # Unflattens an array of MLX::Core::Array objects into 'tree',
      # replacing any MLX::Core::Array leaves in the original structure.
      # Optional index (default 0) determines which element of 'values' to use first.
      # @param tree [Array, Hash, MLX::Core::Array, Object]
      # @param values [Array<MLX::Core::Array>] the flattened arrays
      # @param index [Integer] starting index in 'values'
      # @return [Array, Hash, MLX::Core::Array, Object] the reconstructed tree
      def self.tree_unflatten(tree, values, index = 0)
        # stub
      end
    
      # Maps a given callable over each leaf of a single 'tree'.
      # @param tree [Array, Hash, MLX::Core::Array, Object]
      # @param func [#call] a callable object expecting one argument: a leaf
      # @return [Array, Hash, MLX::Core::Array, Object] the transformed tree
      def self.tree_map(tree, func)
        # stub
      end
    
      # Fills every MLX::Core::Array leaf in 'tree' with arrays from 'values', in order.
      # Mutates the original 'tree' structure in place (though it returns it).
      # @param tree [Array, Hash, MLX::Core::Array, Object]
      # @param values [Array<MLX::Core::Array>] arrays to fill in
      # @return [Array, Hash, MLX::Core::Array, Object] same tree structure with replaced leaves
      def self.tree_fill(tree, values)
        # stub
      end
    
      # Replaces arrays in 'tree' that match any in 'src_values' with corresponding items in 'dst_values'.
      # @param tree [Array, Hash, MLX::Core::Array, Object]
      # @param src_values [Array<MLX::Core::Array>]
      # @param dst_values [Array<MLX::Core::Array>]
      # @return [Array, Hash, MLX::Core::Array, Object] the updated tree
      def self.tree_replace(tree, src_values, dst_values)
        # stub
      end
    
      # Flattens a tree but also replaces each array leaf with a sentinel.
      # Returns a two-element result: [ flattened_arrays, structure_with_sentinels ].
      # @param tree [Array, Hash, MLX::Core::Array, Object]
      # @param strict [Boolean] if true, raise on non-array leaves
      # @return [Array<MLX::Core::Array>, (Array|Hash|Object)] array of leaf arrays, plus sentinel-marked structure
      def self.tree_flatten_with_structure(tree, strict = true)
        # stub
      end
    
      # The inverse of #tree_flatten_with_structure. Takes a structure (with sentinels) and
      # an array of MLX::Core::Array objects. Replaces each sentinel with the next array in 'values'.
      # @param structure [Array, Hash, Object] the sentinel-marked structure
      # @param values [Array<MLX::Core::Array>] arrays to plug in
      # @param index [Integer] optional start index (default 0)
      # @return [Array, Hash, Object] structure with arrays re-inserted
      def self.tree_unflatten_from_structure(structure, values, index = 0)
        # stub
      end
    
      # The multi-tree variant of #tree_map. Takes an array of trees and a callable.
      # Recursively processes them in lockstep, passing an array of leaves to 'func'.
      # @param trees [Array] each element is a tree (Array, Hash, etc.)
      # @param func [#call] a callable object expecting one array argument (the parallel leaves)
      # @return [Array, Hash, Object] new structure derived from the first tree
      def self.tree_map_multi(trees, func)
        # stub
      end
    
      # The multi-tree variant of #tree_visit. Takes an array of trees and a callable.
      # Recursively visits leaves in lockstep, calling 'func' at each leaf with an array of values.
      # No transformed structure is returned (i.e., no structural modifications).
      # @param trees [Array] each element is a tree (Array, Hash, etc.)
      # @param func [#call] a callable object expecting an array (the parallel leaves)
      # @return [nil]
      def self.tree_visit_multi(trees, func)
        # stub
      end
    end
    

    module Convert

      # --- Methods directly visible in the C++ snippet ---

      # to_float16(mlx_arr) => MLX::Core::Array
      # Converts an existing MLX::Core::Array object to float16 dtype
      def self.to_float16(mlx_arr)
        # stub
      end

      # to_float32(mlx_arr) => MLX::Core::Array
      # Converts an existing MLX::Core::Array object to float32 dtype
      def self.to_float32(mlx_arr)
        # stub
      end

      # to_int32(mlx_arr) => MLX::Core::Array
      # Converts an existing MLX::Core::Array object to int32 dtype
      def self.to_int32(mlx_arr)
        # stub
      end

      # to_bool(mlx_arr) => MLX::Core::Array
      # Converts an existing MLX::Core::Array object to bool dtype
      def self.to_bool(mlx_arr)
        # stub
      end

      # to_type(mlx_arr, dtype_int) => MLX::Core::Array
      # Converts an existing MLX::Core::Array to a specified dtype code.
      # dtype_int is an integer tag (0 => bool, 1 => uint8, etc.)
      def self.to_type(mlx_arr, dtype_int)
        # stub
      end


      # --- Methods missing from the original snippet but required for parity
      #     with Python’s interface (if you include the patch or want full parity) ---

      # to_scalar(mlx_arr) => Boolean | Integer | Float | Complex
      # Converts a length-1 array to a simple Ruby scalar.
      def self.to_scalar(mlx_arr)
        # stub
      end

      # to_list(mlx_arr) => Array (potentially nested)
      # Recursively converts an mx::array into nested Ruby arrays (like Python tolist()).
      def self.to_list(mlx_arr)
        # stub
      end

      # ruby_obj_to_mx(obj) => MLX::Core::Array
      # Converts Ruby objects (numeric, arrays, or MLX::Core::Array) into a
      # fresh MLX::Core::Array, inferring shape/dtype if necessary.
      def self.ruby_obj_to_mx(obj)
        # stub
      end

      # If you decide to add dlpack or other bridging, you could have:
      # def self.to_dlpack(mlx_arr)
      #   # Not currently implemented in the snippet, but would match Python’s mlx_to_dlpack
      # end

    end
    
    module Load
      # All of these are defined as module functions via rb_define_module_function.
      # The stubs here have the same signatures (argument counts, defaults) as
      # the C++ binding code.

      # load(file, format=nil, return_metadata=false, stream=nil) => Hash or [Hash, Hash]
      #
      # In Python:
      #   load(file, format=None, return_metadata=False, s=None)
      #
      # Behavior: autodetect or use the specified format. If return_metadata is true
      # for safetensors/gguf, returns [tensor_map, metadata_map].
      # If false, returns just the tensor_map.
      def self.load(file, format = nil, return_metadata = false, stream = nil)
        # ...
      end

      # load_shard(...)
      def self.load_shard(file, index, total, format = nil)
        # Stub only, unimplemented in your snippet
        # ...
      end

      # save(file, array, stream=nil)
      # In Python it picks format by extension or explicitly. In the snippet,
      # it's currently a stub raising NotImplementedError.
      def self.save(file, array, stream = nil)
        # ...
      end

      # save_shard(...)
      def self.save_shard(file, index, total, array, stream = nil)
        # Stub only
        # ...
      end

      # load_safetensors(file, stream=nil) => [Hash<String, MLX::Core::Array>, Hash<String, String>]
      #
      # Returns a two-element array: [tensors_hash, metadata_hash].
      def self.load_safetensors(file, stream = nil)
        # ...
      end

      # save_safetensors(file, tensors_hash, metadata_hash=nil, stream=nil)
      #
      # In Python:
      #   save_safetensors(file, dict_of_arrays, metadata=None)
      def self.save_safetensors(file, tensors_hash, metadata_hash = nil, stream = nil)
        # ...
      end

      # load_gguf(file, stream=nil) => [Hash<String, MLX::Core::Array>, Hash<String, String>]
      #
      # In Python:
      #   load_gguf(file, s=None) => (weights, metadata)
      def self.load_gguf(file, stream = nil)
        # ...
      end

      # save_gguf(file, tensors_hash, metadata_hash=nil, stream=nil)
      #
      # In Python:
      #   save_gguf(file, dict_of_arrays, metadata=None)
      def self.save_gguf(file, tensors_hash, metadata_hash = nil, stream = nil)
        # ...
      end

      # load_npy(file, stream=nil) => MLX::Core::Array
      def self.load_npy(file, stream = nil)
        # ...
      end

      # save_npy(file, array, stream=nil)
      def self.save_npy(file, array, stream = nil)
        # ...
      end

      # load_npz(file, stream=nil) => Hash<String, MLX::Core::Array>
      def self.load_npz(file, stream = nil)
        # ...
      end

      # savez(file, *arrays, **named_arrays) => nil
      #
      # In Python, e.g.:
      #   savez(file, arr1, arr2, named_arr: arr3, ...)
      def self.savez(file, *positional_arrays, **keyword_arrays)
        # ...
      end

      # savez_compressed(file, *arrays, **named_arrays) => nil
      def self.savez_compressed(file, *positional_arrays, **keyword_arrays)
        # ...
      end
    end
    
    module Utils
      # create_stream_context(stream_or_device = nil) => StreamContext
      # Creates a new StreamContext instance wrapping either a Stream,
      # a Device, or nothing (defaults).
      def self.create_stream_context(stream_or_device = nil)
        # Native function: rb_create_stream_context
      end

      # tree_flatten(arr) => Array
      # In the C++ snippet, this simply returns [arr] as a trivial flatten.
      def self.tree_flatten(arr)
        # Native function: utils_tree_flatten
      end

      # is_array_like(obj) => true or false
      # Checks if 'obj' is an MLX::Core::Array (in the snippet).
      def self.is_array_like(obj)
        # Native function: utils_is_array_like
      end

      # is_pytree_leaf(obj) => true or false
      # Same as is_array_like in the snippet.
      def self.is_pytree_leaf(obj)
        # Native function: utils_is_pytree_leaf
      end

      # dtype_to_string(dtype) => String
      # Converts an integer dtype code into a text representation (e.g. "float32").
      def self.dtype_to_string(dtype)
        # Native function: utils_dtype_to_string
      end

      # size_to_string(size_in_bytes) => String
      # Converts a byte size into a human-readable string: "KB", "MB", "GB" etc.
      def self.size_to_string(size_in_bytes)
        # Native function: utils_size_to_string
      end

      # eval_counter() => Integer
      # A placeholder in your snippet returning 0. Could eventually return
      # the "mx::eval_count()".
      def self.eval_counter
        # Native function: utils_eval_counter
      end

      # issubdtype(dtype1, dtype2) => true or false
      # Returns whether dtype1 is a sub-dtype of dtype2 (e.g. int32 < int64).
      def self.issubdtype(dtype1, dtype2)
        # Native function: utils_issubdtype
      end

      # promote_types(dtype1, dtype2) => Integer (the promoted dtype enum)
      # Returns the result of mx::promote_types on the two dtype arguments.
      def self.promote_types(dtype1, dtype2)
        # Native function: utils_promote_types
      end
    end
    
    # Linalg module - implemented in linalg.cpp
    #
    module Linalg

      # norm(a, ord=nil, axis=nil, keepdims=false, stream=nil)
      def self.norm(a, ord = nil, axis = nil, keepdims = false, stream = nil)
        # => returns MLX::Core::Array
      end

      # qr(a, stream=nil)
      def self.qr(a, stream = nil)
        # => returns [Q, R], where Q and R are MLX::Core::Array
      end

      # svd(a, compute_uv=true, stream=nil)
      def self.svd(a, compute_uv = true, stream = nil)
        # => if compute_uv == false: returns S
        # => else: returns [U, S, Vt]
      end

      # inv(a, stream=nil)
      def self.inv(a, stream = nil)
        # => returns the inverse (MLX::Core::Array)
      end

      # tri_inv(a, upper=false, stream=nil)
      def self.tri_inv(a, upper = false, stream = nil)
        # => returns inverse of triangular input (MLX::Core::Array)
      end

      # cholesky(a, upper=false, stream=nil)
      def self.cholesky(a, upper = false, stream = nil)
        # => returns Cholesky factor (MLX::Core::Array)
      end

      # cholesky_inv(a, upper=false, stream=nil)
      def self.cholesky_inv(a, upper = false, stream = nil)
        # => returns inverse from Cholesky factor (MLX::Core::Array)
      end

      # pinv(a, stream=nil)
      def self.pinv(a, stream = nil)
        # => returns Moore-Penrose pseudoinverse (MLX::Core::Array)
      end

      # cross(a, b, axis=-1, stream=nil)
      def self.cross(a, b, axis = -1, stream = nil)
        # => returns cross product (MLX::Core::Array)
      end

      # eigvalsh(a, uplo='L', stream=nil)
      def self.eigvalsh(a, uplo = 'L', stream = nil)
        # => returns eigenvalues (MLX::Core::Array)
      end

      # eigh(a, uplo='L', stream=nil)
      def self.eigh(a, uplo = 'L', stream = nil)
        # => returns [eigenvalues, eigenvectors]
      end

      # lu(a, stream=nil)
      def self.lu(a, stream = nil)
        # => returns [P, L, U]
      end

      # lu_factor(a, stream=nil)
      def self.lu_factor(a, stream = nil)
        # => returns [LU, pivots]
      end

      # solve(a, b, stream=nil)
      def self.solve(a, b, stream = nil)
        # => returns solution of AX = B (MLX::Core::Array)
      end

      # solve_triangular(a, b, upper=false, stream=nil)
      def self.solve_triangular(a, b, upper = false, stream = nil)
        # => returns solution of triangular system (MLX::Core::Array)
      end

      # matmul(a, b, stream=nil)
      def self.matmul(a, b, stream = nil)
        # => matrix multiplication (MLX::Core::Array)
      end

      # det(a, stream=nil)
      def self.det(a, stream = nil)
        # => returns determinant (MLX::Core::Array with scalar shape)
      end

      # slogdet(a, stream=nil)
      def self.slogdet(a, stream = nil)
        # => returns [sign, log|det|]
      end

      # matrix_power(a, n, stream=nil)
      def self.matrix_power(a, n, stream = nil)
        # => returns A^n
      end

    end

    module Export

      #----------------------------------------------------------------
      # 1) Safetensors / GGUF / DOT Export Methods
      #----------------------------------------------------------------

      # (weights: Hash<String, MLX::Core::Array>, path: String, metadata: Hash<String,String> or nil)
      #
      # Saves weights to a .safetensors file. Metadata is optional.
      def self.to_safetensors(weights, path, metadata = nil)
        # native C++ logic
      end

      # (weights: Hash<String, MLX::Core::Array>, path: String, metadata: Hash<String,String> or nil)
      #
      # Saves weights to a .gguf file. Metadata currently not implemented.
      def self.to_gguf(weights, path, metadata = nil)
        # native C++ logic
      end

      # Variadic arguments: (file, *arrays_and/or_hashes)
      #
      # Exports the provided arrays (and named arrays in Hash) to a DOT file for graph visualization.
      def self.to_dot(*args)
        # native C++ logic
      end

      # Alias so the method name matches the Python version exactly
      class << self
        alias_method :export_to_dot, :to_dot
      end

      #----------------------------------------------------------------
      # 2) export_function
      #----------------------------------------------------------------

      # (file: String, fun: #call, *args, shapeless: false, **kwargs)
      #
      # Exports a user-provided Ruby callable to disk along with example arrays.
      # The exported function can later be imported with `import_function`.
      def self.export_function(file, fun, *args, shapeless: false, **kwargs)
        # native C++ logic
      end

      #----------------------------------------------------------------
      # 3) import_function
      #----------------------------------------------------------------

      # (file: String) -> #call
      #
      # Loads a function from disk (previously exported via `export_function`).
      # Returns a Ruby callable (Proc) that you can call with arrays.
      def self.import_function(file)
        # native C++ logic
        # returns something responding to #call
      end

      #----------------------------------------------------------------
      # 4) exporter
      #----------------------------------------------------------------

      # (file: String, fun: #call, shapeless: false) -> FunctionExporter
      #
      # Returns a FunctionExporter object that can record multiple “traces”
      # (call patterns) of the function and save them all to a single file.
      def self.exporter(file, fun, shapeless: false)
        # returns a new FunctionExporter instance
      end

      #================================================================
      # 5) Class: FunctionExporter
      #================================================================
      #     The "context manager" style class that allows multiple calls
      #     to be recorded/serialized. In Python we had __enter__, __exit__
      #     and __call__. Ruby version offers analogous usage.
      #================================================================
      class FunctionExporter
        # Constructor:
        #
        #   initialize(file, fun, shapeless: false)
        #
        # Typically called via `MLX::Export.exporter(file, fun, shapeless: false)`.
        def initialize(file, fun, shapeless: false)
          # native C++ logic
        end

        # Closes the exporter, finalizing the export file.
        def close
          # native C++ logic
        end

        # Called by `exporter_obj.call(...)`.  
        # Accepts a flexible array of arguments and optional last-hash as kwargs.
        def call(*args, **kwargs)
          # native C++ logic
        end

        # Pythonic “with” pattern analog. For convenience, we name them #enter/#exit:
        #   exporter_obj.enter -> returns self
        def enter
          # native C++ logic
        end

        #   exporter_obj.exit(exc_type = nil, exc_val = nil, exc_tb = nil) -> nil
        # Usually calls #close.
        def exit(exc_type = nil, exc_val = nil, exc_tb = nil)
          # native C++ logic
        end
      end
    end
    
    module Distributed

      #-----------------------------------------------------------------------------
      # Class: MLX::Core::Distributed::Group
      #-----------------------------------------------------------------------------
      class Group
        # Although this constructor is technically defined in C++,
        # calling it directly will raise an error. Creation of Group
        # instances happens via MLX::Core::Distributed.init, or
        # split from an existing Group.
        def initialize(size, rank)
          # Raises RuntimeError in the actual binding
        end

        # Returns the rank of the current process in the Group.
        # @return [Integer]
        def rank
          # C++: Group::rank()
        end

        # Returns the total number of processes in the Group.
        # @return [Integer]
        def size
          # C++: Group::size()
        end

        # Splits the Group into subgroups based on color. The key determines
        # the rank ordering in the new group. If the key is -1 or nil, it
        # uses the original rank ordering.
        #
        # @param color [Integer]
        # @param key [Integer] optional, defaults to -1
        # @return [MLX::Core::Distributed::Group]
        def split(color, key = -1)
          # C++: Group::split(color, key)
        end
      end


      #-----------------------------------------------------------------------------
      # Module Methods: MLX::Core::Distributed
      #-----------------------------------------------------------------------------
      
      # Checks if a distributed communication backend is available.
      # @return [Boolean]
      def self.is_available
        # C++: bool mx::distributed::is_available()
      end

      # Initializes the communication backend. Returns the global Group.
      #
      # @param strict [Boolean] (default: false)
      # @param backend [String] (default: "any")
      # @return [MLX::Core::Distributed::Group]
      def self.init(strict = false, backend = "any")
        # C++: mx::distributed::init(strict, backend)
      end

      # All-reduce sum across all processes in the group.
      #
      # @param x [Numeric, MLX::Core::Array] The input array or scalar.
      # @param group [MLX::Core::Distributed::Group, nil] optional
      # @param stream [MLX::Stream, MLX::Device, nil] optional
      # @return [MLX::Core::Array]
      def self.all_sum(x, group = nil, stream = nil)
        # C++: mx::distributed::all_sum(...)
      end

      # Gathers arrays from all processes in the group along the first axis.
      #
      # @param x [Numeric, MLX::Core::Array]
      # @param group [MLX::Core::Distributed::Group, nil] optional
      # @param stream [MLX::Stream, MLX::Device, nil] optional
      # @return [MLX::Core::Array]
      def self.all_gather(x, group = nil, stream = nil)
        # C++: mx::distributed::all_gather(...)
      end

      # Sends an array (or scalar) to the process with rank `dst`.
      #
      # @param x [Numeric, MLX::Core::Array]
      # @param dst [Integer] The destination rank
      # @param group [MLX::Core::Distributed::Group, nil] optional
      # @param stream [MLX::Stream, MLX::Device, nil] optional
      # @return [MLX::Core::Array] (returns an array that triggers send on evaluation)
      def self.send(x, dst, group = nil, stream = nil)
        # C++: mx::distributed::send(...)
      end

      # Receives an array from the process with rank `src`, creating a new array
      # with the given shape and dtype.
      #
      # @param shape [Array<Integer>] The shape of the array to receive
      # @param dtype [Integer] The dtype code (e.g. 10 = float32)
      # @param src [Integer] The source rank
      # @param group [MLX::Core::Distributed::Group, nil] optional
      # @param stream [MLX::Stream, MLX::Device, nil] optional
      # @return [MLX::Core::Array]
      def self.recv(shape, dtype, src, group = nil, stream = nil)
        # C++: mx::distributed::recv(...)
      end

      # Receives an array from the process with rank `src`, using the shape and
      # dtype of `x`.
      #
      # @param x [Numeric, MLX::Core::Array] Defines shape/dtype if it's an array
      # @param src [Integer] The source rank
      # @param group [MLX::Core::Distributed::Group, nil] optional
      # @param stream [MLX::Stream, MLX::Device, nil] optional
      # @return [MLX::Core::Array]
      def self.recv_like(x, src, group = nil, stream = nil)
        # C++: mx::distributed::recv_like(...)
      end

    end

# lib/mlx/core/stream.rb (conceptual stubs)


    # A stream for running operations on a given device.
    class Stream
      # A module function that returns the device's default stream.
      # @param device [MLX::Core::Device, Integer] device or device-type enum
      # @return [MLX::Core::Stream]
      def self.default_stream(device); end

      # A module function that sets the default stream on the device it belongs to.
      # @param stream [MLX::Core::Stream]
      # @return [void]
      def self.set_default_stream(stream); end

      # A module function that creates and returns a new stream on the given device.
      # @param device [MLX::Core::Device, Integer] device or device-type enum
      # @return [MLX::Core::Stream]
      def self.new_stream(device); end

      # A module function that synchronizes on either the default stream or a given stream.
      # @overload synchronize()
      #   Synchronize on the default stream.
      # @overload synchronize(stream)
      #   @param stream [MLX::Core::Stream]
      #   Synchronize on the provided stream.
      # @return [void]
      def self.synchronize(stream = nil); end

      # A convenience method returning a StreamContext object
      # to be used for scoping default device/stream changes.
      # @param stream_or_device [MLX::Core::Stream, MLX::Core::Device, Integer]
      # @return [MLX::Core::StreamContext]
      def self.stream(stream_or_device); end


      # Constructor for Stream objects.
      # If +device+ is provided, the stream is created on that device;
      # otherwise, a default is used (usually CPU).
      # @param device [MLX::Core::Device, Integer, nil]
      def initialize(device = nil); end

      # @return [MLX::Core::Device] the device the stream is bound to
      def device; end

      # Synchronizes on this stream, blocking until all queued operations finish.
      # (Convenience wrapper around MLX::Core.synchronize(self).)
      # @return [void]
      def synchronize; end

      # Compares equality of two Stream objects.
      # @param other [MLX::Core::Stream]
      # @return [Boolean]
      def ==(other); end

      # String representation of the Stream, e.g. "MLX::Core::Stream(index=..., device=...)"
      # @return [String]
      def inspect; end
      alias to_s inspect

    end

    # A context manager for setting the current device and stream.
    class StreamContext
      # Create a new context manager with either a Stream, Device, or device-type integer.
      # @param stream_or_device [MLX::Core::Stream, MLX::Core::Device, Integer]
      def initialize(stream_or_device); end

      # Enters the context, sets the default device/stream.
      # @return [self]
      def enter; end

      # Exits the context, restoring the previous default device/stream.
      # @param exc_type [Class, nil] (ignored by this extension)
      # @param exc_value [Exception, nil] (ignored by this extension)
      # @param traceback [Object, nil] (ignored by this extension)
      # @return [void]
      def exit(exc_type, exc_value, traceback); end
    end

    #--------------------------------------------------------
    # The main N-dimensional array class
    #--------------------------------------------------------
    class Array
      # Create a new array (from Ruby arrays, scalars, or another MLX::Array)
      def initialize(val = nil, dtype = nil); end

      #------------------------------------------------
      # Basic properties
      #------------------------------------------------
      def shape; end      # Returns an Array of dimension sizes
      def dtype; end      # Returns a Dtype
      def ndim; end       # Number of dimensions
      def size; end       # Total number of elements
      def itemsize; end   # Size in bytes of each element
      def nbytes; end     # Total bytes of storage

      #------------------------------------------------
      # Conversions & casting
      #------------------------------------------------
      def to_s; end       # String representation
      def item; end       # Return the single value as a Ruby scalar (size must be 1)
      def tolist; end     # Convert to nested Ruby arrays
      def astype(dtype); end
      def to_i; end       # Convert a size-1 array to integer
      def to_f; end       # Convert a size-1 array to float

      # Coercion hook so that scalar + array calls our + operator
      def coerce(other); end

      #------------------------------------------------
      # Indexing & iteration
      #------------------------------------------------
      def [](index); end
      def []=(index, value); end
      def each; end

      # Special "indexed update" aggregator:
      def at; end   # Returns an MLX::ArrayAt object

      #------------------------------------------------
      # Mathematical unary ops
      #------------------------------------------------
      def abs; end
      def square; end
      def sqrt; end
      def rsqrt; end
      def reciprocal; end
      def exp; end
      def log; end
      def log2; end
      def log10; end
      def log1p; end
      def sin; end
      def cos; end

      # Power operator
      def **(other); end

      # Unary negation
      def -@; end

      #------------------------------------------------
      # Array manipulation
      #------------------------------------------------
      def reshape(*shape_or_tuple); end
      def flatten(start_axis = 0, end_axis = -1); end
      def squeeze(*axes); end
      def transpose(*axes); end
      def t; end
      def moveaxis(source, destination); end
      def swapaxes(axis1, axis2); end
      def split(indices_or_sections, axis = 0); end
      def diagonal(offset = 0, axis1 = 0, axis2 = 1); end
      def diag(k = 0); end
      def view(new_dtype); end
      def conj; end

      #------------------------------------------------
      # Reductions
      #------------------------------------------------
      def all(axis = nil, keepdims = false); end
      def any(axis = nil, keepdims = false); end
      def sum(axis = nil, keepdims = false); end
      def prod(axis = nil, keepdims = false); end
      def min(axis = nil, keepdims = false); end
      def max(axis = nil, keepdims = false); end
      def mean(axis = nil, keepdims = false); end
      def logsumexp(axis = nil, keepdims = false); end
      def std(axis = nil, keepdims = false, ddof = 0); end
      def var(axis = nil, keepdims = false, ddof = 0); end

      #------------------------------------------------
      # Arg-based ops & cumulative ops
      #------------------------------------------------
      def argmin(axis = nil, keepdims = false); end
      def argmax(axis = nil, keepdims = false); end
      def cumsum(axis = nil, reverse = false, inclusive = true); end
      def cumprod(axis = nil, reverse = false, inclusive = true); end
      def cummax(axis = nil, reverse = false, inclusive = true); end
      def cummin(axis = nil, reverse = false, inclusive = true); end

      #------------------------------------------------
      # Other operations
      #------------------------------------------------
      def round(decimals = 0); end

      #------------------------------------------------
      # Arithmetic operators
      #------------------------------------------------
      def +(other); end
      def -(other); end
      def *(other); end
      def /(other); end
      def floor_div(other); end
      def %(other); end
      def matmul(other); end
      # Some Ruby versions let you define '@' for matrix multiply
      # (like Python's a @ b), but it's not universally recognized:
      # def @(other); end

      #------------------------------------------------
      # Comparison operators (elementwise)
      #------------------------------------------------
      def ==(other); end
      def !=(other); end
      def <(other); end
      def <=(other); end
      def >(other); end
      def >=(other); end

      #------------------------------------------------
      # Bitwise operators
      #------------------------------------------------
      def ~@; end   # Bitwise invert or logical_not if bool
      def &(other); end
      def |(other); end
      def ^(other); end
      def <<(other); end
      def >>(other); end
    end

    #--------------------------------------------------------
    # The "at" object for indexed in-place style updates
    #--------------------------------------------------------
    class ArrayAt
      # Specify the indices
      def [](indices); end

      # Perform the specified update at the chosen indices
      def add(value); end
      def subtract(value); end
      def multiply(value); end
      def divide(value); end
      def maximum(value); end
      def minimum(value); end
    end

    


    #----------------------------------------------------------------
    # MLX::Ops module
    #
    # The primary module that defines all of the "ops" (array
    # manipulation, math, creation, etc.). Each method is a stub
    # with no body, mirroring the final Python-equivalent API.
    #----------------------------------------------------------------
    module Ops

      # Already-existing methods (array creation, basic ops, etc.)
      # ----------------------------------------------------------
      def self.zeros(shape, dtype, stream=nil); end
      def self.ones(shape, dtype, stream=nil); end
      def self.full(shape, fill_value, dtype, stream=nil); end
      def self.arange(start, stop, step, dtype, stream=nil); end
      def self.identity(n, dtype, stream=nil); end
      def self.eye(n, m=nil, k=0, dtype=nil, stream=nil); end

      def self.reshape(a, shape, stream=nil); end
      def self.flatten(a, start_axis, end_axis, stream=nil); end
      def self.squeeze(a, axis=nil, stream=nil); end
      def self.expand_dims(a, axis, stream=nil); end

      def self.abs(a, stream=nil); end
      def self.sign(a, stream=nil); end
      def self.negative(a, stream=nil); end

      def self.add(a, b, stream=nil); end
      def self.subtract(a, b, stream=nil); end
      def self.multiply(a, b, stream=nil); end
      def self.divide(a, b, stream=nil); end

      def self.equal(a, b, stream=nil); end
      def self.not_equal(a, b, stream=nil); end
      def self.greater(a, b, stream=nil); end
      def self.greater_equal(a, b, stream=nil); end
      def self.less(a, b, stream=nil); end
      def self.less_equal(a, b, stream=nil); end

      def self.stop_gradient(a, stream=nil); end

      # New or missing methods to match Python’s interface
      # --------------------------------------------------
      def self.unflatten(a, axis, shape, stream=nil); end
      def self.divmod(a, b, stream=nil); end
      def self.floor_divide(a, b, stream=nil); end
      def self.remainder(a, b, stream=nil); end
      def self.array_equal(a, b, equal_nan=false, stream=nil); end
      def self.matmul(a, b, stream=nil); end
      def self.square(a, stream=nil); end
      def self.sqrt(a, stream=nil); end
      def self.rsqrt(a, stream=nil); end
      def self.reciprocal(a, stream=nil); end
      def self.logical_not(a, stream=nil); end
      def self.logical_and(a, b, stream=nil); end
      def self.logical_or(a, b, stream=nil); end
      def self.logaddexp(a, b, stream=nil); end
      def self.exp(a, stream=nil); end
      def self.expm1(a, stream=nil); end
      def self.erf(a, stream=nil); end
      def self.erfinv(a, stream=nil); end
      def self.sin(a, stream=nil); end
      def self.cos(a, stream=nil); end
      def self.tan(a, stream=nil); end
      def self.arcsin(a, stream=nil); end
      def self.arccos(a, stream=nil); end
      def self.arctan(a, stream=nil); end
      def self.arctan2(a, b, stream=nil); end
      def self.sinh(a, stream=nil); end
      def self.cosh(a, stream=nil); end
      def self.tanh(a, stream=nil); end
      def self.arcsinh(a, stream=nil); end
      def self.arccosh(a, stream=nil); end
      def self.arctanh(a, stream=nil); end
      def self.degrees(a, stream=nil); end
      def self.radians(a, stream=nil); end
      def self.log(a, stream=nil); end
      def self.log2(a, stream=nil); end
      def self.log10(a, stream=nil); end
      def self.log1p(a, stream=nil); end
      def self.sigmoid(a, stream=nil); end
      def self.power(a, b, stream=nil); end
      def self.linspace(start, stop, num=50, dtype=nil, stream=nil); end
      def self.kron(a, b, stream=nil); end
      def self.take(a, indices, axis=nil, stream=nil); end
      def self.take_along_axis(a, indices, axis=nil, stream=nil); end
      def self.put_along_axis(a, indices, values, axis=nil, stream=nil); end
      def self.zeros_like(a, stream=nil); end
      def self.ones_like(a, stream=nil); end
      def self.tri(n, m=nil, k=0, dtype=nil, stream=nil); end
      def self.tril(x, k=0, stream=nil); end
      def self.triu(x, k=0, stream=nil); end
      def self.allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=false, stream=nil); end
      def self.isclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=false, stream=nil); end
      def self.all(a, axis=nil, keepdims=false, stream=nil); end
      def self.any(a, axis=nil, keepdims=false, stream=nil); end
      def self.minimum(a, b, stream=nil); end
      def self.maximum(a, b, stream=nil); end
      def self.floor(a, stream=nil); end
      def self.ceil(a, stream=nil); end
      def self.isnan(a, stream=nil); end
      def self.isinf(a, stream=nil); end
      def self.isfinite(a, stream=nil); end
      def self.isposinf(a, stream=nil); end
      def self.isneginf(a, stream=nil); end
      def self.moveaxis(a, source, destination, stream=nil); end
      def self.swapaxes(a, axis1, axis2, stream=nil); end
      def self.transpose(a, axes=nil, stream=nil); end
      def self.permute_dims(a, axes=nil, stream=nil); end
      def self.sum(a, axis=nil, keepdims=false, stream=nil); end
      def self.prod(a, axis=nil, keepdims=false, stream=nil); end
      def self.min(a, axis=nil, keepdims=false, stream=nil); end
      def self.max(a, axis=nil, keepdims=false, stream=nil); end
      def self.logsumexp(a, axis=nil, keepdims=false, stream=nil); end
      def self.mean(a, axis=nil, keepdims=false, stream=nil); end
      def self.var(a, axis=nil, keepdims=false, ddof=0, stream=nil); end
      def self.std(a, axis=nil, keepdims=false, ddof=0, stream=nil); end
      def self.split(a, indices_or_sections, axis=0, stream=nil); end
      def self.argmin(a, axis=nil, keepdims=false, stream=nil); end
      def self.argmax(a, axis=nil, keepdims=false, stream=nil); end
      def self.sort(a, axis=-1, stream=nil); end
      def self.argsort(a, axis=-1, stream=nil); end
      def self.partition(a, kth, axis=-1, stream=nil); end
      def self.argpartition(a, kth, axis=-1, stream=nil); end
      def self.topk(a, k, axis=-1, stream=nil); end
      def self.broadcast_to(a, shape, stream=nil); end
      def self.broadcast_arrays(*arrays, stream:nil); end
      def self.softmax(a, axis=nil, precise=false, stream=nil); end
      def self.concatenate(arrays, axis=0, stream=nil); end
      def self.concat(arrays, axis=0, stream=nil); end
      def self.stack(arrays, axis=0, stream=nil); end
      def self.meshgrid(arrays, sparse=false, indexing="xy", stream=nil); end
      def self.repeat(a, repeats, axis=nil, stream=nil); end
      def self.clip(a, a_min, a_max, stream=nil); end
      def self.pad(a, pad_width, mode="constant", constant_values=0, stream=nil); end
      def self.as_strided(a, shape=nil, strides=nil, offset=0, stream=nil); end
      def self.cumsum(a, axis=nil, reverse=false, inclusive=true, stream=nil); end
      def self.cumprod(a, axis=nil, reverse=false, inclusive=true, stream=nil); end
      def self.cummax(a, axis=nil, reverse=false, inclusive=true, stream=nil); end
      def self.cummin(a, axis=nil, reverse=false, inclusive=true, stream=nil); end
      def self.conj(a, stream=nil); end
      def self.conjugate(a, stream=nil); end
      def self.convolve(a, v, mode="full", stream=nil); end
      def self.conv1d(input, weight, stride=1, padding=0, dilation=1, groups=1, stream=nil); end
      def self.conv2d(input, weight, stride=1, padding=0, dilation=1, groups=1, stream=nil); end
      def self.conv3d(input, weight, stride=1, padding=0, dilation=1, groups=1, stream=nil); end
      def self.conv_transpose1d(input, weight, stride=1, padding=0, output_padding=0, groups=1, dilation=1, stream=nil); end
      def self.conv_transpose2d(input, weight, stride=1, padding=0, output_padding=0, groups=1, dilation=1, stream=nil); end
      def self.conv_transpose3d(input, weight, stride=1, padding=0, output_padding=0, groups=1, dilation=1, stream=nil); end
      def self.conv_general(input, weight, strides, padding, dilation, groups, stream=nil); end

      def self.save(file, arr); end
      def self.savez(file, *args); end
      def self.savez_compressed(file, *args); end
      def self.load(file, format=nil, return_metadata=false, stream=nil); end
      def self.save_safetensors(file, arrays, metadata=nil); end
      def self.save_gguf(file, arrays, metadata=nil); end
      def self.where(condition, x, y, stream=nil); end
      def self.nan_to_num(a, nan=0.0, posinf=nil, neginf=nil, stream=nil); end
      def self.round(a, decimals=0, stream=nil); end
      def self.quantized_matmul(x, w, scales, biases, transpose=true, group_size=64, bits=4, stream=nil); end
      def self.quantize(w, group_size=64, bits=4, stream=nil); end
      def self.dequantize(w, scales, biases, group_size=64, bits=4, stream=nil); end
      def self.gather_qmm(x, w, scales, biases, lhs_indices=nil, rhs_indices=nil, transpose=true, group_size=64, bits=4, stream=nil); end
      def self.tensordot(a, b, axes=2, stream=nil); end
      def self.inner(a, b, stream=nil); end
      def self.outer(a, b, stream=nil); end
      def self.tile(a, reps, stream=nil); end
      def self.block_masked_mm(a, b, block_size=64, mask_out=nil, mask_lhs=nil, mask_rhs=nil, stream=nil); end
      def self.gather_mm(a, b, lhs_indices, rhs_indices, stream=nil); end
      def self.diagonal(a, offset=0, axis1=0, axis2=1, stream=nil); end
      def self.diag(a, k=0, stream=nil); end
      def self.trace(a, offset=0, axis1=0, axis2=1, dtype=nil, stream=nil); end
      def self.atleast_1d(arys, stream=nil); end
      def self.atleast_2d(arys, stream=nil); end
      def self.atleast_3d(arys, stream=nil); end
      def self.issubdtype(arg1, arg2); end
      def self.bitwise_and(a, b, stream=nil); end
      def self.bitwise_or(a, b, stream=nil); end
      def self.bitwise_xor(a, b, stream=nil); end
      def self.left_shift(a, b, stream=nil); end
      def self.right_shift(a, b, stream=nil); end
      def self.bitwise_invert(a, stream=nil); end
      def self.view(a, dtype, stream=nil); end
      def self.hadamard_transform(a, scale=nil, stream=nil); end
      def self.einsum_path(subscripts, operands); end
      def self.einsum(subscripts, operands, stream=nil); end
      def self.roll(a, shift, axis=nil, stream=nil); end
      def self.real(a, stream=nil); end
      def self.imag(a, stream=nil); end
      def self.slice(a, start_indices, axes, slice_size, stream=nil); end
      def self.slice_update(a, update, start_indices, axes, stream=nil); end
      def self.contiguous(a, allow_col_major=false, stream=nil); end
    end

  end

end
  

# Try to load the native extension if available
begin
  # The path where we expect the native extension to be
  extension_path = File.join(File.dirname(__FILE__), 'core.bundle')
  require extension_path if File.exist?(extension_path)
rescue => e
  # Add error handling if needed
  puts "Error loading native extension: #{e.message}"
end 