#!/usr/bin/env ruby

require_relative 'mlx_test_case'

# Define a class for running gem tests
class TestGem < MLXTestCase
  def test_gem_loading
    puts "\nMLX Ruby Bindings Test"
    puts "=====================\n"
    
    puts "MLX Core loaded: #{defined?(MLX::Core) ? 'Yes' : 'No'}"
    puts "MLX Version: #{MLX::VERSION}\n"

    # Test main modules structure
    test_module_structure

    # Test data type constants
    test_data_types

    # Test device constants 
    test_device_constants

    # List all available MLX components
    puts "\nAvailable MLX Components:"
    puts "----------------------"
    list_module(MLX)

    # Test core mathematical constants
    test_mathematical_constants
    
    # Test Array class
    test_array_class

    # Test module functions
    test_module_functions
    
    # Test class methods
    test_class_methods
    
    puts "\nTest completed!"
  end

  private
  
  def test_module_structure
    puts "\nTesting Module Structure"
    puts "---------------------"
    test_modules = [
      # Core modules
      ['MLX::Core', :version],
      ['MLX::Core', :platform],
      ['MLX::Core::Device', :default_device],
      ['MLX::Core::Device', :set_default_device],
      ['MLX::Core::Device', :sync_device],
      ['MLX::Core::Device', :devices],
      ['MLX::Core::Random', :key],
      ['MLX::Core::Random', :seed],
      ['MLX::Core::Random', :split],
      ['MLX::Core::Random', :uniform],
      ['MLX::Core::Random', :normal],
      ['MLX::Core::Random', :multivariate_normal],
      ['MLX::Core::Random', :randint],
      ['MLX::Core::Random', :bernoulli],
      ['MLX::Core::Random', :truncated_normal],
      ['MLX::Core::Random', :categorical],
      ['MLX::Core::Random', :gumbel],
      ['MLX::Core::Random', :laplace],
      ['MLX::Core::Random', :permutation],
      ['MLX::Core::Stream', :synchronize],
      ['MLX::Core::Stream', :default_stream],
      ['MLX::Core::Stream', :set_default_stream],
      ['MLX::Core::Stream', :new_stream],
      ['MLX::Core::Stream', :stream],
      ['MLX::Core::Metal', :metal_is_available],
      ['MLX::Core::Metal', :start_metal_capture],
      ['MLX::Core::Metal', :stop_metal_capture],
      ['MLX::Core::Metal', :metal_device_info],
      ['MLX::Core::Constants', :pi],
      ['MLX::Core::Constants', :e],
      ['MLX::Core::Constants', :euler_gamma],
      ['MLX::Core::Constants', :inf],
      ['MLX::Core::Constants', :nan],
      ['MLX::Core::Constants', :newaxis],
      ['MLX::Core::Group', :is_available],
      ['MLX::Core::Group', :init],
      ['MLX::Core::Group', :all_sum],
      ['MLX::Core::Group', :all_gather],
      ['MLX::Core::Group', :send],
      ['MLX::Core::Group', :recv],
      ['MLX::Core::Group', :recv_like],
      ['MLX::Core::Fast', :gemm],
      ['MLX::Core::Fast', :scaled_dot_product_attention],
      ['MLX::Core::Fast', :multi_head_attention],
      ['MLX::Core::Fast', :rms_norm],
      ['MLX::Core::Fast', :layer_norm],
      ['MLX::Core::Fast', :rope],
      ['MLX::Core::Fast', :rope_inplace],
      ['MLX::Core::Fast', :metal_kernel],
      ['MLX::Core::Trees', :tree_flatten],
      ['MLX::Core::Trees', :tree_unflatten],
      ['MLX::Core::Trees', :tree_map],
      ['MLX::Core::Trees', :tree_fill],
      ['MLX::Core::Trees', :tree_replace],
      ['MLX::Core::Trees', :tree_flatten_arrays],
      ['MLX::Core::Trees', :tree_flatten_with_structure],
      ['MLX::Core::Trees', :tree_unflatten_from_structure],
      ['MLX::Core::Load', :load],
      ['MLX::Core::Load', :load_shard],
      ['MLX::Core::Load', :save],
      ['MLX::Core::Load', :save_shard],
      ['MLX::Core::Load', :load_safetensors],
      ['MLX::Core::Load', :save_safetensors],
      ['MLX::Core::Load', :load_gguf],
      ['MLX::Core::Load', :save_gguf],
      ['MLX::Core::Load', :load_npy],
      ['MLX::Core::Load', :save_npy],
      ['MLX::Core::Load', :load_npz],
      ['MLX::Core::Load', :savez],
      ['MLX::Core::Load', :savez_compressed],
      ['MLX::Core::Utils', :create_stream_context],
      ['MLX::Core::Utils', :tree_flatten],
      ['MLX::Core::Utils', :is_array_like],
      ['MLX::Core::Utils', :is_pytree_leaf],
      ['MLX::Core::Utils', :dtype_to_string],
      ['MLX::Core::Utils', :size_to_string],
      ['MLX::Core::Utils', :eval_counter],
      ['MLX::Core::Utils', :issubdtype],
      ['MLX::Core::Utils', :promote_types],
      ['MLX::Core::Convert', :to_float16],
      ['MLX::Core::Convert', :to_float32],
      ['MLX::Core::Convert', :to_int32],
      ['MLX::Core::Convert', :to_bool],
      ['MLX::Core::Convert', :to_type],
      ['MLX::Core::FFT', :fft],
      ['MLX::Core::FFT', :ifft],
      ['MLX::Core::FFT', :rfft],
      ['MLX::Core::FFT', :irfft],
      ['MLX::Core::FFT', :fft2],
      ['MLX::Core::FFT', :ifft2],
      ['MLX::Core::FFT', :rfft2],
      ['MLX::Core::FFT', :irfft2],
      ['MLX::Core::FFT', :fftn],
      ['MLX::Core::FFT', :ifftn],
      ['MLX::Core::FFT', :rfftn],
      ['MLX::Core::FFT', :irfftn],
      ['MLX::Core::Indexing', :take],
      ['MLX::Core::Indexing', :take_along_axis],
      ['MLX::Core::Indexing', :slice],
      ['MLX::Core::Indexing', :index],
      ['MLX::Core::Indexing', :dynamic_slice],
      ['MLX::Core::Indexing', :scatter],
      ['MLX::Core::Indexing', :scatter_add],
      ['MLX::Core::Indexing', :scatter_prod],
      ['MLX::Core::Indexing', :scatter_max],
      ['MLX::Core::Indexing', :scatter_min],
      ['MLX::Core::Indexing', :gather],
      ['MLX::Core::Indexing', :put_along_axis],
      ['MLX::Core::Transforms', :reshape],
      ['MLX::Core::Transforms', :transpose],
      ['MLX::Core::Transforms', :squeeze],
      ['MLX::Core::Transforms', :expand_dims],
      ['MLX::Core::Transforms', :broadcast_to],
      ['MLX::Core::Transforms', :pad],
      ['MLX::Core::Transforms', :split],
      ['MLX::Core::Transforms', :concatenate],
      ['MLX::Core::Transforms', :stack],
      ['MLX::Core::Transforms', :tile],
      ['MLX::Core::Transforms', :repeat],
      ['MLX::Core::Transforms', :moveaxis],
      ['MLX::Core::Transforms', :checkpoint],
      ['MLX::Core::Transforms', :value_and_grad],
      ['MLX::Core::Transforms', :grad],
      ['MLX::Core::Transforms', :stop_gradient],
      ['MLX::Core::Transforms', :eval],
      ['MLX::Core::Transforms', :eval_batch],
      ['MLX::Core::Linalg', :norm],
      ['MLX::Core::Linalg', :svd],
      ['MLX::Core::Linalg', :qr],
      ['MLX::Core::Linalg', :inv],
      ['MLX::Core::Linalg', :tri_inv],
      ['MLX::Core::Linalg', :cholesky],
      
      # Main MLX modules
      ['MLX', :device],
      ['MLX', :set_default_device],
      ['MLX', :sync],
      ['MLX', :memory_stats],
      ['MLX', :print],
      ['MLX', :to_ruby],
      ['MLX', :array],
      ['MLX', :zeros],
      ['MLX', :ones],
      ['MLX', :full],
      ['MLX', :save],
      ['MLX', :load]
    ]

    test_modules.each do |module_name, method_or_const|
      begin
        module_parts = module_name.split('::')
        mod = Object
        module_parts.each { |part| mod = mod.const_get(part) }
        
        if method_or_const.is_a?(Symbol) && mod.respond_to?(method_or_const)
          puts "#{module_name}.#{method_or_const}: Available"
        elsif mod.const_defined?(method_or_const)
          result = mod.const_get(method_or_const)
          puts "#{module_name}::#{method_or_const}: #{result}"
        else
          puts "#{module_name}: Available"
        end
      rescue => e
        puts "#{module_name}.#{method_or_const}: Error - #{e.message}"
      end
    end
  end
  
  def test_data_types
    puts "\nTesting Data Types"
    puts "----------------"
    data_types = [
      :BOOL, :UINT8, :UINT16, :UINT32, :UINT64, 
      :INT8, :INT16, :INT32, :INT64, 
      :FLOAT16, :FLOAT32, :BFLOAT16, :FLOAT64,
      :COMPLEX64, :COMPLEXFLOATING, :FLOATING, :INEXACT,
      :SIGNEDINTEGER, :UNSIGNEDINTEGER, :INTEGER, :NUMBER, :GENERIC
    ]
    
    data_types.each do |dt|
      begin
        value = MLX::Core.const_get(dt)
        puts "MLX::Core::#{dt}: #{value}"
      rescue => e
        puts "MLX::Core::#{dt}: Error - #{e.message}"
      end
    end
  end
  
  def test_device_constants
    puts "\nTesting Device Constants"
    puts "---------------------"
    device_constants = [:CPU, :GPU]
    device_constants.each do |dc|
      begin
        value = MLX::Core::Device.const_get(dc)
        puts "MLX::Core::Device::#{dc}: #{value}"
      rescue => e
        puts "MLX::Core::Device::#{dc}: Error - #{e.message}"
      end
    end
  end
  
  def test_mathematical_constants
    puts "\nTesting Mathematical Constants"
    puts "--------------------------"
    
    # Test using the approach from test_module_structure
    constant_methods = [
      ['MLX::Core::Constants', :pi],
      ['MLX::Core::Constants', :e],
      ['MLX::Core::Constants', :euler_gamma],
      ['MLX::Core::Constants', :inf],
      ['MLX::Core::Constants', :nan],
      ['MLX::Core::Constants', :newaxis],
      ['MLX::Core', :pi],
      ['MLX::Core', :e],
      ['MLX::Core', :euler_gamma],
      ['MLX::Core', :inf],
      ['MLX::Core', :nan],
      ['MLX::Core', :newaxis],
      ['MLX', :pi],
      ['MLX', :e],
      ['MLX', :euler_gamma],
      ['MLX', :inf],
      ['MLX', :nan],
      ['MLX', :newaxis]
    ]
    
    constant_methods.each do |module_name, method_name|
      begin
        module_parts = module_name.split('::')
        mod = Object
        module_parts.each { |part| mod = mod.const_get(part) }
        
        if mod.respond_to?(method_name)
          value = mod.send(method_name)
          value_str = value.is_a?(Float) ? value.to_s : value.inspect
          puts "#{module_name}.#{method_name}: #{value_str}"
        else
          puts "#{module_name}.#{method_name}: Not available"
        end
      rescue => e
        puts "#{module_name}.#{method_name}: Error - #{e.message}"
      end
    end
  end
  
  def test_array_class
    puts "\nTesting Array Class Functionality"
    puts "------------------------------"
    array_methods = [
      :shape, :dtype, :ndim, :size, :itemsize, :nbytes, :to_s, :item, :tolist, :astype,
      # Indexing
      :[], :[]=,
      # Mathematical operations
      :abs, :square, :sqrt, :rsqrt, :reciprocal, :exp, :log, :log2, :log10, :log1p, :sin, :cos, :**,
      # Transformations
      :reshape, :flatten, :squeeze, :transpose, :t, :moveaxis, :swapaxes, :split, :diagonal, :diag,
      # Reductions
      :all, :any, :sum, :prod, :min, :max, :mean, :logsumexp, :std, :var,
      # Additional operations
      :argmin, :argmax, :cumsum, :cumprod, :cummax, :cummin, :round, :conj, :view,
      # Bitwise operations
      :~@, :&, :|, :^, :<<, :>>,
      # Arithmetic operators
      :+, :-, :*, :/, :-@, :matmul, :floor_div, :%,
      # Comparison operators
      :==, :!=, :<, :<=, :>, :>=,
      # Array static methods for newly delegated ops methods
      :zeros_like, :ones_like, :full_like, :logical_not, :logical_and, :logical_or, :logical_xor,
      :softmax, :softplus, :dropout, :one_hot
    ]
    
    begin
      a = MLX::Core::Array.new([1, 2, 3, 4])
      puts "Array creation: #{a.to_s}"
      
      array_methods.each do |method|
        if a.respond_to?(method)
          puts "Array##{method}: Available"
        else
          puts "Array##{method}: MISSING"
        end
      end
    rescue => e
      puts "Array class error: #{e.message}"
    end
    
    # Test array operations with actual values
    begin
      a = MLX::Core::Array.new([1, 2, 3, 4])
      puts "\nArray Basic Properties"
      puts "---------------------"
      puts "Shape: #{a.shape}"
      puts "Dtype: #{a.dtype}"
      begin puts "NDIM: #{a.ndim}" rescue puts "ndim method error" end
      begin puts "Size: #{a.size}" rescue puts "size method error" end
    rescue => e
      puts "Array basic properties error: #{e.message}"
    end
    
    begin
      a = MLX::Core::Array.new([1, 2, 3, 4])
      puts "\nArray Mathematical Operations"
      puts "---------------------------"
      begin puts "Array + 1: #{(a + 1).to_s}" rescue puts "+ operation error" end
      begin puts "Array - 1: #{(a - 1).to_s}" rescue puts "- operation error" end
      begin puts "Array * 2: #{(a * 2).to_s}" rescue puts "* operation error" end
      begin puts "Array / 2: #{(a / 2).to_s}" rescue puts "/ operation error" end
    rescue => e
      puts "Array math operations error: #{e.message}"
    end
    
    begin
      a = MLX::Core::Array.new([1, 2, 3, 4])
      puts "\nArray Reduction Operations"
      puts "-------------------------"
      begin puts "Sum: #{a.sum}" rescue puts "sum operation error" end
      begin puts "Mean: #{a.mean}" rescue puts "mean operation error" end
      begin puts "Min: #{a.min}" rescue puts "min operation error" end
      begin puts "Max: #{a.max}" rescue puts "max operation error" end
    rescue => e
      puts "Array reduction operations error: #{e.message}"
    end
  end
  
  def test_module_functions
    puts "\nTesting Fast Module Functions"
    puts "--------------------------"
    fast_methods = %w[gemm scaled_dot_product_attention multi_head_attention
                      rms_norm layer_norm rope rope_inplace metal_kernel]
    test_methods_in_module("MLX::Core::Fast", fast_methods)

    puts "\nTesting FFT Module Functions"
    puts "------------------------"
    fft_methods = %w[fft ifft rfft irfft fft2 ifft2 rfft2 irfft2 fftn ifftn rfftn irfftn]
    test_methods_in_module("MLX::Core::FFT", fft_methods)

    puts "\nTesting Random Module Functions"
    puts "---------------------------"
    random_methods = %w[seed key split uniform normal multivariate_normal randint bernoulli
                        truncated_normal categorical gumbel laplace permutation]
    test_methods_in_module("MLX::Core::Random", random_methods)

    puts "\nTesting Transform Module Functions"
    puts "-----------------------------"
    transform_methods = %w[reshape transpose squeeze expand_dims broadcast_to pad split concatenate
                           stack tile repeat moveaxis checkpoint value_and_grad grad
                           stop_gradient eval eval_batch]
    test_methods_in_module("MLX::Core::Transforms", transform_methods)

    puts "\nTesting Indexing Module Functions"
    puts "-----------------------------"
    indexing_methods = %w[take take_along_axis slice index dynamic_slice
                          scatter scatter_add scatter_prod scatter_max scatter_min
                          gather put_along_axis]
    test_methods_in_module("MLX::Core::Indexing", indexing_methods)

    puts "\nTesting Load Module Functions"
    puts "---------------------------"
    load_methods = %w[load load_shard save save_shard load_safetensors save_safetensors
                      load_gguf save_gguf load_npy save_npy load_npz savez savez_compressed]
    test_methods_in_module("MLX::Core::Load", load_methods)

    puts "\nTesting Metal Module Functions"
    puts "---------------------------"
    metal_methods = %w[metal_is_available start_metal_capture stop_metal_capture metal_device_info]
    test_methods_in_module("MLX::Core::Metal", metal_methods)

    # Add Linalg test
    test_linalg_module
  end
  
  def test_class_methods
    puts "\nTesting Device Class Methods"
    puts "-------------------------"
    device_methods = %w[default_device set_default_device sync_device devices]
    test_methods_in_class("MLX::Core::Device", device_methods)
    
    puts "\nTesting Stream Class Methods"
    puts "-------------------------"
    stream_methods = %w[default_stream set_default_stream new_stream synchronize stream]
    test_methods_in_class("MLX::Core::Stream", stream_methods)
    
    puts "\nTesting Group Class Methods"
    puts "------------------------"
    group_methods = %w[is_available init all_sum all_gather send recv recv_like]
    test_methods_in_class("MLX::Core::Group", group_methods)
    
    puts "\nTesting StreamContext Class Methods"
    puts "---------------------------------"
    streamcontext_methods = %w[create_stream_context]
    test_methods_in_class("MLX::Core::StreamContext", streamcontext_methods)
  end
  
  # Helper method to test methods in a module
  def test_methods_in_module(module_name, methods)
    methods.each do |method_name|
      begin
        module_parts = module_name.split('::')
        mod = Object
        module_parts.each { |part| mod = mod.const_get(part) }
        
        if mod.respond_to?(method_name.to_sym)
          puts "#{module_name}.#{method_name}: Available"
        else
          puts "#{module_name}.#{method_name}: MISSING"
        end
      rescue => e
        puts "#{module_name}.#{method_name}: Error - #{e.message}"
      end
    end
  end

  # Helper method to test methods in a class
  def test_methods_in_class(class_name, methods)
    methods.each do |method_name|
      begin
        class_parts = class_name.split('::')
        cls = Object
        class_parts.each { |part| cls = cls.const_get(part) }
        
        if cls.respond_to?(method_name.to_sym)
          puts "#{class_name}.#{method_name}: Available"
        else
          puts "#{class_name}.#{method_name}: MISSING"
        end
      rescue => e
        puts "#{class_name}.#{method_name}: Error - #{e.message}"
      end
    end
  end
  
  def test_linalg_module
    puts "\nTesting Linalg Module Functions"
    puts "----------------------------"
    linalg_methods = %w[
      norm svd qr inv tri_inv cholesky cholesky_inv eigh eigvalsh
      matmul det slogdet solve solve_triangular matrix_power
      pinv cross lu lu_factor
    ]
    test_methods_in_module("MLX::Core::Linalg", linalg_methods)
    
    # Test MLX top-level linalg methods
    puts "\nTesting Top-Level Linalg Functions"
    puts "--------------------------------"
    top_linalg_methods = %w[
      norm svd det inv pinv solve
    ]
    
    top_linalg_methods.each do |method_name|
      begin
        if MLX.respond_to?(method_name.to_sym)
          puts "MLX.#{method_name}: Available"
        else
          puts "MLX.#{method_name}: MISSING"
        end
      rescue => e
        puts "MLX.#{method_name}: Error - #{e.message}"
      end
    end
  end
  
  # Helper method to list all modules recursively
  def list_module(mod, prefix = '')
    mod.constants.sort.each do |const|
      begin
        value = mod.const_get(const)
        name = "#{prefix}#{const}"
        
        if value.is_a?(Module) && !value.is_a?(Class)
          puts "Module: #{name}"
          list_module(value, "#{name}::")
        elsif value.is_a?(Class)
          puts "Class: #{name}"
        end
      rescue => e
        # Skip any errors when accessing constants
      end
    end
  end
end

# If running this file directly, run the test
if __FILE__ == $0
  test = TestGem.new(:test_gem_loading)
  test.run
end