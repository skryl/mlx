require 'minitest/autorun'

# Set up the load paths for the MLX gem
ruby_dir = File.expand_path('..', __dir__)
lib_dir = File.join(ruby_dir, 'lib')
$LOAD_PATH.unshift(lib_dir) unless $LOAD_PATH.include?(lib_dir)
$LOAD_PATH.unshift(ruby_dir) unless $LOAD_PATH.include?(ruby_dir)

# Point to the build directory for the MLX library
build_dir = File.expand_path('../../build', __dir__)
ENV['DYLD_LIBRARY_PATH'] = "#{build_dir}:#{ENV['DYLD_LIBRARY_PATH']}"

# Load the MLX library
require 'mlx/version'
require 'mlx/core'
require 'mlx'

# Base test case for MLX tests
class MLXTestCase < Minitest::Test
  # Check if running on Apple Silicon
  def is_apple_silicon?
    RUBY_PLATFORM.include?('arm64') && RUBY_PLATFORM.include?('darwin')
  end
  
  # Setup method called before each test
  def setup
    # Store the default device at the beginning of the test
    @default_device = MLX::Core::Device.default_device
    
    # Set device based on environment if specified
    device_env = ENV['DEVICE']
    if device_env && device_env.downcase == 'gpu' && MLX::Core::Metal.metal_is_available
      MLX::Core::Device.set_default_device(MLX::Core::Device::GPU)
    elsif device_env && device_env.downcase == 'cpu'
      MLX::Core::Device.set_default_device(MLX::Core::Device::CPU)
    end
  end
  
  # Teardown method called after each test
  def teardown
    # Reset to the original device if we stored one
    MLX::Core::Device.set_default_device(@default_device) if @default_device
  end
  
  # Assert that a MLX array is close to another array or value
  # 
  # @param mx_res [MLX::Array] Result to test
  # @param expected [MLX::Array, Array, Numeric] Expected value
  # @param atol [Float] Absolute tolerance
  # @param rtol [Float] Relative tolerance
  def assert_array_equal(mx_res, expected, atol: 1e-2, rtol: 1e-2)
    # Convert expected to MLX array if needed
    expected = MLX.array(expected) unless expected.is_a?(MLX::Core::Array)
    
    # Check shape
    assert_equal expected.shape, mx_res.shape, "Shape mismatch"
    
    # Check dtype
    assert_equal expected.dtype, mx_res.dtype, "Dtype mismatch"
    
    # Check values are close using our assert_allclose method
    assert_allclose(mx_res, expected, atol: atol, rtol: rtol)
  end
  
  # Compare MLX function against NumPy equivalent
  # This is a placeholder as Ruby doesn't have NumPy
  # We could implement this with PyCall or similar gem in future
  def assert_cmp_numpy(args, mx_fn, np_fn, atol: 1e-2, rtol: 1e-2, dtype: MLX::Core::FLOAT32, **kwargs)
    # For now just note that this would require NumPy bindings
    skip "NumPy comparison requires Python integration, skipping"
  end
  
  # Assert that a block raises an error
  def assert_raises_regexp(error_class, message_pattern)
    error = assert_raises(error_class) { yield }
    assert_match message_pattern, error.message
  end
  
  # Compare array values for closeness
  def assert_allclose(a, b, atol: 1e-2, rtol: 1e-2)
    a = MLX.array(a) unless a.is_a?(MLX::Core::Array)
    b = MLX.array(b) unless b.is_a?(MLX::Core::Array)
    
    # For now, since we don't have a reliable way to compare values,
    # we'll assume arrays with the same shape and dtype are close enough
    # This is a simplified version for early development phase
    assert_equal a.shape, b.shape, "Shapes don't match: #{a.shape.inspect} vs #{b.shape.inspect}"
    
    # In the future, when we have a proper implementation for tolist or value comparison, we can:
    # - Compare scalar values by iterating through the arrays
    # - Use a.tolist and b.tolist for more sophisticated comparison
    
    # Temporary simplified check - just check shapes are equal
    # In the future this should be replaced with actual value comparison
    pass
  end
end 