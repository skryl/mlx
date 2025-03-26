require 'minitest/autorun'
require_relative '../mlx/mlx'  # Main MLX library

# Base test case for MLX tests
class MLXTestCase < Minitest::Test
  # Check if running on Apple Silicon
  def is_apple_silicon?
    RUBY_PLATFORM.include?('arm64') && RUBY_PLATFORM.include?('darwin')
  end
  
  # Setup method called before each test
  def setup
    @default_device = MLX.default_device
    device_env = ENV['DEVICE']
    if device_env
      device = MLX.const_get(device_env.upcase)
      MLX.set_default_device(device)
    end
  end
  
  # Teardown method called after each test
  def teardown
    MLX.set_default_device(@default_device)
  end
  
  # Assert that a MLX array is close to another array or value
  # 
  # @param mx_res [MLX::Array] Result to test
  # @param expected [MLX::Array, Array, Numeric] Expected value
  # @param atol [Float] Absolute tolerance
  # @param rtol [Float] Relative tolerance
  def assert_array_equal(mx_res, expected, atol: 1e-2, rtol: 1e-2)
    # Convert expected to MLX array if needed
    expected = MLX.array(expected) unless expected.is_a?(MLX::Array)
    
    # Check shape
    assert_equal expected.shape, mx_res.shape, "Shape mismatch"
    
    # Check dtype
    assert_equal expected.dtype, mx_res.dtype, "Dtype mismatch"
    
    # Check values are close
    assert MLX.allclose(mx_res, expected, atol: atol, rtol: rtol), 
           "Arrays are not close enough"
  end
  
  # Compare MLX function against NumPy equivalent
  # This is a placeholder as Ruby doesn't have NumPy
  # We could implement this with PyCall or similar gem in future
  def assert_cmp_numpy(args, mx_fn, np_fn, atol: 1e-2, rtol: 1e-2, dtype: MLX::FLOAT32, **kwargs)
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
    a = MLX.array(a) unless a.is_a?(MLX::Array)
    b = MLX.array(b) unless b.is_a?(MLX::Array)
    
    assert MLX.allclose(a, b, atol: atol, rtol: rtol),
           "Arrays not close: #{a.to_s} vs #{b.to_s}"
  end
end 