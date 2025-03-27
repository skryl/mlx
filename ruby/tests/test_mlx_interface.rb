require_relative 'mlx_test_case'

# Test the relationship between MLX and MLX::Core
class TestMLXInterface < MLXTestCase
  def test_mlx_to_core_delegation
    puts "\nTesting MLX to MLX::Core delegation"
    
    # Test if MLX.array delegates to MLX::Core.array
    puts "MLX responds to array: #{MLX.respond_to?(:array)}"
    puts "MLX::Core responds to array: #{MLX::Core.respond_to?(:array)}"
    
    # Create array using MLX::Core directly - this is known to work
    core_arr = MLX::Core::Array.new([1, 2, 3])
    puts "Created array with MLX::Core::Array.new: #{core_arr}"
    
    # Check if constants are properly referenced
    puts "\nTesting constant delegation:"
    puts "MLX::FLOAT32: #{MLX::FLOAT32.inspect}"
    puts "MLX::Core::FLOAT32: #{MLX::Core::FLOAT32.inspect}"
    puts "MLX::FLOAT32 == MLX::Core::FLOAT32: #{MLX::FLOAT32 == MLX::Core::FLOAT32}"
    
    # Check if MLX module has necessary methods
    basic_methods = [:array, :zeros, :ones, :arange, :reshape, :abs, :sum]
    puts "\nChecking if MLX responds to basic methods:"
    basic_methods.each do |method|
      puts "MLX.#{method}: #{MLX.respond_to?(method)}"
    end
    
    # Check that MLX Core class is correctly exposed
    puts "\nCan access MLX::Core::Array class: #{defined?(MLX::Core::Array) != nil}"
    puts "Core::Array is aliased to MLX::Array: #{MLX::Array == MLX::Core::Array}"
  end
  
  def test_array_creation
    puts "\nTesting MLX.array creation"
    
    # Test creating arrays with different data
    arr1 = MLX.array([1, 2, 3, 4])
    puts "Created array with integers: #{arr1}"
    puts "Shape: #{arr1.shape}"
    puts "Dtype: #{arr1.dtype}"
    
    # Test creating arrays with explicit dtype
    arr2 = MLX.array([1.5, 2.5, 3.5], dtype: MLX::FLOAT64)
    puts "Created array with float64 dtype: #{arr2}"
    puts "Dtype: #{arr2.dtype}"
    
    # Test creating multi-dimensional arrays
    arr3 = MLX.array([[1, 2], [3, 4]])
    puts "Created 2D array: #{arr3}"
    puts "Shape: #{arr3.shape}"
    puts "NDIM: #{arr3.ndim}"
    
    # Test zeros and ones
    zeros = MLX.zeros([2, 3])
    puts "Created zeros array: #{zeros}"
    puts "Shape: #{zeros.shape}"
    
    ones = MLX.ones([3, 2], dtype: MLX::INT32)
    puts "Created ones array with int32 dtype: #{ones}"
    puts "Dtype: #{ones.dtype}"
    
    # Test arange
    range = MLX.arange(0, 10, 2)
    puts "Created range with arange: #{range}"
    puts "Values: #{range.tolist}"
  end
end

# Run the test if file is executed directly
if __FILE__ == $0
  test = TestMLXInterface.new(:test_mlx_to_core_delegation)
  test.run
end 