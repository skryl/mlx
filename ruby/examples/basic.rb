require_relative '../mlx/mlx'

# Create an array from a Ruby array
arr1 = MLX.array([1, 2, 3, 4, 5])
puts "Array 1: #{arr1}"
puts "Shape: #{arr1.shape.inspect}"
puts "Size: #{arr1.size}"
puts "Dimensions: #{arr1.ndim}"
puts "Dtype: #{arr1.dtype}"

# Basic arithmetic
arr2 = MLX.array([10, 20, 30, 40, 50])
sum_arr = arr1 + arr2
puts "\nSum: #{sum_arr}"

# Add scalar
scalar_sum = arr1 + 5
puts "Adding scalar 5: #{scalar_sum}" 