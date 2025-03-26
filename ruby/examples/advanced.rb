require_relative '../mlx/mlx'

puts "MLX Ruby Bindings Advanced Example"
puts "-----------------------------------"

# Random arrays
puts "\nRandom Arrays:"
key = MLX::Random.key(42)
uniform = MLX::Random.uniform(key, [3, 3], MLX::FLOAT32)
puts "Uniform random array:\n#{uniform}"

keys = MLX::Random.split(key, 2)
normal = MLX::Random.normal(keys[0], [2, 4], MLX::FLOAT32)
puts "Normal random array:\n#{normal}"

# Array transformations
puts "\nArray Transformations:"
arr = MLX.array([1, 2, 3, 4, 5, 6])
reshaped = MLX.reshape(arr, [2, 3])
puts "Reshaped array (2x3):\n#{reshaped}"

transposed = MLX.transpose(reshaped)
puts "Transposed array (3x2):\n#{transposed}"

# Creating arrays with ops
puts "\nCreating Arrays:"
zeros = MLX.zeros([2, 2], MLX::FLOAT32)
puts "Zeros array:\n#{zeros}"

ones = MLX.ones([2, 2], MLX::FLOAT32)
puts "Ones array:\n#{ones}"

full = MLX.full([2, 2], 7.5, MLX::FLOAT32)
puts "Full array with 7.5:\n#{full}"

# Device information
puts "\nDevice Information:"
puts "Default device: #{MLX.get_default_device}"
puts "Available devices: #{MLX.devices.inspect}"

# Arrays and transformations
puts "\nArray Operations:"
arr1 = MLX.array([1, 2, 3, 4])
arr2 = MLX.array([5, 6, 7, 8])

concat = MLX.concatenate([arr1, arr2])
puts "Concatenated arrays: #{concat}"

split_arrays = MLX.split(concat, 2)
puts "Split arrays:"
split_arrays.each_with_index do |a, i|
  puts "  [#{i}]: #{a}"
end

stacked = MLX.stack([arr1, arr2])
puts "Stacked arrays:\n#{stacked}"

# Evaluation
arr3 = arr1 + arr2
puts "\nEvaluating Computations:"
puts "Before evaluation: #{arr3}"
MLX.eval(arr3)
puts "After evaluation: #{arr3}"

puts "\nExample completed!" 