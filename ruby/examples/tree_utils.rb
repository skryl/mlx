#!/usr/bin/env ruby
# Example of tree manipulation utilities in MLX Ruby bindings

require 'mlx'

# Create some nested data structures
tree1 = {
  "a" => MLX.array([1, 2, 3]),
  "b" => {
    "c" => MLX.array([4, 5, 6]),
    "d" => MLX.array([7, 8, 9])
  },
  "e" => [
    MLX.array([10, 11, 12]),
    MLX.array([13, 14, 15])
  ]
}

tree2 = {
  "a" => MLX.array([2, 3, 4]),
  "b" => {
    "c" => MLX.array([5, 6, 7]),
    "d" => MLX.array([8, 9, 10])
  },
  "e" => [
    MLX.array([11, 12, 13]),
    MLX.array([14, 15, 16])
  ]
}

puts "Example 1: Tree Map"
puts "----------------"
puts "Applying a function to every array in a nested structure"

# Apply a function to each array in the tree
doubled = MLX::Utils.tree_map(->(x) { x * 2 if x.is_a?(MLX::Array) }, tree1)

# Print original and result
puts "Original values:"
MLX::Utils.tree_map(->(x) { puts x.inspect if x.is_a?(MLX::Array) }, tree1)
puts "\nDoubled values:"
MLX::Utils.tree_map(->(x) { puts x.inspect if x.is_a?(MLX::Array) }, doubled)

puts "\nExample 2: Tree Flatten"
puts "----------------"
puts "Flattening a nested structure into a list"

# Flatten the tree into a list of arrays
flattened = MLX::Utils.tree_flatten(tree1)
puts "Flattened tree contains #{flattened.length} arrays:"
flattened.each { |x| puts x.inspect }

puts "\nExample 3: Tree Unflatten"
puts "----------------"
puts "Reconstructing a nested structure from a flattened list"

# Flatten and then unflatten to demonstrate round-trip
flat_values = MLX::Utils.tree_flatten(tree1)
template = MLX::Utils.tree_map(->(x) { nil if x.is_a?(MLX::Array) }, tree1)
reconstructed = MLX::Utils.tree_unflatten(flat_values, template)

# Verify the reconstruction
puts "Original structure:"
puts tree1.inspect
puts "\nReconstructed structure:"
puts reconstructed.inspect

puts "\nExample 4: Tree Reduce"
puts "----------------"
puts "Reducing a tree to a single value"

# Compute the sum of all array elements in the tree
sum = MLX::Utils.tree_reduce(
  ->(acc, x) { acc + MLX.sum(x).item if x.is_a?(MLX::Array) },
  tree1,
  initial_value: 0
)
puts "Sum of all elements: #{sum}"

puts "\nExample 5: Tree Map with Path"
puts "----------------"
puts "Mapping with access to the parameter path"

# Apply a function that uses the path information
path_aware = MLX::Utils.tree_map_with_path(->(path, x) {
  if x.is_a?(MLX::Array)
    puts "At path '#{path}': #{x.shape.inspect}"
    x * (path.include?("b") ? 3 : 1)  # Multiply by 3 only in the "b" subtree
  else
    x
  end
}, tree1)

puts "\nExample 6: Tree Merge"
puts "----------------"
puts "Merging two trees with a combining function"

# Merge the two trees, adding the arrays together
merged = MLX::Utils.tree_merge(
  ->(x, y) { x + y if x.is_a?(MLX::Array) && y.is_a?(MLX::Array) },
  tree1,
  tree2
)

puts "Result of merging (adding arrays):"
MLX::Utils.tree_map(->(x) { puts x.inspect if x.is_a?(MLX::Array) }, merged)

puts "\nExample 7: Practical Application - Gradient Clipping"
puts "----------------"
puts "Using tree utilities for gradient processing"

# Create a simple model
model = MLX::NN::Module.new
model.linear1 = MLX::NN::Layers::Linear.new(10, 20)
model.linear2 = MLX::NN::Layers::Linear.new(20, 1)

# Simulate computing gradients
grads = {
  "linear1" => {
    "weight" => MLX.random.normal([20, 10]) * 10,  # Large gradients
    "bias" => MLX.random.normal([20]) * 10
  },
  "linear2" => {
    "weight" => MLX.random.normal([1, 20]) * 10,
    "bias" => MLX.random.normal([1]) * 10
  }
}

# Compute gradient norm
grad_norm = MLX.sqrt(
  MLX::Utils.tree_reduce(
    ->(acc, x) { acc + MLX.sum(x * x).item if x.is_a?(MLX::Array) },
    grads,
    initial_value: 0.0
  )
)
puts "Gradient norm before clipping: #{grad_norm.round(4)}"

# Clip gradients to max norm of 5.0
max_norm = 5.0
if grad_norm > max_norm
  scale = max_norm / grad_norm
  clipped_grads = MLX::Utils.tree_map(->(x) { x * scale if x.is_a?(MLX::Array) }, grads)
  
  # Verify new norm
  new_norm = MLX.sqrt(
    MLX::Utils.tree_reduce(
      ->(acc, x) { acc + MLX.sum(x * x).item if x.is_a?(MLX::Array) },
      clipped_grads,
      initial_value: 0.0
    )
  )
  puts "Gradient norm after clipping: #{new_norm.round(4)}"
end 