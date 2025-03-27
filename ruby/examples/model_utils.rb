require_relative '../mlx/mlx'

puts "MLX Ruby Bindings - Model Utilities Example"
puts "------------------------------------------"

# Create some sample model weights
puts "\nCreating sample model weights..."
weights = {}
weights["embedding.weight"] = MLX.ones([100, 128])
weights["linear.weight"] = MLX.ones([128, 256])
weights["linear.bias"] = MLX.zeros([256])

puts "Model contains the following weights:"
weights.each do |key, value|
  puts "  #{key}: shape=#{value.shape.inspect}"
end

# Save the model weights
output_path = "/tmp/sample_model.safetensors"
puts "\nSaving model to #{output_path}..."
MLX::Export.to_safetensors(weights, output_path)

# Load the model weights
puts "\nLoading model from #{output_path}..."
loaded_weights = MLX.load(output_path)
puts "Loaded weights:"
loaded_weights.each do |key, value|
  puts "  #{key}: shape=#{value.shape.inspect}"
end

# Demonstrate some fast operations
puts "\nDemonstrating Fast operations..."
puts "Creating sample inputs for attention..."

# Create sample inputs for attention
batch_size = 2
seq_len = 4
hidden_size = 8
num_heads = 2

# Create query, key, value tensors
queries = MLX.reshape(MLX.array((0...batch_size*seq_len*hidden_size).to_a), [batch_size, seq_len, hidden_size])
keys = MLX.reshape(MLX.array((0...batch_size*seq_len*hidden_size).to_a), [batch_size, seq_len, hidden_size])
values = MLX.reshape(MLX.array((0...batch_size*seq_len*hidden_size).to_a), [batch_size, seq_len, hidden_size])

# Calculate attention
scale = 1.0 / Ops.sqrt(hidden_size / num_heads)
puts "Computing scaled dot product attention..."
attention = MLX::Fast.scaled_dot_product_attention(queries, keys, values, scale)
puts "Attention output shape: #{attention.shape.inspect}"

# Create inputs for layer normalization
x = MLX.reshape(MLX.array((0...hidden_size).to_a), [hidden_size])
weight = MLX.ones([hidden_size])
bias = MLX.zeros([hidden_size])

# Apply layer normalization
puts "\nApplying layer normalization..."
norm_result = MLX::Fast.layer_norm(x, weight, bias)
puts "Layer norm output: #{norm_result}"

puts "\nExample completed!" 