#!/usr/bin/env ruby
# Example of model quantization using MLX Ruby bindings

require 'mlx'

# Create a simple transformer model
class SimpleTransformer < MLX::NN::Module
  attr_reader :embedding, :attention, :ffn, :layernorm1, :layernorm2, :output_layer
  
  def initialize(vocab_size, d_model, nhead, d_ff, dropout = 0.1)
    super()
    
    @d_model = d_model
    
    # Embedding layer
    @embedding = MLX::NN::Layers::Embedding.new(vocab_size, d_model)
    
    # Multi-head attention
    @attention = MLX::NN::Layers::MultiheadAttention.new(d_model, nhead, dropout: dropout)
    
    # Feed-forward network
    @ffn = MLX::NN::Sequential.new(
      MLX::NN::Layers::Linear.new(d_model, d_ff),
      MLX::NN::Layers::ReLU.new,
      MLX::NN::Layers::Dropout.new(dropout),
      MLX::NN::Layers::Linear.new(d_ff, d_model)
    )
    
    # Layer normalization
    @layernorm1 = MLX::NN::Layers::LayerNorm.new(d_model)
    @layernorm2 = MLX::NN::Layers::LayerNorm.new(d_model)
    
    # Output layer
    @output_layer = MLX::NN::Layers::Linear.new(d_model, vocab_size)
    
    # Dropout
    @dropout = MLX::NN::Layers::Dropout.new(dropout)
  end
  
  def forward(x, mask = nil)
    # Embedding
    x = @embedding.call(x)
    x = x * Math.sqrt(@d_model)
    
    # Self-attention
    attn_output = @attention.call(x, x, x, mask: mask)
    x = @layernorm1.call(x + @dropout.call(attn_output))
    
    # Feed-forward
    ffn_output = @ffn.call(x)
    x = @layernorm2.call(x + @dropout.call(ffn_output))
    
    # Output projection
    @output_layer.call(x)
  end
end

# Create a model
vocab_size = 10000
d_model = 512
nhead = 8
d_ff = 2048

puts "Creating a transformer model..."
model = SimpleTransformer.new(vocab_size, d_model, nhead, d_ff)

# Print model structure
puts "Model structure:"
puts model.inspect

# Function to test the model with random input
def test_model(model, sequence_length, vocab_size)
  # Generate random input
  input_sequence = MLX.random.randint(0, vocab_size, [1, sequence_length])
  
  # Forward pass
  start_time = Time.now
  output = model.call(input_sequence)
  end_time = Time.now
  
  # Get output shape
  output_shape = output.shape
  
  # Compute memory usage (approximate)
  param_count = model.parameters.values.sum { |p| p.size }
  memory_mb = (param_count * 4) / (1024 * 1024.0)  # Assuming 4 bytes per parameter
  
  {
    output_shape: output_shape,
    time_ms: ((end_time - start_time) * 1000).round(2),
    memory_mb: memory_mb.round(2)
  }
end

# Test the model before quantization
puts "\nTesting original model..."
sequence_length = 64
result_original = test_model(model, sequence_length, vocab_size)
puts "Output shape: #{result_original[:output_shape]}"
puts "Inference time: #{result_original[:time_ms]} ms"
puts "Model size: #{result_original[:memory_mb]} MB"

# Quantize the model
puts "\nQuantizing model..."
weight_params = { group_size: 64, bits: 4 }
quantized_model = MLX::NN::Layers.quantize(model, weight_params: weight_params)

# Test the quantized model
puts "\nTesting quantized model..."
result_quantized = test_model(quantized_model, sequence_length, vocab_size)
puts "Output shape: #{result_quantized[:output_shape]}"
puts "Inference time: #{result_quantized[:time_ms]} ms"
puts "Model size: #{result_quantized[:memory_mb]} MB"

# Compare results
puts "\nComparison:"
puts "Size reduction: #{((1 - result_quantized[:memory_mb] / result_original[:memory_mb]) * 100).round(2)}%"
puts "Speed change: #{((result_quantized[:time_ms] / result_original[:time_ms] - 1) * 100).round(2)}%"

# Check for accuracy difference
puts "\nComparing outputs..."
original_output = model.call(MLX.random.randint(0, vocab_size, [1, sequence_length]))
quantized_output = quantized_model.call(MLX.random.randint(0, vocab_size, [1, sequence_length]))

# Calculate mean absolute error between original and quantized outputs
mae = MLX.mean(MLX.abs(original_output - quantized_output)).item
puts "Mean Absolute Error: #{mae.round(6)}"
puts "This represents the average quantization error per output value"

puts "\nQuantization complete!" 