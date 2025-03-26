require_relative 'mlx_test_case'

class TestQuantization < MLXTestCase
  def setup
    super
    
    # Skip all tests if quantization is not available
    skip unless defined?(MLX::NN::Layers::QuantizedLinear)
  end
  
  def test_quantized_linear
    # Create a regular linear layer
    linear = MLX::NN::Layers::Linear.new(10, 5)
    
    # Quantize it
    quantized = MLX::NN::Layers::QuantizedLinear.from_linear(
      linear,
      weight_params: { group_size: 4, bits: 4 }
    )
    
    # Check that parameters have the expected types and shapes
    assert quantized.parameters.key?("qweight")
    assert quantized.parameters.key?("scales")
    
    # Expected shapes for quantized parameters
    assert_equal [5, 10], quantized.parameters["qweight"].shape
    assert_equal [5, 10 / 4], quantized.parameters["scales"].shape
    
    # Forward pass
    x = MLX.random.normal([2, 10])
    output = quantized.call(x)
    
    # Output should have the expected shape
    assert_equal [2, 5], output.shape
  end
  
  def test_quantized_embedding
    # Create a regular embedding layer
    embedding = MLX::NN::Layers::Embedding.new(100, 16)
    
    # Quantize it
    quantized = MLX::NN::Layers::QuantizedEmbedding.from_embedding(
      embedding,
      weight_params: { group_size: 4, bits: 4 }
    )
    
    # Check that parameters have the expected types and shapes
    assert quantized.parameters.key?("qweight")
    assert quantized.parameters.key?("scales")
    
    # Expected shapes for quantized parameters
    assert_equal [100, 16], quantized.parameters["qweight"].shape
    assert_equal [100, 16 / 4], quantized.parameters["scales"].shape
    
    # Forward pass
    indices = MLX.array([0, 10, 20, 30])
    output = quantized.call(indices)
    
    # Output should have the expected shape
    assert_equal [4, 16], output.shape
  end
  
  def test_model_quantization
    # Create a simple model
    model = MLX::NN::Sequential.new(
      MLX::NN::Layers::Linear.new(10, 20),
      MLX::NN::Layers::ReLU.new,
      MLX::NN::Layers::Linear.new(20, 5)
    )
    
    # Quantize the model
    quantized_model = MLX::NN::Layers.quantize(
      model,
      weight_params: { group_size: 4, bits: 4 }
    )
    
    # Check that model structure is preserved
    assert_equal 3, quantized_model.modules.length
    
    # Check that linear layers are quantized
    assert quantized_model.modules[0].is_a?(MLX::NN::Layers::QuantizedLinear)
    assert quantized_model.modules[1].is_a?(MLX::NN::Layers::ReLU)
    assert quantized_model.modules[2].is_a?(MLX::NN::Layers::QuantizedLinear)
    
    # Forward pass
    x = MLX.random.normal([2, 10])
    output = quantized_model.call(x)
    
    # Output should have the expected shape
    assert_equal [2, 5], output.shape
  end
  
  def test_different_quantization_parameters
    # Create a regular linear layer
    linear = MLX::NN::Layers::Linear.new(10, 5)
    
    # Test different quantization bit widths
    [2, 4, 8].each do |bits|
      # Quantize with current bit width
      quantized = MLX::NN::Layers::QuantizedLinear.from_linear(
        linear,
        weight_params: { group_size: 4, bits: bits }
      )
      
      # Check that quantization worked
      assert quantized.parameters.key?("qweight")
      
      # Forward pass
      x = MLX.random.normal([2, 10])
      output = quantized.call(x)
      
      # Output should have the expected shape
      assert_equal [2, 5], output.shape
    end
    
    # Test different group sizes
    [2, 4, 8].each do |group_size|
      next unless 10 % group_size == 0  # Skip if not divisible
      
      # Quantize with current group size
      quantized = MLX::NN::Layers::QuantizedLinear.from_linear(
        linear,
        weight_params: { group_size: group_size, bits: 4 }
      )
      
      # Check that quantization worked
      assert quantized.parameters.key?("qweight")
      
      # Forward pass
      x = MLX.random.normal([2, 10])
      output = quantized.call(x)
      
      # Output should have the expected shape
      assert_equal [2, 5], output.shape
    end
  end
  
  def test_quantized_distributed_linear
    # Skip if distributed linear is not available
    skip unless defined?(MLX::NN::Layers::QuantizedShardedToAllLinear)
    
    # Create a regular linear layer
    linear = MLX::NN::Layers::Linear.new(10, 5)
    
    # First quantize it
    quantized = MLX::NN::Layers::QuantizedLinear.from_linear(
      linear,
      weight_params: { group_size: 2, bits: 4 }
    )
    
    # Initialize distributed environment
    world = MLX::Distributed.init
    
    # Create distributed quantized layer
    distributed = MLX::NN::Layers.shard_linear(
      quantized,
      "all-to-sharded",
      group: world
    )
    
    # Forward pass
    x = MLX.random.normal([2, 10])
    output = distributed.call(x)
    
    # Output should have the expected shape
    assert_equal [2, 5], output.shape
  end
  
  def test_dequantize_function
    # Create a regular linear layer
    linear = MLX::NN::Layers::Linear.new(10, 5)
    
    # Quantize it
    quantized = MLX::NN::Layers::QuantizedLinear.from_linear(
      linear,
      weight_params: { group_size: 2, bits: 4 }
    )
    
    # Check if dequantize method exists
    if quantized.respond_to?(:dequantize_weight)
      # Get dequantized weight
      dequantized = quantized.dequantize_weight
      
      # It should be a float array with the same shape as the original weight
      assert_equal MLX::FLOAT32, dequantized.dtype
      assert_equal [5, 10], dequantized.shape
    end
  end
end 