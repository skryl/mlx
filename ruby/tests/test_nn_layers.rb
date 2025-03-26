require_relative 'mlx_test_case'

class TestNNLayers < MLXTestCase
  def setup
    # Set random seed for reproducibility
    MLX.random.seed(42)
    
    # Common dimensions
    @batch_size = 4
    @in_features = 8
    @out_features = 6
    @seq_length = 10
    @embedding_dim = 16
    @vocab_size = 1000
    @num_heads = 4
    @channels_in = 3
    @channels_out = 16
    @height = 32
    @width = 32
  end
  
  def test_linear
    # Test basic linear layer
    layer = MLX.nn.Linear(@in_features, @out_features)
    
    # Check parameters
    assert_equal [[@out_features, @in_features], [@out_features]], [layer.weight.shape, layer.bias.shape]
    
    # Test forward pass with 2D input
    x = MLX.random.normal(shape: [@batch_size, @in_features])
    y = layer.call(x)
    
    # Check output shape
    assert_equal [@batch_size, @out_features], y.shape
    
    # Test forward pass with 3D input
    x = MLX.random.normal(shape: [@batch_size, @seq_length, @in_features])
    y = layer.call(x)
    
    # Check output shape
    assert_equal [@batch_size, @seq_length, @out_features], y.shape
    
    # Test linear layer without bias
    layer_no_bias = MLX.nn.Linear(@in_features, @out_features, bias: false)
    assert_nil layer_no_bias.bias
    
    y_no_bias = layer_no_bias.call(x)
    assert_equal [@batch_size, @seq_length, @out_features], y_no_bias.shape
    
    # Test with custom initialization
    initializer = MLX.nn.initializers.Constant(0.5)
    layer_custom_init = MLX.nn.Linear(@in_features, @out_features, 
                                  weight_initializer: initializer,
                                  bias_initializer: initializer)
    
    assert MLX.allclose(layer_custom_init.weight, MLX.full([
      @out_features, @in_features], 0.5))
    assert MLX.allclose(layer_custom_init.bias, MLX.full([@out_features], 0.5))
  end
  
  def test_conv1d
    # Test 1D convolution layer
    in_channels = 4
    out_channels = 8
    kernel_size = 3
    
    # Create input with shape [batch_size, in_channels, sequence_length]
    x = MLX.random.normal(shape: [@batch_size, in_channels, @seq_length])
    
    # Test with default parameters
    layer = MLX.nn.Conv1d(in_channels, out_channels, kernel_size)
    y = layer.call(x)
    
    # Output shape should be [batch_size, out_channels, sequence_length - kernel_size + 1]
    expected_output_length = @seq_length - kernel_size + 1
    assert_equal [@batch_size, out_channels, expected_output_length], y.shape
    
    # Test with padding
    padding = 1
    layer = MLX.nn.Conv1d(in_channels, out_channels, kernel_size, padding: padding)
    y = layer.call(x)
    
    # Output shape should be [batch_size, out_channels, sequence_length - kernel_size + 1 + 2*padding]
    expected_output_length = @seq_length - kernel_size + 1 + 2 * padding
    assert_equal [@batch_size, out_channels, expected_output_length], y.shape
    
    # Test with stride
    stride = 2
    layer = MLX.nn.Conv1d(in_channels, out_channels, kernel_size, stride: stride)
    y = layer.call(x)
    
    # Output shape should be [batch_size, out_channels, (sequence_length - kernel_size + 1) / stride]
    expected_output_length = (@seq_length - kernel_size + 1) / stride
    assert_equal [@batch_size, out_channels, expected_output_length], y.shape
    
    # Test with dilation
    dilation = 2
    layer = MLX.nn.Conv1d(in_channels, out_channels, kernel_size, dilation: dilation)
    y = layer.call(x)
    
    # Output shape with dilation
    expected_output_length = @seq_length - (kernel_size - 1) * dilation - 1
    assert_equal [@batch_size, out_channels, expected_output_length], y.shape
    
    # Test without bias
    layer = MLX.nn.Conv1d(in_channels, out_channels, kernel_size, bias: false)
    assert_nil layer.bias
    y = layer.call(x)
    assert_equal [@batch_size, out_channels, @seq_length - kernel_size + 1], y.shape
    
    # Test with groups
    groups = 2  # Must divide both in_channels and out_channels
    layer = MLX.nn.Conv1d(4, 8, kernel_size, groups: groups)
    y = layer.call(x)
    
    # Output shape should be the same as without groups
    assert_equal [@batch_size, out_channels, @seq_length - kernel_size + 1], y.shape
  end
  
  def test_conv2d
    # Test 2D convolution layer
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    
    # Create input with shape [batch_size, in_channels, height, width]
    x = MLX.random.normal(shape: [@batch_size, in_channels, @height, @width])
    
    # Test with default parameters
    layer = MLX.nn.Conv2d(in_channels, out_channels, kernel_size)
    y = layer.call(x)
    
    # Output shape should be [batch_size, out_channels, height - kernel_size + 1, width - kernel_size + 1]
    expected_output_height = @height - kernel_size + 1
    expected_output_width = @width - kernel_size + 1
    assert_equal [@batch_size, out_channels, expected_output_height, expected_output_width], y.shape
    
    # Test with padding
    padding = 1
    layer = MLX.nn.Conv2d(in_channels, out_channels, kernel_size, padding: padding)
    y = layer.call(x)
    
    # Output shape with padding
    expected_output_height = @height - kernel_size + 1 + 2 * padding
    expected_output_width = @width - kernel_size + 1 + 2 * padding
    assert_equal [@batch_size, out_channels, expected_output_height, expected_output_width], y.shape
    
    # Test with stride
    stride = 2
    layer = MLX.nn.Conv2d(in_channels, out_channels, kernel_size, stride: stride)
    y = layer.call(x)
    
    # Output shape with stride
    expected_output_height = (@height - kernel_size + 1) / stride
    expected_output_width = (@width - kernel_size + 1) / stride
    assert_equal [@batch_size, out_channels, expected_output_height, expected_output_width], y.shape
    
    # Test with dilation
    dilation = 2
    layer = MLX.nn.Conv2d(in_channels, out_channels, kernel_size, dilation: dilation)
    y = layer.call(x)
    
    # Output shape with dilation
    expected_output_height = @height - (kernel_size - 1) * dilation - 1
    expected_output_width = @width - (kernel_size - 1) * dilation - 1
    assert_equal [@batch_size, out_channels, expected_output_height, expected_output_width], y.shape
    
    # Test with rectangular kernel
    kernel_size = [3, 5]
    layer = MLX.nn.Conv2d(in_channels, out_channels, kernel_size)
    y = layer.call(x)
    
    # Output shape with rectangular kernel
    expected_output_height = @height - kernel_size[0] + 1
    expected_output_width = @width - kernel_size[1] + 1
    assert_equal [@batch_size, out_channels, expected_output_height, expected_output_width], y.shape
    
    # Test without bias
    layer = MLX.nn.Conv2d(in_channels, out_channels, 3, bias: false)
    assert_nil layer.bias
    y = layer.call(x)
    
    # Output shape should be the same as with default parameters
    expected_output_height = @height - 3 + 1
    expected_output_width = @width - 3 + 1
    assert_equal [@batch_size, out_channels, expected_output_height, expected_output_width], y.shape
  end
  
  def test_embedding
    # Test embedding layer
    layer = MLX.nn.Embedding(@vocab_size, @embedding_dim)
    
    # Check parameter shapes
    assert_equal [@vocab_size, @embedding_dim], layer.weight.shape
    
    # Test with 1D input (single batch)
    x = MLX.array([1, 2, 3, 4, 5])
    y = layer.call(x)
    
    # Output shape should be [5, embedding_dim]
    assert_equal [5, @embedding_dim], y.shape
    
    # Test with 2D input (batch)
    x = MLX.array([[1, 2, 3], [4, 5, 6]])
    y = layer.call(x)
    
    # Output shape should be [2, 3, embedding_dim]
    assert_equal [2, 3, @embedding_dim], y.shape
    
    # Test embedding lookup (the first row of output should be the first row of the embedding table)
    x = MLX.array([0])
    y = layer.call(x)
    assert MLX.array_equal(y[0], layer.weight[0])
    
    # Test with OOV indices
    x = MLX.array([@vocab_size + 10])  # Out of range
    # This should not raise an error because MLX uses modulo indexing
    y = layer.call(x)
    
    # Test with custom initialization
    initializer = MLX.nn.initializers.Constant(0.5)
    layer = MLX.nn.Embedding(@vocab_size, @embedding_dim, weight_initializer: initializer)
    assert MLX.allclose(layer.weight, MLX.full([@vocab_size, @embedding_dim], 0.5))
  end
  
  def test_layer_norm
    # Test layer normalization
    normalized_shape = [@in_features]
    layer = MLX.nn.LayerNorm(normalized_shape)
    
    # Check parameters
    assert_equal [@in_features], layer.weight.shape
    assert_equal [@in_features], layer.bias.shape
    
    # Test with 2D input
    x = MLX.random.normal(shape: [@batch_size, @in_features])
    y = layer.call(x)
    
    # Check output shape
    assert_equal x.shape, y.shape
    
    # Test with 3D input
    x = MLX.random.normal(shape: [@batch_size, @seq_length, @in_features])
    y = layer.call(x)
    
    # Check output shape
    assert_equal x.shape, y.shape
    
    # Test without bias
    layer = MLX.nn.LayerNorm(normalized_shape, bias: false)
    assert_nil layer.bias
    y = layer.call(x)
    assert_equal x.shape, y.shape
    
    # Test with custom epsilon
    layer = MLX.nn.LayerNorm(normalized_shape, eps: 1e-3)
    y = layer.call(x)
    assert_equal x.shape, y.shape
    
    # Test with larger normalized shape
    normalized_shape = [@seq_length, @in_features]
    layer = MLX.nn.LayerNorm(normalized_shape)
    
    # Parameter shapes should match the normalized shape
    assert_equal normalized_shape, layer.weight.shape
    assert_equal normalized_shape, layer.bias.shape
    
    y = layer.call(x)
    assert_equal x.shape, y.shape
  end
  
  def test_batch_norm
    # Test batch normalization
    layer = MLX.nn.BatchNorm2d(@channels_in)
    
    # Check parameters
    assert_equal [@channels_in], layer.weight.shape
    assert_equal [@channels_in], layer.bias.shape
    assert_equal [@channels_in], layer.running_mean.shape
    assert_equal [@channels_in], layer.running_var.shape
    
    # Test in training mode
    x = MLX.random.normal(shape: [@batch_size, @channels_in, @height, @width])
    y = layer.call(x, training: true)
    
    # Check output shape
    assert_equal x.shape, y.shape
    
    # Test in evaluation mode
    y = layer.call(x, training: false)
    assert_equal x.shape, y.shape
    
    # Test BatchNorm1d
    layer = MLX.nn.BatchNorm1d(@in_features)
    x = MLX.random.normal(shape: [@batch_size, @in_features])
    y = layer.call(x)
    assert_equal x.shape, y.shape
    
    # Test without affine parameters
    layer = MLX.nn.BatchNorm2d(@channels_in, affine: false)
    assert_nil layer.weight
    assert_nil layer.bias
    y = layer.call(x, training: true)
    assert_equal x.shape, y.shape
    
    # Test with momentum
    layer = MLX.nn.BatchNorm2d(@channels_in, momentum: 0.9)
    y = layer.call(x, training: true)
    assert_equal x.shape, y.shape
    
    # Test BatchNorm3d
    channels = 4
    depth = 8
    layer = MLX.nn.BatchNorm3d(channels)
    x = MLX.random.normal(shape: [@batch_size, channels, depth, @height, @width])
    y = layer.call(x)
    assert_equal x.shape, y.shape
  end
  
  def test_instance_norm
    # Test instance normalization
    layer = MLX.nn.InstanceNorm2d(@channels_in)
    
    # Check parameters
    assert_equal [@channels_in], layer.weight.shape
    assert_equal [@channels_in], layer.bias.shape
    
    # Test forward pass
    x = MLX.random.normal(shape: [@batch_size, @channels_in, @height, @width])
    y = layer.call(x)
    
    # Check output shape
    assert_equal x.shape, y.shape
    
    # Test without affine parameters
    layer = MLX.nn.InstanceNorm2d(@channels_in, affine: false)
    assert_nil layer.weight
    assert_nil layer.bias
    y = layer.call(x)
    assert_equal x.shape, y.shape
    
    # Test InstanceNorm1d
    layer = MLX.nn.InstanceNorm1d(@in_features)
    x = MLX.random.normal(shape: [@batch_size, @in_features, @seq_length])
    y = layer.call(x)
    assert_equal x.shape, y.shape
    
    # Test InstanceNorm3d
    channels = 4
    depth = 8
    layer = MLX.nn.InstanceNorm3d(channels)
    x = MLX.random.normal(shape: [@batch_size, channels, depth, @height, @width])
    y = layer.call(x)
    assert_equal x.shape, y.shape
  end
  
  def test_group_norm
    # Test group normalization
    num_groups = 2  # Must divide channels
    channels = 4
    layer = MLX.nn.GroupNorm(num_groups, channels)
    
    # Check parameters
    assert_equal [channels], layer.weight.shape
    assert_equal [channels], layer.bias.shape
    
    # Test forward pass
    x = MLX.random.normal(shape: [@batch_size, channels, @height, @width])
    y = layer.call(x)
    
    # Check output shape
    assert_equal x.shape, y.shape
    
    # Test without affine parameters
    layer = MLX.nn.GroupNorm(num_groups, channels, affine: false)
    assert_nil layer.weight
    assert_nil layer.bias
    y = layer.call(x)
    assert_equal x.shape, y.shape
    
    # Test with 1 group (should be similar to layer norm)
    layer = MLX.nn.GroupNorm(1, channels)
    y = layer.call(x)
    assert_equal x.shape, y.shape
    
    # Test with num_groups = channels (should be similar to instance norm)
    layer = MLX.nn.GroupNorm(channels, channels)
    y = layer.call(x)
    assert_equal x.shape, y.shape
  end
  
  def test_dropout
    # Test dropout layer
    rate = 0.5
    layer = MLX.nn.Dropout(rate)
    
    # Test in training mode
    x = MLX.ones([@batch_size, @in_features])
    y = layer.call(x, training: true)
    
    # Check output shape
    assert_equal x.shape, y.shape
    
    # Check that some elements are dropped
    zeros_count = MLX.sum(y == 0).item
    assert zeros_count > 0
    
    # Test in evaluation mode
    y = layer.call(x, training: false)
    assert MLX.array_equal(x, y)
    
    # Test Dropout2d
    layer = MLX.nn.Dropout2d(rate)
    x = MLX.ones([@batch_size, @channels_in, @height, @width])
    y = layer.call(x, training: true)
    assert_equal x.shape, y.shape
    
    # Test Dropout3d
    layer = MLX.nn.Dropout3d(rate)
    x = MLX.ones([@batch_size, @channels_in, 4, @height, @width])
    y = layer.call(x, training: true)
    assert_equal x.shape, y.shape
  end
  
  def test_flatten
    # Test flatten layer
    layer = MLX.nn.Flatten()
    
    # Test with 4D input
    x = MLX.random.normal(shape: [@batch_size, @channels_in, @height, @width])
    y = layer.call(x)
    
    # Output shape should be [batch_size, channels_in * height * width]
    assert_equal [@batch_size, @channels_in * @height * @width], y.shape
    
    # Test with 3D input
    x = MLX.random.normal(shape: [@batch_size, @seq_length, @in_features])
    y = layer.call(x)
    
    # Output shape should be [batch_size, seq_length * in_features]
    assert_equal [@batch_size, @seq_length * @in_features], y.shape
    
    # Test with start_dim and end_dim
    layer = MLX.nn.Flatten(start_dim: 2)
    x = MLX.random.normal(shape: [@batch_size, @channels_in, @height, @width])
    y = layer.call(x)
    
    # Output shape should be [batch_size, channels_in, height * width]
    assert_equal [@batch_size, @channels_in, @height * @width], y.shape
    
    # Test with 2D input (should be unchanged)
    x = MLX.random.normal(shape: [@batch_size, @in_features])
    y = MLX.nn.Flatten().call(x)
    assert_equal x.shape, y.shape
  end
  
  def test_identity
    # Test identity layer
    layer = MLX.nn.Identity()
    
    # Test with various input shapes
    shapes = [
      [@batch_size],
      [@batch_size, @in_features],
      [@batch_size, @seq_length, @in_features],
      [@batch_size, @channels_in, @height, @width]
    ]
    
    shapes.each do |shape|
      x = MLX.random.normal(shape: shape)
      y = layer.call(x)
      
      # Output should be identical to input
      assert_equal shape, y.shape
      assert MLX.array_equal(x, y)
    end
  end
  
  def test_pooling_layers
    # Test max pooling layers
    x = MLX.random.normal(shape: [@batch_size, @channels_in, @height, @width])
    
    # Test MaxPool2d
    kernel_size = 2
    layer = MLX.nn.MaxPool2d(kernel_size)
    y = layer.call(x)
    assert_equal [@batch_size, @channels_in, @height / kernel_size, @width / kernel_size], y.shape
    
    # Test with stride
    stride = 1
    layer = MLX.nn.MaxPool2d(kernel_size, stride: stride)
    y = layer.call(x)
    expected_height = @height - kernel_size + 1
    expected_width = @width - kernel_size + 1
    assert_equal [@batch_size, @channels_in, expected_height, expected_width], y.shape
    
    # Test with padding
    padding = 1
    layer = MLX.nn.MaxPool2d(kernel_size, padding: padding)
    y = layer.call(x)
    expected_height = (@height + 2 * padding - kernel_size) / 1 + 1
    expected_width = (@width + 2 * padding - kernel_size) / 1 + 1
    assert_equal [@batch_size, @channels_in, expected_height, expected_width], y.shape
    
    # Test AvgPool2d
    layer = MLX.nn.AvgPool2d(kernel_size)
    y = layer.call(x)
    assert_equal [@batch_size, @channels_in, @height / kernel_size, @width / kernel_size], y.shape
    
    # Test GlobalAvgPool2d
    layer = MLX.nn.GlobalAvgPool2d()
    y = layer.call(x)
    assert_equal [@batch_size, @channels_in, 1, 1], y.shape
    
    # Test GlobalMaxPool2d
    layer = MLX.nn.GlobalMaxPool2d()
    y = layer.call(x)
    assert_equal [@batch_size, @channels_in, 1, 1], y.shape
    
    # Test 1D pooling
    x_1d = MLX.random.normal(shape: [@batch_size, @channels_in, @seq_length])
    
    # Test MaxPool1d
    layer = MLX.nn.MaxPool1d(kernel_size)
    y = layer.call(x_1d)
    assert_equal [@batch_size, @channels_in, @seq_length / kernel_size], y.shape
    
    # Test AvgPool1d
    layer = MLX.nn.AvgPool1d(kernel_size)
    y = layer.call(x_1d)
    assert_equal [@batch_size, @channels_in, @seq_length / kernel_size], y.shape
  end
end 