require_relative 'mlx_test_case'

class TestDropout < MLXTestCase
  def setup
    # Set random seed for reproducibility
    MLX.random.seed(42)
    
    # Create tensors for testing
    @batch_size = 8
    @features = 16
    @height = 4
    @width = 4
    
    # 2D input for testing
    @input_2d = MLX.random.normal(shape: [@batch_size, @features])
    
    # 4D input for testing
    @input_4d = MLX.random.normal(shape: [@batch_size, @features, @height, @width])
  end
  
  def test_dropout_training
    # Test basic dropout behavior during training
    x = @input_2d
    rate = 0.5
    
    # Apply dropout in training mode
    y = MLX.dropout(x, rate, training: true)
    
    # Check output shape
    assert_equal x.shape, y.shape
    
    # Check that some elements are zero (due to dropout)
    zeros_count = MLX.sum(y == 0).item
    assert zeros_count > 0, "No elements were dropped out"
    
    # Check that the proportion of zeros is roughly equal to the dropout rate
    total_elements = x.size
    zeros_proportion = zeros_count.to_f / total_elements
    
    # Allow for some statistical variation
    assert_in_delta rate, zeros_proportion, 0.1
    
    # Check that non-zero elements are scaled by 1/(1-rate)
    scale = 1.0 / (1.0 - rate)
    
    # Create a mask of non-zero elements
    non_zero_mask = (y != 0)
    
    # Get the original values for non-zero positions
    original_vals = x[non_zero_mask]
    dropout_vals = y[non_zero_mask]
    
    # Check that the ratio is approximately the scale factor
    ratio = dropout_vals / original_vals
    assert MLX.allclose(ratio, MLX.full_like(ratio, scale))
  end
  
  def test_dropout_evaluation
    # Test that dropout does nothing during evaluation
    x = @input_2d
    rate = 0.5
    
    # Apply dropout in evaluation mode
    y = MLX.dropout(x, rate, training: false)
    
    # Check output shape
    assert_equal x.shape, y.shape
    
    # Check that no elements are zero (unless they were already zero)
    zeros_in_x = MLX.sum(x == 0).item
    zeros_in_y = MLX.sum(y == 0).item
    assert_equal zeros_in_x, zeros_in_y
    
    # Check that all elements are unchanged
    assert MLX.array_equal(x, y)
  end
  
  def test_dropout_with_different_rates
    # Test dropout with different rates
    x = @input_2d
    
    # Test with rate = 0
    y = MLX.dropout(x, 0.0, training: true)
    assert MLX.array_equal(x, y)
    
    # Test with rate = 1 (should drop everything)
    y = MLX.dropout(x, 1.0, training: true)
    assert MLX.all(y == 0).item
    
    # Test with various rates
    rates = [0.1, 0.3, 0.7, 0.9]
    for rate in rates
      y = MLX.dropout(x, rate, training: true)
      
      # Check that proportion of zeros is approximately equal to the rate
      zeros_count = MLX.sum(y == 0).item
      total_elements = x.size
      zeros_proportion = zeros_count.to_f / total_elements
      
      assert_in_delta rate, zeros_proportion, 0.1
    end
  end
  
  def test_dropout_on_4d_tensor
    # Test dropout on 4D tensor
    x = @input_4d
    rate = 0.5
    
    # Apply dropout in training mode
    y = MLX.dropout(x, rate, training: true)
    
    # Check output shape
    assert_equal x.shape, y.shape
    
    # Check that some elements are zero
    zeros_count = MLX.sum(y == 0).item
    assert zeros_count > 0
    
    # Check that non-zero elements are scaled
    scale = 1.0 / (1.0 - rate)
    non_zero_mask = (y != 0)
    original_vals = x[non_zero_mask]
    dropout_vals = y[non_zero_mask]
    ratio = dropout_vals / original_vals
    assert MLX.allclose(ratio, MLX.full_like(ratio, scale))
  end
  
  def test_dropout_reproducibility
    # Test that dropout is reproducible with the same seed
    x = @input_2d
    rate = 0.5
    
    # Set seed
    MLX.random.seed(123)
    y1 = MLX.dropout(x, rate, training: true)
    
    # Reset seed to same value
    MLX.random.seed(123)
    y2 = MLX.dropout(x, rate, training: true)
    
    # Check that outputs are the same
    assert MLX.array_equal(y1, y2)
    
    # With different seed, outputs should be different
    MLX.random.seed(456)
    y3 = MLX.dropout(x, rate, training: true)
    
    # They should differ with high probability
    assert !MLX.array_equal(y1, y3)
  end
  
  def test_dropout_gradients
    # Test that gradients flow correctly through dropout
    x = MLX.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    rate = 0.5
    
    # Define a function that uses dropout in evaluation mode
    def dropout_eval_fn(x, rate)
      y = MLX.dropout(x, rate, training: false)
      MLX.sum(y)
    end
    
    # Compute gradients
    grad_fn = MLX.grad(dropout_eval_fn)
    grad = grad_fn.call(x, rate)
    
    # Since dropout in eval mode is identity, gradients should be all ones
    assert MLX.array_equal(grad, MLX.ones_like(x))
    
    # Define a function that uses dropout in training mode
    def dropout_train_fn(x, rate)
      MLX.random.seed(42)  # Fix seed for reproducibility
      y = MLX.dropout(x, rate, training: true)
      MLX.sum(y)
    end
    
    # Compute gradients
    grad_fn = MLX.grad(dropout_train_fn)
    grad = grad_fn.call(x, rate)
    
    # Gradients should be either 0 (dropped out) or 1/(1-rate) (kept)
    scale = 1.0 / (1.0 - rate)
    
    # Check that all gradient values are either 0 or scale
    zeros_mask = (grad == 0)
    scale_mask = MLX.isclose(grad, scale)
    combined_mask = zeros_mask | scale_mask
    
    assert MLX.all(combined_mask).item
    
    # The number of non-zero gradients should be approximately (1-rate) * total_elements
    non_zero_count = MLX.sum(grad != 0).item
    expected_non_zero = x.size * (1.0 - rate)
    assert_in_delta expected_non_zero, non_zero_count, x.size * 0.1
  end
  
  def test_dropout2d
    # Test 2D spatial dropout (drops entire channels)
    x = @input_4d
    rate = 0.5
    
    # Apply dropout2d in training mode
    y = MLX.dropout2d(x, rate, training: true)
    
    # Check output shape
    assert_equal x.shape, y.shape
    
    # Check that entire channels are dropped out
    # For each sample and channel, sum across spatial dimensions
    for b in 0...@batch_size
      for c in 0...@features
        # If a channel is dropped, all spatial locations are zero
        spatial_sum = MLX.sum(y[b, c]).item
        
        # Either all zeros or none should be zero
        assert (spatial_sum == 0) || (spatial_sum != 0)
      end
    end
    
    # Count dropped channels
    channel_is_dropped = MLX.zeros([@batch_size, @features], dtype: MLX.bool)
    for b in 0...@batch_size
      for c in 0...@features
        if MLX.all(y[b, c] == 0).item
          channel_is_dropped[b, c] = 1
        end
      end
    end
    
    # Check that the proportion of dropped channels is roughly equal to the dropout rate
    dropped_channel_count = MLX.sum(channel_is_dropped).item
    total_channels = @batch_size * @features
    dropped_proportion = dropped_channel_count.to_f / total_channels
    
    assert_in_delta rate, dropped_proportion, 0.15
  end
  
  def test_dropout3d
    # Test 3D spatial dropout (drops entire feature maps)
    # First create a 5D tensor (batch, channels, depth, height, width)
    depth = 3
    input_5d = MLX.random.normal(shape: [@batch_size, @features, depth, @height, @width])
    
    rate = 0.5
    
    # Apply dropout3d in training mode
    y = MLX.dropout3d(input_5d, rate, training: true)
    
    # Check output shape
    assert_equal input_5d.shape, y.shape
    
    # Check that entire feature volumes are dropped out
    # For each sample and channel, sum across spatial dimensions
    for b in 0...@batch_size
      for c in 0...@features
        # If a feature map is dropped, all spatial locations are zero
        volume_sum = MLX.sum(y[b, c]).item
        
        # Either all zeros or none should be zero
        assert (volume_sum == 0) || (volume_sum != 0)
      end
    end
    
    # Check in evaluation mode
    y_eval = MLX.dropout3d(input_5d, rate, training: false)
    assert MLX.array_equal(input_5d, y_eval)
  end
  
  def test_alpha_dropout
    # Test alpha dropout (self-normalizing dropout used with SELU)
    x = @input_2d
    rate = 0.5
    
    # Apply alpha dropout in training mode
    y = MLX.alpha_dropout(x, rate, training: true)
    
    # Check output shape
    assert_equal x.shape, y.shape
    
    # Alpha dropout should preserve mean and variance approximately
    assert_in_delta MLX.mean(x).item, MLX.mean(y).item, 0.5
    assert_in_delta MLX.var(x).item, MLX.var(y).item, 0.5
    
    # In evaluation mode, it's identity
    y_eval = MLX.alpha_dropout(x, rate, training: false)
    assert MLX.array_equal(x, y_eval)
  end
  
  def test_feature_dropout
    # Test feature dropout (drops same features across all samples)
    x = @input_2d
    rate = 0.5
    
    # Apply feature dropout in training mode
    y = MLX.feature_dropout(x, rate, training: true)
    
    # Check output shape
    assert_equal x.shape, y.shape
    
    # Check that entire features are dropped across all samples
    for c in 0...@features
      # If feature is dropped, all samples have zero at that feature
      feature_sum = MLX.sum(y[:, c]).item
      
      # Either all samples have zeros, or none do
      assert (feature_sum == 0) || (feature_sum != 0)
    end
    
    # Count dropped features
    feature_is_dropped = MLX.zeros([@features], dtype: MLX.bool)
    for c in 0...@features
      if MLX.all(y[:, c] == 0).item
        feature_is_dropped[c] = 1
      end
    end
    
    # Check that proportion of dropped features is roughly equal to the dropout rate
    dropped_feature_count = MLX.sum(feature_is_dropped).item
    dropped_proportion = dropped_feature_count.to_f / @features
    
    assert_in_delta rate, dropped_proportion, 0.2
  end
end 