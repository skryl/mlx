require_relative 'mlx_test_case'

class TestNormalization < MLXTestCase
  def setup
    # Set random seed for reproducibility
    MLX.random.seed(42)
    
    # Create tensors for testing
    @batch_size = 4
    @channels = 3
    @height = 2
    @width = 2
    
    # 4D input for BatchNorm
    @input_4d = MLX.random.normal(shape: [@batch_size, @channels, @height, @width])
    
    # 3D input for LayerNorm
    @seq_len = 5
    @hidden_size = 8
    @input_3d = MLX.random.normal(shape: [@batch_size, @seq_len, @hidden_size])
    
    # 2D input for testing
    @input_2d = MLX.random.normal(shape: [@batch_size, @hidden_size])
  end
  
  def test_batch_norm_2d
    # Test BatchNorm on 2D input (batch_size, features)
    x = @input_2d
    
    # Parameters
    num_features = x.shape[1]
    gamma = MLX.ones(num_features)
    beta = MLX.zeros(num_features)
    
    # Forward pass in training mode
    y, running_mean, running_var = MLX.batch_norm(
      x, gamma, beta, 
      running_mean: nil, 
      running_var: nil, 
      training: true, 
      momentum: 0.1, 
      eps: 1e-5
    )
    
    # Check output shape
    assert_equal x.shape, y.shape
    assert_equal [num_features], running_mean.shape
    assert_equal [num_features], running_var.shape
    
    # Check that output has mean close to 0 and variance close to 1 for each feature
    mean = MLX.mean(y, axis: 0)
    var = MLX.var(y, axis: 0)
    assert MLX.allclose(mean, MLX.zeros_like(mean), atol: 1e-5)
    assert MLX.allclose(var, MLX.ones_like(var), atol: 1e-5)
    
    # Forward pass in evaluation mode
    y_eval = MLX.batch_norm(
      x, gamma, beta, 
      running_mean: running_mean, 
      running_var: running_var, 
      training: false, 
      momentum: 0.1, 
      eps: 1e-5
    )
    
    # Check output shape
    assert_equal x.shape, y_eval.shape
    
    # Test with custom gamma and beta
    gamma = MLX.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    beta = MLX.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    
    y, _, _ = MLX.batch_norm(
      x, gamma, beta, 
      running_mean: nil, 
      running_var: nil, 
      training: true, 
      momentum: 0.1, 
      eps: 1e-5
    )
    
    # Check output shape
    assert_equal x.shape, y.shape
    
    # Check that scaling and shifting works as expected
    mean_expected = beta
    # Since variance is normalized to 1, the std dev after scaling would be gamma
    # Therefore, variance should be gamma^2
    var_expected = gamma * gamma
    
    # Since we can't directly check each sample, we can verify statistics over the batch
    # are close to what we expect
    mean_actual = MLX.mean(y, axis: 0)
    var_actual = MLX.var(y, axis: 0)
    
    assert MLX.allclose(mean_actual, mean_expected, atol: 0.2)
    assert MLX.allclose(var_actual, var_expected, atol: 0.5)
  end
  
  def test_batch_norm_4d
    # Test BatchNorm on 4D input (batch_size, channels, height, width)
    x = @input_4d
    
    # Parameters
    num_features = x.shape[1]  # Number of channels
    gamma = MLX.ones(num_features)
    beta = MLX.zeros(num_features)
    
    # Forward pass in training mode
    y, running_mean, running_var = MLX.batch_norm(
      x, gamma, beta, 
      running_mean: nil, 
      running_var: nil, 
      training: true, 
      momentum: 0.1, 
      eps: 1e-5
    )
    
    # Check output shape
    assert_equal x.shape, y.shape
    assert_equal [num_features], running_mean.shape
    assert_equal [num_features], running_var.shape
    
    # Check that output has mean close to 0 and variance close to 1 for each channel
    # Need to average over batch, height, and width dimensions
    mean = MLX.mean(MLX.mean(MLX.mean(y, axis: 0), axis: 1), axis: 1)
    var = MLX.var(MLX.reshape(y, [@batch_size * @height * @width, num_features]), axis: 0)
    
    assert MLX.allclose(mean, MLX.zeros_like(mean), atol: 1e-4)
    assert MLX.allclose(var, MLX.ones_like(var), atol: 0.1)
    
    # Forward pass in evaluation mode
    y_eval = MLX.batch_norm(
      x, gamma, beta, 
      running_mean: running_mean, 
      running_var: running_var, 
      training: false, 
      momentum: 0.1, 
      eps: 1e-5
    )
    
    # Check output shape
    assert_equal x.shape, y_eval.shape
  end
  
  def test_layer_norm
    # Test LayerNorm on 2D input (batch_size, features)
    x = @input_2d
    
    # Parameters
    normalized_shape = [x.shape[1]]  # Normalize over features
    gamma = MLX.ones(normalized_shape)
    beta = MLX.zeros(normalized_shape)
    
    # Forward pass
    y = MLX.layer_norm(x, normalized_shape, gamma, beta, eps: 1e-5)
    
    # Check output shape
    assert_equal x.shape, y.shape
    
    # Check that output has mean close to 0 and variance close to 1 for each sample in batch
    mean = MLX.mean(y, axis: 1)
    var = MLX.var(y, axis: 1)
    
    assert MLX.allclose(mean, MLX.zeros_like(mean), atol: 1e-5)
    assert MLX.allclose(var, MLX.ones_like(var), atol: 1e-5)
    
    # Test with custom gamma and beta
    gamma = MLX.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    beta = MLX.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    
    y = MLX.layer_norm(x, normalized_shape, gamma, beta, eps: 1e-5)
    
    # Check output shape
    assert_equal x.shape, y.shape
  end
  
  def test_layer_norm_3d
    # Test LayerNorm on 3D input (batch_size, seq_len, features)
    x = @input_3d
    
    # Parameters
    normalized_shape = [x.shape[2]]  # Normalize over features
    gamma = MLX.ones(normalized_shape)
    beta = MLX.zeros(normalized_shape)
    
    # Forward pass
    y = MLX.layer_norm(x, normalized_shape, gamma, beta, eps: 1e-5)
    
    # Check output shape
    assert_equal x.shape, y.shape
    
    # Check that output has mean close to 0 and variance close to 1 for each position in each sample
    mean = MLX.mean(y, axis: 2)
    var = MLX.var(y, axis: 2)
    
    assert MLX.allclose(mean, MLX.zeros_like(mean), atol: 1e-5)
    assert MLX.allclose(var, MLX.ones_like(var), atol: 1e-5)
    
    # Test with larger normalized shape [seq_len, features]
    normalized_shape = [x.shape[1], x.shape[2]]
    gamma = MLX.ones(normalized_shape)
    beta = MLX.zeros(normalized_shape)
    
    y = MLX.layer_norm(x, normalized_shape, gamma, beta, eps: 1e-5)
    
    # Check output shape
    assert_equal x.shape, y.shape
  end
  
  def test_instance_norm
    # Test InstanceNorm on 4D input (batch_size, channels, height, width)
    x = @input_4d
    
    # Parameters
    num_features = x.shape[1]  # Number of channels
    gamma = MLX.ones(num_features)
    beta = MLX.zeros(num_features)
    
    # Forward pass
    y = MLX.instance_norm(x, gamma, beta, eps: 1e-5)
    
    # Check output shape
    assert_equal x.shape, y.shape
    
    # For each sample and channel, mean should be close to 0 and variance close to 1
    # We need to check each (batch, channel) pair
    for b in 0...@batch_size
      for c in 0...@channels
        # Extract the feature map for this sample and channel
        feature_map = x[b, c]
        normalized_map = y[b, c]
        
        # Check mean and variance
        mean = MLX.mean(normalized_map)
        var = MLX.var(normalized_map)
        
        assert MLX.allclose(mean, MLX.zeros_like(mean), atol: 1e-4)
        assert MLX.allclose(var, MLX.ones_like(var), atol: 1e-4)
      end
    end
    
    # Test with custom gamma and beta
    gamma = MLX.array([1.0, 2.0, 3.0])
    beta = MLX.array([0.1, 0.2, 0.3])
    
    y = MLX.instance_norm(x, gamma, beta, eps: 1e-5)
    
    # Check output shape
    assert_equal x.shape, y.shape
  end
  
  def test_group_norm
    # Test GroupNorm on 4D input (batch_size, channels, height, width)
    x = @input_4d
    
    # Parameters
    num_features = x.shape[1]  # Number of channels
    num_groups = 3  # Same as number of channels in this case
    gamma = MLX.ones(num_features)
    beta = MLX.zeros(num_features)
    
    # Forward pass
    y = MLX.group_norm(x, num_groups, gamma, beta, eps: 1e-5)
    
    # Check output shape
    assert_equal x.shape, y.shape
    
    # With num_groups = num_channels, this should be the same as instance norm
    y_instance = MLX.instance_norm(x, gamma, beta, eps: 1e-5)
    assert MLX.allclose(y, y_instance)
    
    # Test with num_groups = 1 (should be similar to layer norm across channels, height, width)
    num_groups = 1
    y = MLX.group_norm(x, num_groups, gamma, beta, eps: 1e-5)
    assert_equal x.shape, y.shape
    
    # Test with custom gamma and beta
    gamma = MLX.array([1.0, 2.0, 3.0])
    beta = MLX.array([0.1, 0.2, 0.3])
    
    y = MLX.group_norm(x, num_groups, gamma, beta, eps: 1e-5)
    
    # Check output shape
    assert_equal x.shape, y.shape
  end
  
  def test_spectral_norm
    # Create a weight matrix
    weight = MLX.random.normal(shape: [5, 3])
    
    # Forward pass
    weight_normalized, u = MLX.spectral_norm(weight)
    
    # Check output shape
    assert_equal weight.shape, weight_normalized.shape
    assert_equal [weight.shape[0]], u.shape
    
    # Spectral norm should constrain the largest singular value to 1
    u, s, v = MLX.linalg.svd(weight_normalized)
    largest_singular_value = s[0]
    
    assert_in_delta 1.0, largest_singular_value.item, 0.01
    
    # Test with custom dimension and num_iterations
    weight = MLX.random.normal(shape: [3, 7, 5])
    weight_normalized, u = MLX.spectral_norm(weight, dim: 1, num_iterations: 3)
    
    # Check output shape
    assert_equal weight.shape, weight_normalized.shape
  end
  
  def test_batch_norm_backprop
    # Test that gradients flow correctly through BatchNorm
    x = MLX.random.normal(shape: [4, 3])
    gamma = MLX.ones(3)
    beta = MLX.zeros(3)
    
    # Define a function that uses batch norm
    def batch_norm_fn(x, gamma, beta)
      y, _, _ = MLX.batch_norm(x, gamma, beta, training: true)
      MLX.sum(y)
    end
    
    # Compute gradients
    grads_fn = MLX.grad(batch_norm_fn, argnums: [0, 1, 2])
    grads = grads_fn.call(x, gamma, beta)
    
    # Check gradient shapes
    grad_x, grad_gamma, grad_beta = grads
    assert_equal x.shape, grad_x.shape
    assert_equal gamma.shape, grad_gamma.shape
    assert_equal beta.shape, grad_beta.shape
    
    # Since we're summing the output, each beta should get gradient of batch_size
    assert MLX.allclose(grad_beta, MLX.full_like(beta, x.shape[0]))
  end
  
  def test_layer_norm_backprop
    # Test that gradients flow correctly through LayerNorm
    x = MLX.random.normal(shape: [4, 3])
    normalized_shape = [3]
    gamma = MLX.ones(normalized_shape)
    beta = MLX.zeros(normalized_shape)
    
    # Define a function that uses layer norm
    def layer_norm_fn(x, gamma, beta, normalized_shape)
      y = MLX.layer_norm(x, normalized_shape, gamma, beta)
      MLX.sum(y)
    end
    
    # Compute gradients
    grads_fn = MLX.grad(layer_norm_fn, argnums: [0, 1, 2])
    grads = grads_fn.call(x, gamma, beta, normalized_shape)
    
    # Check gradient shapes
    grad_x, grad_gamma, grad_beta = grads
    assert_equal x.shape, grad_x.shape
    assert_equal gamma.shape, grad_gamma.shape
    assert_equal beta.shape, grad_beta.shape
    
    # Since we're summing the output, each beta should get gradient of batch_size
    assert MLX.allclose(grad_beta, MLX.full_like(beta, x.shape[0]))
  end
  
  def test_batch_norm_1d
    # Test BatchNorm1d (common in language models)
    x = MLX.random.normal(shape: [@batch_size, @seq_len, @hidden_size])
    
    # Parameters
    num_features = x.shape[2]  # hidden_size
    gamma = MLX.ones(num_features)
    beta = MLX.zeros(num_features)
    
    # Forward pass in training mode
    y, running_mean, running_var = MLX.batch_norm1d(
      x, gamma, beta, 
      running_mean: nil, 
      running_var: nil, 
      training: true, 
      momentum: 0.1, 
      eps: 1e-5
    )
    
    # Check output shape
    assert_equal x.shape, y.shape
    assert_equal [num_features], running_mean.shape
    assert_equal [num_features], running_var.shape
    
    # Reshape to 2D for verification
    x_2d = MLX.reshape(x, [@batch_size * @seq_len, @hidden_size])
    y_2d = MLX.reshape(y, [@batch_size * @seq_len, @hidden_size])
    
    # Check that output has mean close to 0 and variance close to 1 for each feature
    mean = MLX.mean(y_2d, axis: 0)
    var = MLX.var(y_2d, axis: 0)
    assert MLX.allclose(mean, MLX.zeros_like(mean), atol: 1e-4)
    assert MLX.allclose(var, MLX.ones_like(var), atol: 0.1)
  end
  
  def test_rmsnorm
    # Test RMSNorm (common in transformer models like LLaMA)
    x = MLX.random.normal(shape: [@batch_size, @seq_len, @hidden_size])
    
    # Parameters
    normalized_shape = [@hidden_size]  # Normalize over features
    gamma = MLX.ones(normalized_shape)
    
    # Forward pass
    y = MLX.rmsnorm(x, gamma, eps: 1e-5)
    
    # Check output shape
    assert_equal x.shape, y.shape
    
    # For RMSNorm, each position in each sample should have RMS = 1
    # RMS = sqrt(mean(x^2))
    
    # Compute RMS for each position in each sample
    x_squared = x * x
    mean_squared = MLX.mean(x_squared, axis: -1, keepdims: true)
    rms = MLX.sqrt(mean_squared)
    
    # Compute the effective RMS of the output
    y_squared = y * y
    y_mean_squared = MLX.mean(y_squared, axis: -1, keepdims: true)
    y_rms = MLX.sqrt(y_mean_squared)
    
    # Check that RMS is close to 1 for all positions
    assert MLX.allclose(y_rms, MLX.ones_like(y_rms), atol: 1e-4)
    
    # Test with custom gamma
    gamma = MLX.random.uniform(shape: normalized_shape, low: 0.5, high: 1.5)
    y = MLX.rmsnorm(x, gamma, eps: 1e-5)
    
    # Check output shape
    assert_equal x.shape, y.shape
  end
end 