require_relative 'mlx_test_case'

class TestLosses < MLXTestCase
  def test_cross_entropy
    # No weights, no label smoothing
    logits = MLX.array([[0.0, -Float::INFINITY], [-Float::INFINITY, 0.0]])
    indices = MLX.array([0, 1])
    expected = MLX.array([0.0, 0.0])
    loss = MLX.nn.losses.cross_entropy(logits, indices, reduction: "none")
    assert MLX.allclose(loss, expected)
    
    probs = MLX.array([[1.0, 0.0], [0.0, 1.0]])
    loss = MLX.nn.losses.cross_entropy(logits, probs, reduction: "none")
    assert MLX.isnan(loss).all  # produce NaNs, like PyTorch
    
    # With weights, no label smoothing
    logits = MLX.array([[2.0, -1.0], [-1.0, 2.0]])
    indices = MLX.array([0, 1])
    weights = MLX.array([1.0, 2.0])
    expected = MLX.array([0.04858735, 0.0971747])
    loss = MLX.nn.losses.cross_entropy(
      logits, indices, weights: weights, reduction: "none"
    )
    assert MLX.allclose(loss, expected)
    
    probs = MLX.array([[1.0, 0.0], [0.0, 1.0]])
    loss = MLX.nn.losses.cross_entropy(logits, probs, weights: weights, reduction: "none")
    assert MLX.allclose(loss, expected)
    
    # No weights, with label smoothing
    logits = MLX.array([[2.0, -1.0], [-1.0, 2.0]])
    indices = MLX.array([0, 1])
    expected = MLX.array([0.498587, 0.498587])
    loss = MLX.nn.losses.cross_entropy(
      logits, indices, label_smoothing: 0.3, reduction: "none"
    )
    assert MLX.allclose(loss, expected)
    
    probs = MLX.array([[1.0, 0.0], [0.0, 1.0]])
    loss = MLX.nn.losses.cross_entropy(
      logits, probs, label_smoothing: 0.3, reduction: "none"
    )
    assert MLX.allclose(loss, expected)
    
    # With weights and label smoothing
    logits = MLX.array([[2.0, -1.0], [-1.0, 2.0]])
    indices = MLX.array([0, 1])
    weights = MLX.array([1.0, 2.0])
    expected = MLX.array([0.49858734, 0.9971747])
    loss = MLX.nn.losses.cross_entropy(
      logits, indices, weights: weights, label_smoothing: 0.3, reduction: "none"
    )
    assert MLX.allclose(loss, expected)
    
    probs = MLX.array([[1.0, 0.0], [0.0, 1.0]])
    loss = MLX.nn.losses.cross_entropy(
      logits, probs, weights: weights, label_smoothing: 0.3, reduction: "none"
    )
    assert MLX.allclose(loss, expected)
  end
  
  def test_binary_cross_entropy
    def test_logits_as_inputs
      logits = MLX.array([0.105361, 0.223144, 1.20397, 0.916291])
      targets = MLX.array([0, 0, 1, 1])
      
      # Test with reduction 'none'
      losses_none = MLX.nn.losses.binary_cross_entropy(
        logits, targets, reduction: "none"
      )
      expected_none = MLX.array([0.747215, 0.810930, 0.262365, 0.336472])
      assert MLX.allclose(losses_none, expected_none)
      
      # Test with reduction 'mean'
      losses_mean = MLX.nn.losses.binary_cross_entropy(
        logits, targets, reduction: "mean"
      )
      expected_mean = MLX.mean(expected_none)
      assert_equal expected_mean, losses_mean
      
      # Test with reduction 'sum'
      losses_sum = MLX.nn.losses.binary_cross_entropy(
        logits, targets, reduction: "sum"
      )
      expected_sum = MLX.sum(expected_none)
      assert_equal expected_sum, losses_sum
      
      # With weights
      weights = MLX.array([1.0, 2.0, 1.0, 2.0])
      expected = MLX.array([0.747215, 1.62186, 0.262365, 0.672944])
      loss = MLX.nn.losses.binary_cross_entropy(
        logits, targets, weights: weights, reduction: "none"
      )
      assert MLX.allclose(loss, expected)
    end
    
    def test_probs_as_inputs
      probs = MLX.array([0.5, 0.6, 0.7, 0.8])
      targets = MLX.array([0, 0, 1, 1])
      
      # Test with reduction 'none'
      losses_none = MLX.nn.losses.binary_cross_entropy(
        probs, targets, with_logits: false, reduction: "none"
      )
      expected_none = MLX.array([0.693147, 0.916291, 0.356675, 0.223144])
      assert MLX.allclose(losses_none, expected_none)
      
      # Test with reduction 'mean'
      losses_mean = MLX.nn.losses.binary_cross_entropy(
        probs, targets, with_logits: false, reduction: "mean"
      )
      expected_mean = MLX.mean(expected_none)
      assert MLX.allclose(losses_mean, expected_mean)
      
      # Test with reduction 'sum'
      losses_sum = MLX.nn.losses.binary_cross_entropy(
        probs, targets, with_logits: false, reduction: "sum"
      )
      expected_sum = MLX.sum(expected_none)
      assert MLX.allclose(losses_sum, expected_sum)
    end
    
    def test_tiny_probs_as_inputs
      TINY_PROB = 1e-59
      probs = MLX.array([0, TINY_PROB, 1 - TINY_PROB, 1])
      targets = MLX.array([0, 0, 1, 1])
      
      losses_none = MLX.nn.losses.binary_cross_entropy(
        probs, targets, with_logits: false, reduction: "none"
      )
      expected_none = MLX.array([0.0, TINY_PROB, TINY_PROB, 0.0])
      assert MLX.allclose(losses_none, expected_none)
      
      # Test with reduction 'mean'
      losses_mean = MLX.nn.losses.binary_cross_entropy(
        probs, targets, with_logits: false, reduction: "mean"
      )
      expected_mean = MLX.mean(expected_none)
      assert MLX.allclose(losses_mean, expected_mean)
      
      # Test with reduction 'sum'
      losses_sum = MLX.nn.losses.binary_cross_entropy(
        probs, targets, with_logits: false, reduction: "sum"
      )
      expected_sum = MLX.sum(expected_none)
      assert MLX.allclose(losses_sum, expected_sum)
    end
    
    test_logits_as_inputs
    test_probs_as_inputs
    test_tiny_probs_as_inputs
  end
  
  def test_l1_loss
    predictions = MLX.array([0.5, 0.2, 0.9, 0.0])
    targets = MLX.array([0.5, 0.2, 0.9, 0.0])
    
    # Expected result
    expected_none = MLX.array([0, 0, 0, 0]).astype(MLX.float32)
    expected_sum = MLX.sum(expected_none)
    expected_mean = MLX.mean(expected_none)
    
    losses = MLX.nn.losses.l1_loss(predictions, targets, reduction: "none")
    assert MLX.array_equal(losses, expected_none), "Test failed for l1_loss --reduction='none'"
    
    losses = MLX.nn.losses.l1_loss(predictions, targets, reduction: "sum")
    assert MLX.array_equal(losses, expected_sum)
    
    losses = MLX.nn.losses.l1_loss(predictions, targets, reduction: "mean")
    assert MLX.array_equal(losses, expected_mean)
  end
  
  def test_mse_loss
    predictions = MLX.array([0.5, 0.2, 0.9, 0.0])
    targets = MLX.array([0.7, 0.1, 0.8, 0.2])
    
    expected_none = MLX.array([0.04, 0.01, 0.01, 0.04])
    expected_mean = MLX.mean(expected_none)
    expected_sum = MLX.sum(expected_none)
    
    # Test with reduction 'none'
    losses_none = MLX.nn.losses.mse_loss(predictions, targets, reduction: "none")
    assert_array_close(losses_none, expected_none, 1e-5), "Test case failed for mse_loss --reduction='none'"
    
    # Test with reduction 'mean'
    losses_mean = MLX.nn.losses.mse_loss(predictions, targets, reduction: "mean")
    assert_equal expected_mean, losses_mean, "Test case failed for mse_loss --reduction='mean'"
    
    # Test with reduction 'sum'
    losses_sum = MLX.nn.losses.mse_loss(predictions, targets, reduction: "sum")
    assert_equal expected_sum, losses_sum, "Test case failed for mse_loss --reduction='sum'"
  end
  
  def test_smooth_l1_loss
    predictions = MLX.array([0.5, 0.2, 0.9, 0.0, 3.0])
    targets = MLX.array([0.7, 0.1, 0.8, 0.2, 0.0])
    
    # Beta = 1.0 (default)
    expected_none = MLX.array([0.04, 0.02, 0.02, 0.04, 2.5])
    expected_mean = MLX.mean(expected_none)
    expected_sum = MLX.sum(expected_none)
    
    # Test with reduction 'none'
    losses_none = MLX.nn.losses.smooth_l1_loss(predictions, targets, reduction: "none")
    assert_array_close(losses_none, expected_none, 1e-5), "Test case failed for smooth_l1_loss --reduction='none'"
    
    # Test with reduction 'mean'
    losses_mean = MLX.nn.losses.smooth_l1_loss(predictions, targets, reduction: "mean")
    assert_equal expected_mean, losses_mean, "Test case failed for smooth_l1_loss --reduction='mean'"
    
    # Test with reduction 'sum'
    losses_sum = MLX.nn.losses.smooth_l1_loss(predictions, targets, reduction: "sum")
    assert_equal expected_sum, losses_sum, "Test case failed for smooth_l1_loss --reduction='sum'"
    
    # Beta = 0.5
    beta = 0.5
    expected_none_beta = MLX.array([0.08, 0.04, 0.04, 0.08, 2.75])
    expected_mean_beta = MLX.mean(expected_none_beta)
    expected_sum_beta = MLX.sum(expected_none_beta)
    
    # Test with reduction 'none'
    losses_none = MLX.nn.losses.smooth_l1_loss(predictions, targets, beta: beta, reduction: "none")
    assert_array_close(losses_none, expected_none_beta, 1e-5), "Test case failed for smooth_l1_loss with beta --reduction='none'"
  end
  
  def test_nll_loss
    # Test for negative log likelihood loss
    predictions = MLX.array([[-0.5, -1.5, -1.0], [-1.0, -0.5, -1.5]])
    targets = MLX.array([0, 2])
    
    expected_none = MLX.array([0.5, 1.5])
    expected_mean = MLX.mean(expected_none)
    expected_sum = MLX.sum(expected_none)
    
    # Test with reduction 'none'
    losses_none = MLX.nn.losses.nll_loss(predictions, targets, reduction: "none")
    assert MLX.allclose(losses_none, expected_none)
    
    # Test with reduction 'mean'
    losses_mean = MLX.nn.losses.nll_loss(predictions, targets, reduction: "mean")
    assert_equal expected_mean, losses_mean
    
    # Test with reduction 'sum'
    losses_sum = MLX.nn.losses.nll_loss(predictions, targets, reduction: "sum")
    assert_equal expected_sum, losses_sum
  end
  
  def test_gaussian_nll_loss
    # Test for Gaussian negative log likelihood loss
    predictions = MLX.array([[0.5, 1.0], [0.2, 0.5], [0.4, 0.8]])
    targets = MLX.array([0.7, 0.1, 0.5])
    
    # Get means and variances from predictions
    means = predictions[:, 0]
    variances = predictions[:, 1]
    
    # Test with reduction 'none'
    losses_none = MLX.nn.losses.gaussian_nll_loss(means, targets, variances, reduction: "none")
    expected_none = MLX.array([0.7, 0.41, 0.61])
    assert_array_close(losses_none, expected_none, 1e-2)
    
    # Test with reduction 'mean'
    losses_mean = MLX.nn.losses.gaussian_nll_loss(means, targets, variances, reduction: "mean")
    expected_mean = MLX.mean(expected_none)
    assert_equal_float expected_mean, losses_mean, 1e-2
    
    # Test with reduction 'sum'
    losses_sum = MLX.nn.losses.gaussian_nll_loss(means, targets, variances, reduction: "sum")
    expected_sum = MLX.sum(expected_none)
    assert_equal_float expected_sum, losses_sum, 1e-2
    
    # Test with epsilon
    epsilon = 1e-5
    losses_with_epsilon = MLX.nn.losses.gaussian_nll_loss(
      means, targets, variances, epsilon: epsilon, reduction: "none"
    )
    assert_array_close(losses_with_epsilon, expected_none, 1e-2)
    
    # Test with full variance
    full_variances = MLX.array([[1.0, 0.5, 0.3], [0.5, 0.2, 0.1], [0.3, 0.1, 0.8]])
    losses_full = MLX.nn.losses.gaussian_nll_loss(
      means, targets, full_variances, full: true, reduction: "mean"
    )
    # We would need expected values for full case
  end
  
  def test_kl_div_loss
    # Test for KL divergence loss
    predictions = MLX.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]])
    targets = MLX.array([[0.4, 0.4, 0.2], [0.3, 0.4, 0.3]])
    
    # Test with reduction 'none' and 'batchmean'
    losses_none = MLX.nn.losses.kl_div_loss(
      predictions, targets, reduction: "none"
    )
    expected_none = MLX.array([[-0.2231, -0.4055, -0.2231], [-0.3567, -0.4055, -0.3567]])
    assert_array_close(losses_none, expected_none, 1e-3)
    
    # Test with reduction 'batchmean'
    losses_batchmean = MLX.nn.losses.kl_div_loss(
      predictions, targets, reduction: "batchmean"
    )
    expected_batchmean = MLX.sum(expected_none) / 2.0
    assert_equal_float expected_batchmean, losses_batchmean, 1e-3
  end
  
  def test_triplet_loss
    # Test for triplet loss
    anchors = MLX.array([[1.0, 2.0], [3.0, 4.0]])
    positives = MLX.array([[1.1, 2.1], [3.1, 4.1]])
    negatives = MLX.array([[2.0, 3.0], [5.0, 6.0]])
    
    # With margin=1.0 (default)
    expected_none = MLX.array([0.0, 0.0])
    
    # Test with reduction 'none'
    losses_none = MLX.nn.losses.triplet_loss(
      anchors, positives, negatives, reduction: "none"
    )
    assert_array_close(losses_none, expected_none, 1e-5)
    
    # Test with reduction 'mean'
    losses_mean = MLX.nn.losses.triplet_loss(
      anchors, positives, negatives, reduction: "mean"
    )
    expected_mean = MLX.mean(expected_none)
    assert_equal expected_mean, losses_mean
    
    # With margin=2.0
    margin = 2.0
    expected_none_margin = MLX.array([0.284, 0.284])
    
    # Test with reduction 'none'
    losses_none_margin = MLX.nn.losses.triplet_loss(
      anchors, positives, negatives, margin: margin, reduction: "none"
    )
    assert_array_close(losses_none_margin, expected_none_margin, 1e-3)
  end
  
  def test_hinge_loss
    # Test for hinge loss
    predictions = MLX.array([[0.5, -0.5], [-0.5, 0.5]])
    targets = MLX.array([[1.0, -1.0], [-1.0, 1.0]])
    
    expected_none = MLX.array([[0.5, 0.5], [0.5, 0.5]])
    
    # Test with reduction 'none'
    losses_none = MLX.nn.losses.hinge_loss(predictions, targets, reduction: "none")
    assert_array_close(losses_none, expected_none)
  end
  
  def test_huber_loss
    # Test for huber loss
    predictions = MLX.array([0.5, 0.2, 0.9, 0.0, 3.0])
    targets = MLX.array([0.7, 0.1, 0.8, 0.2, 0.0])
    
    # With delta=1.0 (default)
    expected_none = MLX.array([0.04, 0.02, 0.02, 0.04, 2.5])
    
    # Test with reduction 'none'
    losses_none = MLX.nn.losses.huber_loss(predictions, targets, reduction: "none")
    assert_array_close(losses_none, expected_none, 1e-5)
  end
  
  def test_log_cosh_loss
    # Test for log cosh loss
    predictions = MLX.array([0.5, 0.2, 0.9, 0.0])
    targets = MLX.array([0.7, 0.1, 0.8, 0.2])
    
    expected_none = MLX.array([0.0324, 0.0050, 0.0050, 0.0324])
    
    # Test with reduction 'none'
    losses_none = MLX.nn.losses.log_cosh_loss(predictions, targets, reduction: "none")
    assert_array_close(losses_none, expected_none, 1e-4)
  end
  
  def test_cosine_similarity_loss
    # Test for cosine similarity loss
    predictions = MLX.array([[1.0, 0.0], [0.0, 1.0]])
    targets = MLX.array([[0.0, 1.0], [0.0, 1.0]])
    
    # Test with reduction 'none'
    losses_none = MLX.nn.losses.cosine_similarity_loss(
      predictions, targets, reduction: "none"
    )
    expected_none = MLX.array([1.0, 0.0])
    assert_array_close(losses_none, expected_none)
    
    # Test with reduction 'mean'
    losses_mean = MLX.nn.losses.cosine_similarity_loss(
      predictions, targets, reduction: "mean"
    )
    expected_mean = MLX.mean(expected_none)
    assert_equal expected_mean, losses_mean
    
    # Test with reduction 'sum'
    losses_sum = MLX.nn.losses.cosine_similarity_loss(
      predictions, targets, reduction: "sum"
    )
    expected_sum = MLX.sum(expected_none)
    assert_equal expected_sum, losses_sum
    
    # Test with dim=0
    losses_dim = MLX.nn.losses.cosine_similarity_loss(
      predictions, targets, dim: 0, reduction: "none"
    )
    expected_dim = MLX.array([0.0, 0.0])
    assert_array_close(losses_dim, expected_dim)
  end
  
  def test_margin_ranking_loss
    # Test for margin ranking loss
    input1 = MLX.array([[1.0, 2.0], [3.0, 4.0]])
    input2 = MLX.array([[2.0, 1.0], [4.0, 3.0]])
    targets = MLX.array([[1.0, -1.0], [-1.0, 1.0]])
    
    # With margin=0.0
    expected_none = MLX.array([[1.0, 0.0], [0.0, 1.0]])
    
    # Test with reduction 'none'
    losses_none = MLX.nn.losses.margin_ranking_loss(
      input1, input2, targets, margin: 0.0, reduction: "none"
    )
    assert_array_close(losses_none, expected_none)
    
    # Test with reduction 'mean'
    losses_mean = MLX.nn.losses.margin_ranking_loss(
      input1, input2, targets, margin: 0.0, reduction: "mean"
    )
    expected_mean = MLX.mean(expected_none)
    assert_equal expected_mean, losses_mean
    
    # Test with reduction 'sum'
    losses_sum = MLX.nn.losses.margin_ranking_loss(
      input1, input2, targets, margin: 0.0, reduction: "sum"
    )
    expected_sum = MLX.sum(expected_none)
    assert_equal expected_sum, losses_sum
    
    # With margin=1.0 (default)
    expected_none_margin = MLX.array([[2.0, 0.0], [0.0, 2.0]])
    
    # Test with reduction 'none'
    losses_none_margin = MLX.nn.losses.margin_ranking_loss(
      input1, input2, targets, reduction: "none"
    )
    assert_array_close(losses_none_margin, expected_none_margin)
  end
  
  # Helper method for comparing arrays with a tolerance
  def assert_array_close(actual, expected, tolerance = 1e-5)
    assert MLX.all(MLX.abs(actual - expected) < tolerance).item
  end
  
  # Helper method for comparing floating point values with a tolerance
  def assert_equal_float(expected, actual, tolerance = 1e-5)
    assert (expected - actual).abs < tolerance, "Expected #{expected} but got #{actual}, difference exceeds #{tolerance}"
  end
end 