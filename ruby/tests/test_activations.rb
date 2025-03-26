require_relative 'mlx_test_case'

class TestActivations < MLXTestCase
  def test_relu
    # Test ReLU on 1D array
    x = MLX.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = MLX.relu(x)
    expected = MLX.array([0.0, 0.0, 0.0, 1.0, 2.0])
    assert MLX.array_equal(y, expected)
    
    # Test ReLU on 2D array
    x = MLX.array([[-1.0, 1.0], [-2.0, 2.0]])
    y = MLX.relu(x)
    expected = MLX.array([[0.0, 1.0], [0.0, 2.0]])
    assert MLX.array_equal(y, expected)
    
    # Test ReLU gradient
    x = MLX.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    grad_fn = MLX.grad(->(x) { MLX.relu(x).sum })
    grad = grad_fn.call(x)
    expected_grad = MLX.array([0.0, 0.0, 0.0, 1.0, 1.0])
    assert MLX.array_equal(grad, expected_grad)
  end
  
  def test_sigmoid
    # Test sigmoid on 1D array
    x = MLX.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = MLX.sigmoid(x)
    
    # Check shape and type
    assert_equal x.shape, y.shape
    assert_equal x.dtype, y.dtype
    
    # Check sigmoid values (approximate)
    expected = MLX.array([0.119, 0.269, 0.5, 0.731, 0.881])
    assert MLX.allclose(y, expected, atol: 0.001)
    
    # Test sigmoid bounds (0 to 1)
    x_extreme = MLX.array([-100.0, 100.0])
    y_extreme = MLX.sigmoid(x_extreme)
    assert MLX.all(y_extreme >= 0).item
    assert MLX.all(y_extreme <= 1).item
    
    # Test sigmoid gradient
    x = MLX.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    grad_fn = MLX.grad(->(x) { MLX.sigmoid(x).sum })
    grad = grad_fn.call(x)
    
    # Sigmoid gradient is sigmoid(x) * (1 - sigmoid(x))
    sigmoid_x = MLX.sigmoid(x)
    expected_grad = sigmoid_x * (1 - sigmoid_x)
    assert MLX.allclose(grad, expected_grad)
  end
  
  def test_tanh
    # Test tanh on 1D array
    x = MLX.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = MLX.tanh(x)
    
    # Check shape and type
    assert_equal x.shape, y.shape
    assert_equal x.dtype, y.dtype
    
    # Check tanh values (approximate)
    expected = MLX.array([-0.964, -0.762, 0.0, 0.762, 0.964])
    assert MLX.allclose(y, expected, atol: 0.001)
    
    # Test tanh bounds (-1 to 1)
    x_extreme = MLX.array([-100.0, 100.0])
    y_extreme = MLX.tanh(x_extreme)
    assert MLX.all(y_extreme >= -1).item
    assert MLX.all(y_extreme <= 1).item
    
    # Test tanh gradient
    x = MLX.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    grad_fn = MLX.grad(->(x) { MLX.tanh(x).sum })
    grad = grad_fn.call(x)
    
    # Tanh gradient is (1 - tanh(x)^2)
    expected_grad = 1 - MLX.tanh(x)**2
    assert MLX.allclose(grad, expected_grad)
  end
  
  def test_leaky_relu
    # Test leaky_relu on 1D array with default negative_slope
    x = MLX.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = MLX.leaky_relu(x)
    expected = MLX.array([-0.02, -0.01, 0.0, 1.0, 2.0])  # default negative_slope is 0.01
    assert MLX.allclose(y, expected)
    
    # Test with custom negative_slope
    negative_slope = 0.1
    y = MLX.leaky_relu(x, negative_slope)
    expected = MLX.array([-0.2, -0.1, 0.0, 1.0, 2.0])
    assert MLX.allclose(y, expected)
    
    # Test gradient with default negative_slope
    grad_fn = MLX.grad(->(x) { MLX.leaky_relu(x).sum })
    grad = grad_fn.call(x)
    expected_grad = MLX.array([0.01, 0.01, 0.01, 1.0, 1.0])
    assert MLX.allclose(grad, expected_grad, atol: 1e-5)
    
    # Test gradient with custom negative_slope
    grad_fn = MLX.grad(->(x) { MLX.leaky_relu(x, negative_slope).sum })
    grad = grad_fn.call(x)
    expected_grad = MLX.array([0.1, 0.1, 0.1, 1.0, 1.0])
    assert MLX.allclose(grad, expected_grad, atol: 1e-5)
  end
  
  def test_gelu
    # Test GELU on 1D array
    x = MLX.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = MLX.gelu(x)
    
    # Check shape and type
    assert_equal x.shape, y.shape
    assert_equal x.dtype, y.dtype
    
    # Approximate expected values
    expected = MLX.array([-0.046, -0.159, 0.0, 0.841, 1.954])
    assert MLX.allclose(y, expected, atol: 0.001)
    
    # Test GELU gradient
    grad_fn = MLX.grad(->(x) { MLX.gelu(x).sum })
    grad = grad_fn.call(x)
    
    # Make sure gradient has correct shape and is not zero
    assert_equal x.shape, grad.shape
    refute MLX.all(grad == 0).item
  end
  
  def test_silu
    # Test SiLU/Swish on 1D array
    x = MLX.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = MLX.silu(x)
    
    # Check shape and type
    assert_equal x.shape, y.shape
    assert_equal x.dtype, y.dtype
    
    # SiLU is x * sigmoid(x)
    expected = x * MLX.sigmoid(x)
    assert MLX.allclose(y, expected)
    
    # Approximate expected values
    expected = MLX.array([-0.239, -0.269, 0.0, 0.731, 1.762])
    assert MLX.allclose(y, expected, atol: 0.001)
    
    # Test SiLU gradient
    grad_fn = MLX.grad(->(x) { MLX.silu(x).sum })
    grad = grad_fn.call(x)
    
    # Make sure gradient has correct shape and is not zero
    assert_equal x.shape, grad.shape
    refute MLX.all(grad == 0).item
  end
  
  def test_softplus
    # Test softplus on 1D array
    x = MLX.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = MLX.softplus(x)
    
    # Check shape and type
    assert_equal x.shape, y.shape
    assert_equal x.dtype, y.dtype
    
    # Softplus is ln(1 + exp(x))
    expected = MLX.log(1 + MLX.exp(x))
    assert MLX.allclose(y, expected)
    
    # Approximate expected values
    expected = MLX.array([0.127, 0.313, 0.693, 1.313, 2.127])
    assert MLX.allclose(y, expected, atol: 0.001)
    
    # Test softplus gradient
    grad_fn = MLX.grad(->(x) { MLX.softplus(x).sum })
    grad = grad_fn.call(x)
    
    # Softplus gradient is sigmoid(x)
    expected_grad = MLX.sigmoid(x)
    assert MLX.allclose(grad, expected_grad)
  end
  
  def test_softsign
    # Test softsign on 1D array
    x = MLX.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = MLX.softsign(x)
    
    # Check shape and type
    assert_equal x.shape, y.shape
    assert_equal x.dtype, y.dtype
    
    # Softsign is x / (1 + |x|)
    expected = x / (1 + MLX.abs(x))
    assert MLX.allclose(y, expected)
    
    # Expected values
    expected = MLX.array([-0.667, -0.5, 0.0, 0.5, 0.667])
    assert MLX.allclose(y, expected, atol: 0.001)
    
    # Test softsign gradient
    grad_fn = MLX.grad(->(x) { MLX.softsign(x).sum })
    grad = grad_fn.call(x)
    
    # Softsign gradient is 1 / (1 + |x|)^2
    expected_grad = 1 / (1 + MLX.abs(x))**2
    assert MLX.allclose(grad, expected_grad)
  end
  
  def test_elu
    # Test ELU on 1D array with default alpha
    x = MLX.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = MLX.elu(x)
    
    # Check shape and type
    assert_equal x.shape, y.shape
    assert_equal x.dtype, y.dtype
    
    # Default alpha is 1.0
    alpha = 1.0
    expected_neg = alpha * (MLX.exp(x[x < 0]) - 1)
    expected_pos = x[x >= 0]
    
    # Approximate expected values
    expected = MLX.array([-0.865, -0.632, 0.0, 1.0, 2.0])
    assert MLX.allclose(y, expected, atol: 0.001)
    
    # Test with custom alpha
    alpha = 0.5
    y = MLX.elu(x, alpha)
    expected = MLX.array([-0.432, -0.316, 0.0, 1.0, 2.0])
    assert MLX.allclose(y, expected, atol: 0.001)
    
    # Test gradient
    grad_fn = MLX.grad(->(x) { MLX.elu(x).sum })
    grad = grad_fn.call(x)
    
    # Make sure gradient has correct shape
    assert_equal x.shape, grad.shape
  end
  
  def test_mish
    # Test Mish activation on 1D array
    x = MLX.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = MLX.mish(x)
    
    # Check shape and type
    assert_equal x.shape, y.shape
    assert_equal x.dtype, y.dtype
    
    # Mish is x * tanh(softplus(x))
    expected = x * MLX.tanh(MLX.softplus(x))
    assert MLX.allclose(y, expected)
    
    # Approximate expected values
    expected = MLX.array([-0.25, -0.30, 0.0, 0.86, 1.94])
    assert MLX.allclose(y, expected, atol: 0.01)
    
    # Test gradient
    grad_fn = MLX.grad(->(x) { MLX.mish(x).sum })
    grad = grad_fn.call(x)
    
    # Make sure gradient has correct shape
    assert_equal x.shape, grad.shape
  end
  
  def test_hardtanh
    # Test hardtanh on 1D array with default min/max
    x = MLX.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = MLX.hardtanh(x)
    
    # Check shape and type
    assert_equal x.shape, y.shape
    assert_equal x.dtype, y.dtype
    
    # Default min_val=-1, max_val=1
    expected = MLX.array([-1.0, -1.0, 0.0, 1.0, 1.0])
    assert MLX.array_equal(y, expected)
    
    # Test with custom min/max
    y = MLX.hardtanh(x, min_val: -0.5, max_val: 0.5)
    expected = MLX.array([-0.5, -0.5, 0.0, 0.5, 0.5])
    assert MLX.array_equal(y, expected)
    
    # Test gradient with default min/max
    grad_fn = MLX.grad(->(x) { MLX.hardtanh(x).sum })
    grad = grad_fn.call(x)
    
    # Gradient should be 1 in the range (-1, 1), 0 outside
    expected_grad = MLX.array([0.0, 0.0, 1.0, 1.0, 0.0])
    assert MLX.allclose(grad, expected_grad)
  end
  
  def test_logsigmoid
    # Test logsigmoid on 1D array
    x = MLX.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = MLX.logsigmoid(x)
    
    # Check shape and type
    assert_equal x.shape, y.shape
    assert_equal x.dtype, y.dtype
    
    # logsigmoid is log(sigmoid(x))
    expected = MLX.log(MLX.sigmoid(x))
    assert MLX.allclose(y, expected)
    
    # Approximate expected values
    expected = MLX.array([-2.13, -1.31, -0.693, -0.313, -0.127])
    assert MLX.allclose(y, expected, atol: 0.01)
    
    # Test gradient
    grad_fn = MLX.grad(->(x) { MLX.logsigmoid(x).sum })
    grad = grad_fn.call(x)
    
    # Gradient of logsigmoid is sigmoid(-x)
    expected_grad = MLX.sigmoid(-x)
    assert MLX.allclose(grad, expected_grad)
  end
  
  def test_celu
    # Test CELU activation on 1D array with default alpha
    x = MLX.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = MLX.celu(x)
    
    # Check shape and type
    assert_equal x.shape, y.shape
    assert_equal x.dtype, y.dtype
    
    # Default alpha is 1.0
    alpha = 1.0
    
    # CELU: max(0,x) + min(0,alpha * (exp(x/alpha) - 1))
    pos_part = MLX.maximum(0, x)
    neg_part = MLX.minimum(0, alpha * (MLX.exp(x / alpha) - 1))
    expected = pos_part + neg_part
    
    assert MLX.allclose(y, expected)
    
    # Approximate expected values
    expected = MLX.array([-0.865, -0.632, 0.0, 1.0, 2.0])
    assert MLX.allclose(y, expected, atol: 0.001)
    
    # Test with custom alpha
    alpha = 0.5
    y = MLX.celu(x, alpha)
    
    # Recompute with new alpha
    pos_part = MLX.maximum(0, x)
    neg_part = MLX.minimum(0, alpha * (MLX.exp(x / alpha) - 1))
    expected = pos_part + neg_part
    
    assert MLX.allclose(y, expected)
  end
  
  def test_softmax
    # Test softmax on 1D array
    x = MLX.array([1.0, 2.0, 3.0, 4.0])
    y = MLX.softmax(x)
    
    # Check shape and type
    assert_equal x.shape, y.shape
    assert_equal x.dtype, y.dtype
    
    # Softmax should sum to 1
    assert_in_delta 1.0, MLX.sum(y).item, 1e-6
    
    # Values should be in (0,1) range
    assert MLX.all(y > 0).item
    assert MLX.all(y < 1).item
    
    # Verify softmax formula
    exp_x = MLX.exp(x)
    expected = exp_x / MLX.sum(exp_x)
    assert MLX.allclose(y, expected)
    
    # Test on 2D array
    x = MLX.array([[1.0, 2.0], [3.0, 4.0]])
    y = MLX.softmax(x)
    
    # Each row should sum to 1
    row_sums = MLX.sum(y, axis: 1)
    assert MLX.allclose(row_sums, MLX.ones_like(row_sums))
    
    # Test with axis parameter
    y = MLX.softmax(x, axis: 0)
    
    # Each column should sum to 1
    col_sums = MLX.sum(y, axis: 0)
    assert MLX.allclose(col_sums, MLX.ones_like(col_sums))
  end
  
  def test_log_softmax
    # Test log_softmax on 1D array
    x = MLX.array([1.0, 2.0, 3.0, 4.0])
    y = MLX.log_softmax(x)
    
    # Check shape and type
    assert_equal x.shape, y.shape
    assert_equal x.dtype, y.dtype
    
    # log_softmax should be equal to log(softmax(x))
    expected = MLX.log(MLX.softmax(x))
    assert MLX.allclose(y, expected)
    
    # Test on 2D array
    x = MLX.array([[1.0, 2.0], [3.0, 4.0]])
    y = MLX.log_softmax(x)
    
    # log_softmax should be equal to log(softmax(x))
    expected = MLX.log(MLX.softmax(x))
    assert MLX.allclose(y, expected)
    
    # Test with axis parameter
    y = MLX.log_softmax(x, axis: 0)
    expected = MLX.log(MLX.softmax(x, axis: 0))
    assert MLX.allclose(y, expected)
  end
end 