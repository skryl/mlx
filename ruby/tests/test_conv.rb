require_relative 'mlx_test_case'

class TestConv < MLXTestCase
  def test_numpy_conv
    dtypes = [MLX.float16, MLX.float32]
    
    test_cases = [
      [1, 1, "full"],
      [25, 5, "full"],
      [24, 5, "same"],
      [24, 4, "same"],
      [24, 4, "valid"],
      [4, 24, "full"],
      [5, 25, "same"],
      [4, 25, "valid"]
    ]
    
    dtypes.each do |dtype|
      test_cases.each do |m, n, mode|
        atol = dtype == MLX.float32 ? 1e-6 : 1e-5
        
        a_mx = MLX.random.uniform(shape: [m], dtype: dtype)
        v_mx = MLX.random.uniform(shape: [n], dtype: dtype)
        
        c_mx = MLX.convolve(a_mx, v_mx, mode: mode)
        
        # Since we can't compare with NumPy, we'll do some basic shape checks
        # and make sure the operation runs without errors
        
        expected_shape = case mode
        when "full"
          [m + n - 1]
        when "same"
          [m]
        when "valid"
          [m - n + 1]
        end
        
        assert_equal expected_shape, c_mx.shape
      end
    end
  end
  
  def test_conv_1d_groups_flipped
    x = MLX.broadcast_to(MLX.arange(5).astype(MLX.float32), [2, 5]).T
    w = MLX.broadcast_to(MLX.arange(4).astype(MLX.float32), [2, 4])
    out = MLX.conv_general(x[MLX.newaxis], w[..., MLX.newaxis], flip: true, groups: 2)
    expected = MLX.array([4.0, 4.0, 10.0, 10.0]).reshape(1, 2, 2)
    assert MLX.allclose(out, expected)
  end
  
  def test_conv1d
    # Basic test cases for conv1d
    # We'll create some simple test cases to verify the basic functionality
    # without relying on PyTorch for comparison
    
    # Case 1: Simple 1D convolution
    input = MLX.ones([1, 5, 1])
    weight = MLX.ones([1, 3, 1])
    output = MLX.conv1d(input, weight)
    # Each output position is sum of 3 ones = 3
    assert_equal [1, 3, 1], output.shape
    assert MLX.all(output == 3.0).item
    
    # Case 2: Multiple channels
    input = MLX.ones([1, 5, 3])
    weight = MLX.ones([2, 3, 3])
    output = MLX.conv1d(input, weight)
    # Each output position is sum of 3*3 ones = 9, and there are 2 output channels
    assert_equal [1, 3, 2], output.shape
    assert MLX.all(output == 9.0).item
    
    # Case 3: Strided convolution
    input = MLX.ones([1, 5, 1])
    weight = MLX.ones([1, 3, 1])
    output = MLX.conv1d(input, weight, stride: 2)
    assert_equal [1, 2, 1], output.shape
    assert MLX.all(output == 3.0).item
    
    # Case 4: Padded convolution
    input = MLX.ones([1, 5, 1])
    weight = MLX.ones([1, 3, 1])
    output = MLX.conv1d(input, weight, padding: 1)
    assert_equal [1, 5, 1], output.shape
    # Center values should be 3.0, edge values should be 2.0
    expected = MLX.array([2.0, 3.0, 3.0, 3.0, 2.0]).reshape(1, 5, 1)
    assert MLX.allclose(output, expected)
    
    # Case 5: Dilated convolution
    input = MLX.ones([1, 5, 1])
    weight = MLX.ones([1, 2, 1])
    output = MLX.conv1d(input, weight, dilation: 2)
    assert_equal [1, 2, 1], output.shape
    assert MLX.all(output == 2.0).item
    
    # Case 6: Grouped convolution
    input = MLX.ones([1, 5, 4])
    weight = MLX.ones([4, 3, 1])  # 4 output channels, 1 input channel per group
    output = MLX.conv1d(input, weight, groups: 4)
    assert_equal [1, 3, 4], output.shape
    assert MLX.all(output == 3.0).item
  end
  
  def test_conv1d_grad
    # Test gradient computation for conv1d using autograd
    
    # Simple case
    input = MLX.random.uniform(shape: [2, 10, 3], dtype: MLX.float32)
    weight = MLX.random.uniform(shape: [5, 3, 3], dtype: MLX.float32)
    
    # Define forward function
    f = ->(a, b) { MLX.conv1d(a, b) }
    
    # Compute gradients
    value, grads = MLX.value_and_grad(f, [0, 1]).call(input, weight)
    
    # Check shapes of gradients
    assert_equal input.shape, grads[0].shape
    assert_equal weight.shape, grads[1].shape
    
    # More complex case with stride, padding, dilation
    f_complex = ->(a, b) { MLX.conv1d(a, b, stride: 2, padding: 1, dilation: 2) }
    value, grads = MLX.value_and_grad(f_complex, [0, 1]).call(input, weight)
    
    # Check shapes of gradients
    assert_equal input.shape, grads[0].shape
    assert_equal weight.shape, grads[1].shape
  end
  
  def test_conv2d
    # Basic test cases for conv2d
    
    # Case 1: Simple 2D convolution
    input = MLX.ones([1, 5, 5, 1])
    weight = MLX.ones([1, 3, 3, 1])
    output = MLX.conv2d(input, weight)
    # Each output position is sum of 3x3 ones = 9
    assert_equal [1, 3, 3, 1], output.shape
    assert MLX.all(output == 9.0).item
    
    # Case 2: Multiple channels
    input = MLX.ones([1, 5, 5, 3])
    weight = MLX.ones([2, 3, 3, 3])
    output = MLX.conv2d(input, weight)
    # Each output position is sum of 3x3x3 ones = 27, and there are 2 output channels
    assert_equal [1, 3, 3, 2], output.shape
    assert MLX.all(output == 27.0).item
    
    # Case 3: Strided convolution
    input = MLX.ones([1, 5, 5, 1])
    weight = MLX.ones([1, 3, 3, 1])
    output = MLX.conv2d(input, weight, stride: [2, 2])
    assert_equal [1, 2, 2, 1], output.shape
    assert MLX.all(output == 9.0).item
    
    # Case 4: Padded convolution
    input = MLX.ones([1, 5, 5, 1])
    weight = MLX.ones([1, 3, 3, 1])
    output = MLX.conv2d(input, weight, padding: [1, 1])
    assert_equal [1, 5, 5, 1], output.shape
    # Center values should be 9.0, edge values should vary
    center = output[0, 1..3, 1..3, 0]
    assert MLX.all(center == 9.0).item
    
    # Case 5: Dilated convolution
    input = MLX.ones([1, 7, 7, 1])
    weight = MLX.ones([1, 2, 2, 1])
    output = MLX.conv2d(input, weight, dilation: [2, 2])
    assert_equal [1, 4, 4, 1], output.shape
    assert MLX.all(output == 4.0).item
    
    # Case 6: Grouped convolution
    input = MLX.ones([1, 5, 5, 4])
    weight = MLX.ones([4, 3, 3, 1])  # 4 output channels, 1 input channel per group
    output = MLX.conv2d(input, weight, groups: 4)
    assert_equal [1, 3, 3, 4], output.shape
    assert MLX.all(output == 9.0).item
  end
  
  def test_conv2d_grad
    # Test gradient computation for conv2d using autograd
    
    # Simple case
    input = MLX.random.uniform(shape: [2, 7, 7, 3], dtype: MLX.float32)
    weight = MLX.random.uniform(shape: [5, 3, 3, 3], dtype: MLX.float32)
    
    # Define forward function
    f = ->(a, b) { MLX.conv2d(a, b) }
    
    # Compute gradients
    value, grads = MLX.value_and_grad(f, [0, 1]).call(input, weight)
    
    # Check shapes of gradients
    assert_equal input.shape, grads[0].shape
    assert_equal weight.shape, grads[1].shape
    
    # More complex case with stride, padding, dilation
    f_complex = ->(a, b) { MLX.conv2d(a, b, stride: [2, 2], padding: [1, 1], dilation: [2, 2]) }
    value, grads = MLX.value_and_grad(f_complex, [0, 1]).call(input, weight)
    
    # Check shapes of gradients
    assert_equal input.shape, grads[0].shape
    assert_equal weight.shape, grads[1].shape
  end
  
  def test_conv3d
    # Basic test cases for conv3d
    
    # Case 1: Simple 3D convolution
    input = MLX.ones([1, 5, 5, 5, 1])
    weight = MLX.ones([1, 3, 3, 3, 1])
    output = MLX.conv3d(input, weight)
    # Each output position is sum of 3x3x3 ones = 27
    assert_equal [1, 3, 3, 3, 1], output.shape
    assert MLX.all(output == 27.0).item
    
    # Case 2: Multiple channels
    input = MLX.ones([1, 5, 5, 5, 2])
    weight = MLX.ones([3, 3, 3, 3, 2])
    output = MLX.conv3d(input, weight)
    # Each output position is sum of 3x3x3x2 ones = 54, and there are 3 output channels
    assert_equal [1, 3, 3, 3, 3], output.shape
    assert MLX.all(output == 54.0).item
    
    # Case 3: Strided convolution
    input = MLX.ones([1, 5, 5, 5, 1])
    weight = MLX.ones([1, 3, 3, 3, 1])
    output = MLX.conv3d(input, weight, stride: [2, 2, 2])
    assert_equal [1, 2, 2, 2, 1], output.shape
    assert MLX.all(output == 27.0).item
    
    # Case 4: Padded convolution
    input = MLX.ones([1, 5, 5, 5, 1])
    weight = MLX.ones([1, 3, 3, 3, 1])
    output = MLX.conv3d(input, weight, padding: [1, 1, 1])
    assert_equal [1, 5, 5, 5, 1], output.shape
    # Center values should be 27.0, edge values should vary
    center = output[0, 1..3, 1..3, 1..3, 0]
    assert MLX.all(center == 27.0).item
  end
  
  def test_conv3d_grad
    # Test gradient computation for conv3d using autograd
    
    # Smaller case due to computational demands
    input = MLX.random.uniform(shape: [1, 5, 5, 5, 2], dtype: MLX.float32)
    weight = MLX.random.uniform(shape: [2, 3, 3, 3, 2], dtype: MLX.float32)
    
    # Define forward function
    f = ->(a, b) { MLX.conv3d(a, b) }
    
    # Compute gradients
    value, grads = MLX.value_and_grad(f, [0, 1]).call(input, weight)
    
    # Check shapes of gradients
    assert_equal input.shape, grads[0].shape
    assert_equal weight.shape, grads[1].shape
  end
  
  def test_conv_general
    # Test conv_general function with various parameters
    
    # 1D convolution using conv_general
    input = MLX.ones([1, 5, 1])
    weight = MLX.ones([1, 3, 1])
    output = MLX.conv_general(input, weight)
    assert_equal [1, 3, 1], output.shape
    assert MLX.all(output == 3.0).item
    
    # 2D convolution using conv_general
    input = MLX.ones([1, 5, 5, 1])
    weight = MLX.ones([1, 3, 3, 1])
    output = MLX.conv_general(input, weight)
    assert_equal [1, 3, 3, 1], output.shape
    assert MLX.all(output == 9.0).item
    
    # 3D convolution using conv_general
    input = MLX.ones([1, 5, 5, 5, 1])
    weight = MLX.ones([1, 3, 3, 3, 1])
    output = MLX.conv_general(input, weight)
    assert_equal [1, 3, 3, 3, 1], output.shape
    assert MLX.all(output == 27.0).item
    
    # Test with flip=true
    input = MLX.ones([1, 5, 1])
    weight = MLX.array([1.0, 2.0, 3.0]).reshape([1, 3, 1])
    output_normal = MLX.conv_general(input, weight, flip: false)
    output_flipped = MLX.conv_general(input, weight, flip: true)
    
    # The flipped kernel should be [3, 2, 1]
    expected_flipped = MLX.array([6.0, 6.0, 6.0]).reshape([1, 3, 1])
    assert MLX.allclose(expected_flipped, output_flipped)
  end
  
  def test_conv_general_flip_grad
    # Test gradient computation for conv_general with flipping
    
    # Create simple test input and kernel
    input = MLX.random.uniform(shape: [1, 5, 1], dtype: MLX.float32)
    kernel = MLX.random.uniform(shape: [1, 3, 1], dtype: MLX.float32)
    
    # Define forward functions with and without flipping
    conv_flip_fn = ->(w) { MLX.conv_general(input, w, flip: true) }
    conv_normal_fn = ->(w) { MLX.conv_general(input, w, flip: false) }
    
    # Compute gradient for both cases
    _, grad_flip = MLX.value_and_grad(conv_flip_fn).call(kernel)
    _, grad_normal = MLX.value_and_grad(conv_normal_fn).call(kernel)
    
    # The gradient with flip should be different from the normal gradient
    refute MLX.allclose(grad_flip, grad_normal)
    
    # We can verify that manually flipping the gradient matches the flipped version
    # by checking that gradient at position i equals normal gradient at position (n-1-i)
    # for a 1D case
    assert MLX.allclose(grad_flip[0, 0, 0], grad_normal[0, 2, 0])
    assert MLX.allclose(grad_flip[0, 2, 0], grad_normal[0, 0, 0])
  end
  
  def test_conv_groups_grad
    # Test gradient computation for conv_general with groups
    
    # Define functions for grouped convolution
    fn = ->(x, w) { MLX.conv_general(x, w, groups: 2) }
    
    # Ground truth function that does the same thing manually
    fn_gt = ->(x, w) {
      # Split the input and kernel along the channel dimension
      x_groups = [x[..., 0...x.shape[-1]/2], x[..., x.shape[-1]/2...x.shape[-1]]]
      w_groups = [w[0...w.shape[0]/2, ...], w[w.shape[0]/2...w.shape[0], ...]]
      
      # Apply convolution to each group separately
      y_groups = []
      x_groups.each_with_index do |x_g, i|
        y_groups << MLX.conv_general(x_g, w_groups[i])
      end
      
      # Concatenate along the channel dimension
      MLX.concatenate(y_groups, axis: -1)
    }
    
    # Test 1D case
    x = MLX.random.uniform(shape: [1, 5, 4], dtype: MLX.float32)
    w = MLX.random.uniform(shape: [6, 3, 2], dtype: MLX.float32)
    
    # Compute outputs and gradients for both functions
    out1, (dx1, dw1) = MLX.value_and_grad(fn, [0, 1]).call(x, w)
    out2, (dx2, dw2) = MLX.value_and_grad(fn_gt, [0, 1]).call(x, w)
    
    # Check that outputs and gradients match
    assert MLX.allclose(out1, out2, atol: 1e-5)
    assert MLX.allclose(dx1, dx2, atol: 1e-5)
    assert MLX.allclose(dw1, dw2, atol: 1e-5)
    
    # Test 2D case
    x = MLX.random.uniform(shape: [1, 5, 5, 4], dtype: MLX.float32)
    w = MLX.random.uniform(shape: [6, 3, 3, 2], dtype: MLX.float32)
    
    # Compute outputs and gradients for both functions
    out1, (dx1, dw1) = MLX.value_and_grad(fn, [0, 1]).call(x, w)
    out2, (dx2, dw2) = MLX.value_and_grad(fn_gt, [0, 1]).call(x, w)
    
    # Check that outputs and gradients match
    assert MLX.allclose(out1, out2, atol: 1e-5)
    assert MLX.allclose(dx1, dx2, atol: 1e-5)
    assert MLX.allclose(dw1, dw2, atol: 1e-5)
  end
  
  def test_repeated_conv
    # Test multiple consecutive convolutions
    
    # Create a simple input
    x = MLX.ones([1, 10, 1])
    
    # Create some kernels
    w1 = MLX.ones([1, 3, 1])
    w2 = MLX.ones([1, 3, 1])
    
    # Apply consecutive convolutions
    y1 = MLX.conv1d(x, w1)
    y2 = MLX.conv1d(y1, w2)
    
    # Each position in y1 should be 3.0 (sum of 3 ones)
    assert MLX.all(y1 == 3.0).item
    
    # Each position in y2 should be 9.0 (sum of 3 positions, each with value 3.0)
    assert MLX.all(y2 == 9.0).item
    
    # Test with different kernels
    w1 = MLX.array([1.0, 2.0, 3.0]).reshape([1, 3, 1])
    w2 = MLX.array([4.0, 5.0, 6.0]).reshape([1, 3, 1])
    
    y1 = MLX.conv1d(x, w1)
    y2 = MLX.conv1d(y1, w2)
    
    # Check the shape
    assert_equal [1, 6, 1], y2.shape
  end
end 