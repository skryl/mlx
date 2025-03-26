require_relative 'mlx_test_case'

class TestConvTranspose < MLXTestCase
  def test_conv_transpose1d
    # Basic tests for conv_transpose1d
    
    # Case 1: Simple 1D transposed convolution
    input = MLX.ones([1, 3, 1])
    weight = MLX.ones([1, 3, 1])
    output = MLX.conv_transpose1d(input, weight)
    
    # In a simple case with ones, each output position that overlaps with kernel * input
    # will have a value equal to the overlap count
    assert_equal [1, 5, 1], output.shape
    expected = MLX.array([1.0, 2.0, 3.0, 2.0, 1.0]).reshape(1, 5, 1)
    assert MLX.allclose(output, expected)
    
    # Case 2: Multiple channels
    input = MLX.ones([1, 3, 2])
    weight = MLX.ones([2, 3, 2])
    output = MLX.conv_transpose1d(input, weight)
    
    # Each output position is the sum of the corresponding kernel-input overlaps across channels
    assert_equal [1, 5, 2], output.shape
    expected = MLX.array([[2.0, 4.0, 6.0, 4.0, 2.0], [2.0, 4.0, 6.0, 4.0, 2.0]]).reshape(1, 5, 2)
    assert MLX.allclose(output, expected)
    
    # Case 3: Strided transposed convolution
    input = MLX.ones([1, 3, 1])
    weight = MLX.ones([1, 3, 1])
    output = MLX.conv_transpose1d(input, weight, stride: 2)
    
    assert_equal [1, 7, 1], output.shape
    expected = MLX.array([1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 1.0]).reshape(1, 7, 1)
    assert MLX.allclose(output, expected)
    
    # Case 4: Padded transposed convolution
    input = MLX.ones([1, 3, 1])
    weight = MLX.ones([1, 3, 1])
    output = MLX.conv_transpose1d(input, weight, padding: 1)
    
    assert_equal [1, 3, 1], output.shape
    expected = MLX.array([2.0, 3.0, 2.0]).reshape(1, 3, 1)
    assert MLX.allclose(output, expected)
    
    # Case 5: Dilated transposed convolution
    input = MLX.ones([1, 3, 1])
    weight = MLX.ones([1, 3, 1])
    output = MLX.conv_transpose1d(input, weight, dilation: 2)
    
    assert_equal [1, 7, 1], output.shape
    expected = MLX.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).reshape(1, 7, 1)
    assert MLX.allclose(output, expected)
    
    # Case 6: Grouped transposed convolution
    input = MLX.ones([1, 3, 2])
    weight = MLX.ones([2, 3, 1])  # 2 input channels, 1 input channel per group
    output = MLX.conv_transpose1d(input, weight, groups: 2)
    
    assert_equal [1, 5, 2], output.shape
    expected = MLX.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [2.0, 0.0], [1.0, 0.0]]).reshape(1, 5, 2)
    expected += MLX.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 2.0], [0.0, 1.0]]).reshape(1, 5, 2)
    assert MLX.allclose(output, expected)
  end
  
  def test_conv_transpose1d_grad
    # Test gradient computation for conv_transpose1d using autograd
    
    # Simple case
    input = MLX.random.uniform(shape: [2, 10, 3], dtype: MLX.float32)
    weight = MLX.random.uniform(shape: [5, 3, 3], dtype: MLX.float32)
    
    # Define forward function
    f = ->(a, b) { MLX.conv_transpose1d(a, b) }
    
    # Compute gradients
    value, grads = MLX.value_and_grad(f, [0, 1]).call(input, weight)
    
    # Check shapes of gradients
    assert_equal input.shape, grads[0].shape
    assert_equal weight.shape, grads[1].shape
    
    # More complex case with stride, padding, dilation
    f_complex = ->(a, b) { MLX.conv_transpose1d(a, b, stride: 2, padding: 1, dilation: 2) }
    value, grads = MLX.value_and_grad(f_complex, [0, 1]).call(input, weight)
    
    # Check shapes of gradients
    assert_equal input.shape, grads[0].shape
    assert_equal weight.shape, grads[1].shape
  end
  
  def test_conv_transpose2d
    # Basic tests for conv_transpose2d
    
    # Case 1: Simple 2D transposed convolution
    input = MLX.ones([1, 3, 3, 1])
    weight = MLX.ones([1, 3, 3, 1])
    output = MLX.conv_transpose2d(input, weight)
    
    # Each output position will have a value equal to the overlap count
    assert_equal [1, 5, 5, 1], output.shape
    
    # Center of the output should have maximum overlap (3x3=9 positions)
    center_value = output[0, 2, 2, 0].item
    assert_equal 9.0, center_value
    
    # Corners should have minimum overlap (1 position)
    corner_value = output[0, 0, 0, 0].item
    assert_equal 1.0, corner_value
    
    # Case 2: Multiple channels
    input = MLX.ones([1, 3, 3, 2])
    weight = MLX.ones([2, 3, 3, 2])
    output = MLX.conv_transpose2d(input, weight)
    
    # Each output position is the sum of overlaps across channels
    assert_equal [1, 5, 5, 2], output.shape
    center_value = output[0, 2, 2, 0].item
    assert_equal 18.0, center_value  # 9 positions * 2 channels
    
    # Case 3: Strided transposed convolution
    input = MLX.ones([1, 2, 2, 1])
    weight = MLX.ones([1, 3, 3, 1])
    output = MLX.conv_transpose2d(input, weight, stride: [2, 2])
    
    assert_equal [1, 5, 5, 1], output.shape
    # Check specific patterns of strided transposed convolution
    # Center values should reflect overlapping regions
    center_row = output[0, 2, 0...5, 0]
    expected_center_row = MLX.array([1.0, 0.0, 1.0, 0.0, 1.0])
    assert MLX.allclose(center_row, expected_center_row)
    
    # Case 4: Padded transposed convolution
    input = MLX.ones([1, 3, 3, 1])
    weight = MLX.ones([1, 3, 3, 1])
    output = MLX.conv_transpose2d(input, weight, padding: [1, 1])
    
    assert_equal [1, 3, 3, 1], output.shape
    # Center value should reflect full overlap
    center_value = output[0, 1, 1, 0].item
    assert_equal 9.0, center_value
    
    # Case 5: Dilated transposed convolution
    input = MLX.ones([1, 2, 2, 1])
    weight = MLX.ones([1, 2, 2, 1])
    output = MLX.conv_transpose2d(input, weight, dilation: [2, 2])
    
    assert_equal [1, 5, 5, 1], output.shape
    # Check pattern of dilated convolution (should have holes)
    center_row = output[0, 2, 0...5, 0]
    expected_center_row = MLX.array([1.0, 0.0, 1.0, 0.0, 1.0])
    assert MLX.allclose(center_row, expected_center_row)
  end
  
  def test_conv_transpose2d_grad
    # Test gradient computation for conv_transpose2d using autograd
    
    # Simple case
    input = MLX.random.uniform(shape: [2, 5, 5, 3], dtype: MLX.float32)
    weight = MLX.random.uniform(shape: [4, 3, 3, 3], dtype: MLX.float32)
    
    # Define forward function
    f = ->(a, b) { MLX.conv_transpose2d(a, b) }
    
    # Compute gradients
    value, grads = MLX.value_and_grad(f, [0, 1]).call(input, weight)
    
    # Check shapes of gradients
    assert_equal input.shape, grads[0].shape
    assert_equal weight.shape, grads[1].shape
    
    # More complex case with stride, padding, dilation
    f_complex = ->(a, b) { MLX.conv_transpose2d(a, b, stride: [2, 2], padding: [1, 1], dilation: [2, 2]) }
    value, grads = MLX.value_and_grad(f_complex, [0, 1]).call(input, weight)
    
    # Check shapes of gradients
    assert_equal input.shape, grads[0].shape
    assert_equal weight.shape, grads[1].shape
  end
  
  def test_conv_transpose3d
    # Basic tests for conv_transpose3d
    
    # Case 1: Simple 3D transposed convolution
    input = MLX.ones([1, 3, 3, 3, 1])
    weight = MLX.ones([1, 3, 3, 3, 1])
    output = MLX.conv_transpose3d(input, weight)
    
    # Each output position will have a value equal to the overlap count
    assert_equal [1, 5, 5, 5, 1], output.shape
    
    # Center of the output should have maximum overlap (3x3x3=27 positions)
    center_value = output[0, 2, 2, 2, 0].item
    assert_equal 27.0, center_value
    
    # Corners should have minimum overlap (1 position)
    corner_value = output[0, 0, 0, 0, 0].item
    assert_equal 1.0, corner_value
    
    # Case 2: Multiple channels
    input = MLX.ones([1, 2, 2, 2, 2])
    weight = MLX.ones([2, 2, 2, 2, 2])
    output = MLX.conv_transpose3d(input, weight)
    
    # Each output position is the sum of overlaps across channels
    assert_equal [1, 3, 3, 3, 2], output.shape
    center_value = output[0, 1, 1, 1, 0].item
    assert_equal 16.0, center_value  # 8 positions * 2 channels
    
    # Case 3: Strided transposed convolution
    input = MLX.ones([1, 2, 2, 2, 1])
    weight = MLX.ones([1, 2, 2, 2, 1])
    output = MLX.conv_transpose3d(input, weight, stride: [2, 2, 2])
    
    assert_equal [1, 5, 5, 5, 1], output.shape
    # Check pattern of strided convolution
    assert output[0, 0, 0, 0, 0].item == 1.0
    assert output[0, 0, 0, 2, 0].item == 1.0
    assert output[0, 0, 2, 0, 0].item == 1.0
    assert output[0, 2, 0, 0, 0].item == 1.0
    
    # Case 4: Padded transposed convolution
    input = MLX.ones([1, 3, 3, 3, 1])
    weight = MLX.ones([1, 3, 3, 3, 1])
    output = MLX.conv_transpose3d(input, weight, padding: [1, 1, 1])
    
    assert_equal [1, 3, 3, 3, 1], output.shape
    # Center value should reflect full overlap
    center_value = output[0, 1, 1, 1, 0].item
    assert_equal 27.0, center_value
  end
  
  def test_conv_transpose3d_grad
    # Test gradient computation for conv_transpose3d using autograd
    
    # Small case to keep computation manageable
    input = MLX.random.uniform(shape: [1, 3, 3, 3, 2], dtype: MLX.float32)
    weight = MLX.random.uniform(shape: [3, 2, 2, 2, 2], dtype: MLX.float32)
    
    # Define forward function
    f = ->(a, b) { MLX.conv_transpose3d(a, b) }
    
    # Compute gradients
    value, grads = MLX.value_and_grad(f, [0, 1]).call(input, weight)
    
    # Check shapes of gradients
    assert_equal input.shape, grads[0].shape
    assert_equal weight.shape, grads[1].shape
  end
  
  def test_opposite_operations
    # Test that conv and conv_transpose are approximately opposite operations
    
    # 1D case
    x = MLX.random.uniform(shape: [1, 5, 2], dtype: MLX.float32)
    w = MLX.random.uniform(shape: [3, 3, 2], dtype: MLX.float32)
    
    # Apply convolution
    y = MLX.conv1d(x, w)
    
    # Apply transposed convolution
    x_reconstructed = MLX.conv_transpose1d(y, w)
    
    # The shape of x_reconstructed should be the same as x
    assert_equal x.shape, x_reconstructed.shape
    
    # 2D case
    x = MLX.random.uniform(shape: [1, 5, 5, 2], dtype: MLX.float32)
    w = MLX.random.uniform(shape: [3, 3, 3, 2], dtype: MLX.float32)
    
    # Apply convolution
    y = MLX.conv2d(x, w)
    
    # Apply transposed convolution
    x_reconstructed = MLX.conv_transpose2d(y, w)
    
    # The shape of x_reconstructed should be the same as x
    assert_equal x.shape, x_reconstructed.shape
    
    # 3D case
    x = MLX.random.uniform(shape: [1, 4, 4, 4, 2], dtype: MLX.float32)
    w = MLX.random.uniform(shape: [3, 2, 2, 2, 2], dtype: MLX.float32)
    
    # Apply convolution
    y = MLX.conv3d(x, w)
    
    # Apply transposed convolution
    x_reconstructed = MLX.conv_transpose3d(y, w)
    
    # The shape of x_reconstructed should be the same as x
    assert_equal x.shape, x_reconstructed.shape
  end
  
  def test_output_shape
    # Test output shapes for various configurations
    
    # 1D case
    input_shape = [2, 10, 3]
    weight_shape = [5, 3, 3]
    input = MLX.ones(input_shape)
    weight = MLX.ones(weight_shape)
    
    # Standard transposed convolution
    output = MLX.conv_transpose1d(input, weight)
    expected_shape = [2, 12, 5]
    assert_equal expected_shape, output.shape
    
    # Strided transposed convolution
    output = MLX.conv_transpose1d(input, weight, stride: 2)
    expected_shape = [2, 21, 5]
    assert_equal expected_shape, output.shape
    
    # Padded transposed convolution
    output = MLX.conv_transpose1d(input, weight, padding: 1)
    expected_shape = [2, 10, 5]
    assert_equal expected_shape, output.shape
    
    # Dilated transposed convolution
    output = MLX.conv_transpose1d(input, weight, dilation: 2)
    expected_shape = [2, 14, 5]
    assert_equal expected_shape, output.shape
    
    # 2D case
    input_shape = [2, 8, 8, 3]
    weight_shape = [5, 3, 3, 3]
    input = MLX.ones(input_shape)
    weight = MLX.ones(weight_shape)
    
    # Standard transposed convolution
    output = MLX.conv_transpose2d(input, weight)
    expected_shape = [2, 10, 10, 5]
    assert_equal expected_shape, output.shape
    
    # Strided transposed convolution
    output = MLX.conv_transpose2d(input, weight, stride: [2, 2])
    expected_shape = [2, 17, 17, 5]
    assert_equal expected_shape, output.shape
    
    # 3D case
    input_shape = [2, 4, 4, 4, 3]
    weight_shape = [5, 3, 3, 3, 3]
    input = MLX.ones(input_shape)
    weight = MLX.ones(weight_shape)
    
    # Standard transposed convolution
    output = MLX.conv_transpose3d(input, weight)
    expected_shape = [2, 6, 6, 6, 5]
    assert_equal expected_shape, output.shape
  end
end 