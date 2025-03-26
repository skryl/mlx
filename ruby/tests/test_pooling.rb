require_relative 'mlx_test_case'

class TestPooling < MLXTestCase
  def setup
    # Create a 4D input tensor (N, C, H, W) for testing pooling operations
    @input_4d = MLX.reshape(MLX.arange(2 * 3 * 4 * 4, dtype: MLX.float32), [2, 3, 4, 4])
    
    # Create a 3D input tensor (C, H, W) for testing pooling operations
    @input_3d = MLX.reshape(MLX.arange(3 * 4 * 4, dtype: MLX.float32), [3, 4, 4])
  end

  def test_max_pool_2d
    # Test basic max pooling 2x2 with stride 2
    output = MLX.max_pool2d(@input_4d, kernel_size: 2, stride: 2)
    assert_equal [2, 3, 2, 2], output.shape
    
    # For the first channel of the first batch, we'd have:
    # [[0, 1, 2, 3],      [[5, 7],
    #  [4, 5, 6, 7],   ->  [13, 15]]
    #  [8, 9, 10, 11],
    #  [12, 13, 14, 15]]
    # where 5 is max(0,1,4,5), 7 is max(2,3,6,7), etc.
    expected_first_batch_first_channel = MLX.array([[5, 7], [13, 15]])
    assert MLX.array_equal(output[0, 0], expected_first_batch_first_channel)
    
    # Test with different kernel size and stride
    output = MLX.max_pool2d(@input_4d, kernel_size: 3, stride: 1)
    assert_equal [2, 3, 2, 2], output.shape
    
    # Test with 3D input (single batch)
    output = MLX.max_pool2d(@input_3d, kernel_size: 2, stride: 2)
    assert_equal [3, 2, 2], output.shape
    
    # Test with rectangular kernel
    output = MLX.max_pool2d(@input_4d, kernel_size: [2, 3], stride: [2, 2])
    assert_equal [2, 3, 2, 1], output.shape
    
    # Test with padding
    output = MLX.max_pool2d(@input_4d, kernel_size: 3, stride: 1, padding: 1)
    assert_equal [2, 3, 4, 4], output.shape
    
    # Test with dilation
    output = MLX.max_pool2d(@input_4d, kernel_size: 2, stride: 1, dilation: 2)
    assert_equal [2, 3, 2, 2], output.shape
  end
  
  def test_avg_pool_2d
    # Test basic average pooling 2x2 with stride 2
    output = MLX.avg_pool2d(@input_4d, kernel_size: 2, stride: 2)
    assert_equal [2, 3, 2, 2], output.shape
    
    # For the first channel of the first batch, we'd have:
    # [[0, 1, 2, 3],      [[2.5, 4.5],
    #  [4, 5, 6, 7],   ->  [10.5, 12.5]]
    #  [8, 9, 10, 11],
    #  [12, 13, 14, 15]]
    # where 2.5 is (0+1+4+5)/4, 4.5 is (2+3+6+7)/4, etc.
    expected_first_batch_first_channel = MLX.array([[2.5, 4.5], [10.5, 12.5]])
    assert MLX.allclose(output[0, 0], expected_first_batch_first_channel)
    
    # Test with different kernel size and stride
    output = MLX.avg_pool2d(@input_4d, kernel_size: 3, stride: 1)
    assert_equal [2, 3, 2, 2], output.shape
    
    # Test with 3D input (single batch)
    output = MLX.avg_pool2d(@input_3d, kernel_size: 2, stride: 2)
    assert_equal [3, 2, 2], output.shape
    
    # Test with rectangular kernel
    output = MLX.avg_pool2d(@input_4d, kernel_size: [2, 3], stride: [2, 2])
    assert_equal [2, 3, 2, 1], output.shape
    
    # Test with padding
    output = MLX.avg_pool2d(@input_4d, kernel_size: 3, stride: 1, padding: 1)
    assert_equal [2, 3, 4, 4], output.shape
    
    # Test with count_include_pad=false
    output = MLX.avg_pool2d(@input_4d, kernel_size: 2, stride: 1, padding: 1, count_include_pad: false)
    assert_equal [2, 3, 5, 5], output.shape
    
    # Test with count_include_pad=true
    output = MLX.avg_pool2d(@input_4d, kernel_size: 2, stride: 1, padding: 1, count_include_pad: true)
    assert_equal [2, 3, 5, 5], output.shape
  end
  
  def test_global_avg_pool
    # Test global average pooling on 4D input
    output = MLX.global_avg_pool(@input_4d)
    assert_equal [2, 3, 1, 1], output.shape
    
    # The global average of the first channel of the first batch is the average of all values
    # [[0, 1, 2, 3],
    #  [4, 5, 6, 7],   -> 7.5
    #  [8, 9, 10, 11],
    #  [12, 13, 14, 15]]
    expected_first_batch_first_channel = MLX.array([[7.5]])
    assert MLX.allclose(output[0, 0], expected_first_batch_first_channel)
    
    # Test global average pooling on 3D input
    output = MLX.global_avg_pool(@input_3d)
    assert_equal [3, 1, 1], output.shape
  end
  
  def test_global_max_pool
    # Test global max pooling on 4D input
    output = MLX.global_max_pool(@input_4d)
    assert_equal [2, 3, 1, 1], output.shape
    
    # The global max of the first channel of the first batch is 15
    expected_first_batch_first_channel = MLX.array([[15.0]])
    assert MLX.array_equal(output[0, 0], expected_first_batch_first_channel)
    
    # Test global max pooling on 3D input
    output = MLX.global_max_pool(@input_3d)
    assert_equal [3, 1, 1], output.shape
  end
  
  def test_adaptive_avg_pool_2d
    # Test adaptive average pooling to fixed output size
    output = MLX.adaptive_avg_pool2d(@input_4d, output_size: [2, 2])
    assert_equal [2, 3, 2, 2], output.shape
    
    # Test with single output size
    output = MLX.adaptive_avg_pool2d(@input_4d, output_size: 1)
    assert_equal [2, 3, 1, 1], output.shape
    
    # For adaptive pooling to size 1, result should be equivalent to global pooling
    global_output = MLX.global_avg_pool(@input_4d)
    assert MLX.allclose(output, global_output)
    
    # Test with 3D input (single batch)
    output = MLX.adaptive_avg_pool2d(@input_3d, output_size: [2, 2])
    assert_equal [3, 2, 2], output.shape
  end
  
  def test_adaptive_max_pool_2d
    # Test adaptive max pooling to fixed output size
    output = MLX.adaptive_max_pool2d(@input_4d, output_size: [2, 2])
    assert_equal [2, 3, 2, 2], output.shape
    
    # Test with single output size
    output = MLX.adaptive_max_pool2d(@input_4d, output_size: 1)
    assert_equal [2, 3, 1, 1], output.shape
    
    # For adaptive pooling to size 1, result should be equivalent to global pooling
    global_output = MLX.global_max_pool(@input_4d)
    assert MLX.allclose(output, global_output)
    
    # Test with 3D input (single batch)
    output = MLX.adaptive_max_pool2d(@input_3d, output_size: [2, 2])
    assert_equal [3, 2, 2], output.shape
  end
  
  def test_max_pool_1d
    # Create a 3D input tensor (N, C, L) for 1D pooling
    input_1d = MLX.reshape(MLX.arange(2 * 3 * 8, dtype: MLX.float32), [2, 3, 8])
    
    # Test basic max pooling with kernel_size=2, stride=2
    output = MLX.max_pool1d(input_1d, kernel_size: 2, stride: 2)
    assert_equal [2, 3, 4], output.shape
    
    # For the first channel of the first batch, we'd have:
    # [0, 1, 2, 3, 4, 5, 6, 7] -> [1, 3, 5, 7]
    # where 1 is max(0,1), 3 is max(2,3), etc.
    expected_first_batch_first_channel = MLX.array([1, 3, 5, 7])
    assert MLX.array_equal(output[0, 0], expected_first_batch_first_channel)
    
    # Test with different kernel size and stride
    output = MLX.max_pool1d(input_1d, kernel_size: 3, stride: 1)
    assert_equal [2, 3, 6], output.shape
    
    # Test with padding
    output = MLX.max_pool1d(input_1d, kernel_size: 2, stride: 1, padding: 1)
    assert_equal [2, 3, 9], output.shape
    
    # Test with dilation
    output = MLX.max_pool1d(input_1d, kernel_size: 2, stride: 1, dilation: 2)
    assert_equal [2, 3, 6], output.shape
    
    # Test with 2D input (single batch)
    input_2d = MLX.reshape(MLX.arange(3 * 8, dtype: MLX.float32), [3, 8])
    output = MLX.max_pool1d(input_2d, kernel_size: 2, stride: 2)
    assert_equal [3, 4], output.shape
  end
  
  def test_avg_pool_1d
    # Create a 3D input tensor (N, C, L) for 1D pooling
    input_1d = MLX.reshape(MLX.arange(2 * 3 * 8, dtype: MLX.float32), [2, 3, 8])
    
    # Test basic average pooling with kernel_size=2, stride=2
    output = MLX.avg_pool1d(input_1d, kernel_size: 2, stride: 2)
    assert_equal [2, 3, 4], output.shape
    
    # For the first channel of the first batch, we'd have:
    # [0, 1, 2, 3, 4, 5, 6, 7] -> [0.5, 2.5, 4.5, 6.5]
    # where 0.5 is (0+1)/2, 2.5 is (2+3)/2, etc.
    expected_first_batch_first_channel = MLX.array([0.5, 2.5, 4.5, 6.5])
    assert MLX.allclose(output[0, 0], expected_first_batch_first_channel)
    
    # Test with different kernel size and stride
    output = MLX.avg_pool1d(input_1d, kernel_size: 3, stride: 1)
    assert_equal [2, 3, 6], output.shape
    
    # Test with padding
    output = MLX.avg_pool1d(input_1d, kernel_size: 2, stride: 1, padding: 1)
    assert_equal [2, 3, 9], output.shape
    
    # Test with count_include_pad=false
    output = MLX.avg_pool1d(input_1d, kernel_size: 2, stride: 1, padding: 1, count_include_pad: false)
    assert_equal [2, 3, 9], output.shape
    
    # Test with count_include_pad=true
    output = MLX.avg_pool1d(input_1d, kernel_size: 2, stride: 1, padding: 1, count_include_pad: true)
    assert_equal [2, 3, 9], output.shape
    
    # Test with 2D input (single batch)
    input_2d = MLX.reshape(MLX.arange(3 * 8, dtype: MLX.float32), [3, 8])
    output = MLX.avg_pool1d(input_2d, kernel_size: 2, stride: 2)
    assert_equal [3, 4], output.shape
  end
  
  def test_pooling_backprop
    # Test that gradients flow correctly through pooling operations
    input = MLX.array([[[[1.0, 2.0], [3.0, 4.0]]]])
    
    # Test max_pool2d gradient
    def max_pool_fn(x)
      result = MLX.max_pool2d(x, kernel_size: 2, stride: 1)
      MLX.sum(result)
    end
    
    grad_fn = MLX.grad(max_pool_fn)
    grad = grad_fn.call(input)
    assert_equal input.shape, grad.shape
    
    # Only the maximum value in each pooling window should get gradient of 1.0
    expected_grad = MLX.array([[[[0.0, 0.0], [0.0, 1.0]]]])
    assert MLX.array_equal(grad, expected_grad)
    
    # Test avg_pool2d gradient
    def avg_pool_fn(x)
      result = MLX.avg_pool2d(x, kernel_size: 2, stride: 1)
      MLX.sum(result)
    end
    
    grad_fn = MLX.grad(avg_pool_fn)
    grad = grad_fn.call(input)
    assert_equal input.shape, grad.shape
    
    # Each value in the pooling window should get gradient of 0.25 (1/4)
    expected_grad = MLX.array([[[[0.25, 0.25], [0.25, 0.25]]]])
    assert MLX.allclose(grad, expected_grad)
    
    # Test global_avg_pool gradient
    def global_avg_pool_fn(x)
      result = MLX.global_avg_pool(x)
      MLX.sum(result)
    end
    
    grad_fn = MLX.grad(global_avg_pool_fn)
    grad = grad_fn.call(input)
    assert_equal input.shape, grad.shape
    
    # Each value should get gradient of 0.25 (1/4)
    expected_grad = MLX.array([[[[0.25, 0.25], [0.25, 0.25]]]])
    assert MLX.allclose(grad, expected_grad)
  end
  
  def test_pooling_with_nan
    # Test how pooling handles NaN values
    input_with_nan = MLX.array([[[[1.0, Float::NAN], [3.0, 4.0]]]])
    
    # Max pooling should ignore NaN values if possible
    output = MLX.max_pool2d(input_with_nan, kernel_size: 2, stride: 1)
    
    # Expected output depends on implementation, but ideally it would be:
    # max(1.0, NaN, 3.0, 4.0) = 4.0
    # However, some implementations might propagate NaN
    if !MLX.isnan(output[0, 0, 0, 0]).item
      assert_equal 4.0, output[0, 0, 0, 0].item
    end
    
    # Average pooling with NaN should result in NaN
    output = MLX.avg_pool2d(input_with_nan, kernel_size: 2, stride: 1)
    assert MLX.isnan(output[0, 0, 0, 0]).item
  end
end 