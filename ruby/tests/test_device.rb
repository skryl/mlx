require_relative 'mlx_test_case'
require 'test/unit'

# Don't inherit from MLXTestCase to avoid call to setup
class TestDefaultDevice < Test::Unit::TestCase
  def test_mlx_default_device
    device = MLX.default_device
    
    if MLX.metal.available?
      assert_equal :gpu, device
      assert_equal :gpu, MLX.default_device
    else
      assert_equal :cpu, device
      assert_raises(ArgumentError) do
        MLX.set_default_device(:gpu)
      end
    end
  end
end

class TestDevice < MLXTestCase
  def test_device
    # Store original device
    original_device = MLX.default_device
    
    # Set device to CPU
    MLX.set_default_device(:cpu)
    assert_equal :cpu, MLX.default_device
    
    # Test equality
    assert_equal :cpu, MLX.default_device
    
    # Restore device
    MLX.set_default_device(original_device)
  end
  
  def test_device_context
    # Skip if Metal is not available
    skip "Metal is not available" unless MLX.metal.available?
    
    default = MLX.default_device
    diff = default == :gpu ? :cpu : :gpu
    
    assert_not_equal default, diff
    
    MLX.stream(diff) do
      a = MLX.add(MLX.zeros([2, 2]), MLX.ones([2, 2]))
      MLX.eval(a)
      assert_equal diff, MLX.default_device
    end
    
    assert_equal default, MLX.default_device
  end
  
  def test_op_on_device
    x = MLX.array(1.0)
    y = MLX.array(1.0)
    
    # Test with default device
    a = MLX.add(x, y, stream: nil)
    b = MLX.add(x, y, stream: MLX.default_device)
    assert_equal a.item, b.item
    
    # Test with explicit CPU
    b = MLX.add(x, y, stream: :cpu)
    assert_equal a.item, b.item
    
    # Test with explicit GPU if available
    if MLX.metal.available?
      b = MLX.add(x, y, stream: :gpu)
      assert_equal a.item, b.item
    end
  end
end

class TestStream < MLXTestCase
  def test_stream
    # Test default stream on default device
    s1 = MLX.default_stream(MLX.default_device)
    assert_equal MLX.default_device, s1.device
    
    # Test new stream on default device
    s2 = MLX.new_stream(MLX.default_device)
    assert_equal MLX.default_device, s2.device
    assert_not_equal s1, s2
    
    # Test GPU stream if available
    if MLX.metal.available?
      s_gpu = MLX.default_stream(:gpu)
      assert_equal :gpu, s_gpu.device
    else
      assert_raises(ArgumentError) do
        MLX.default_stream(:gpu)
      end
    end
    
    # Test CPU stream
    s_cpu = MLX.default_stream(:cpu)
    assert_equal :cpu, s_cpu.device
    
    # Test new CPU stream
    s_cpu = MLX.new_stream(:cpu)
    assert_equal :cpu, s_cpu.device
    
    # Test new GPU stream if available
    if MLX.metal.available?
      s_gpu = MLX.new_stream(:gpu)
      assert_equal :gpu, s_gpu.device
    else
      assert_raises(ArgumentError) do
        MLX.new_stream(:gpu)
      end
    end
  end
  
  def test_op_on_stream
    x = MLX.array(1.0)
    y = MLX.array(1.0)
    
    # Test with default stream
    a = MLX.add(x, y, stream: MLX.default_stream(MLX.default_device))
    
    # Test with GPU stream if available
    if MLX.metal.available?
      b = MLX.add(x, y, stream: MLX.default_stream(:gpu))
      assert_equal a.item, b.item
      
      s_gpu = MLX.new_stream(:gpu)
      b = MLX.add(x, y, stream: s_gpu)
      assert_equal a.item, b.item
    end
    
    # Test with CPU stream
    b = MLX.add(x, y, stream: MLX.default_stream(:cpu))
    assert_equal a.item, b.item
    
    s_cpu = MLX.new_stream(:cpu)
    b = MLX.add(x, y, stream: s_cpu)
    assert_equal a.item, b.item
  end
  
  def test_stream_context
    # Test stream context manager
    original_device = MLX.default_device
    
    # Create a stream on a different device
    diff_device = original_device == :gpu ? :cpu : :gpu
    
    # Skip if different device is GPU but not available
    if diff_device == :gpu && !MLX.metal.available?
      skip "GPU not available for stream context test"
    end
    
    stream = MLX.new_stream(diff_device)
    
    # Before context
    assert_equal original_device, MLX.default_device
    
    MLX.stream(stream) do
      # Inside context
      assert_equal diff_device, MLX.default_device
      
      # Create and evaluate tensor on this stream
      a = MLX.ones([2, 2])
      b = MLX.ones([2, 2])
      c = a + b
      MLX.eval(c)
      
      assert_equal [2, 2], c.shape
      assert_equal [[2.0, 2.0], [2.0, 2.0]], c.to_a
    end
    
    # After context - should be back to original
    assert_equal original_device, MLX.default_device
  end
  
  def test_device_placement
    # Test explicit device placement for arrays
    x = MLX.array([1, 2, 3], device: :cpu)
    assert_equal :cpu, x.device
    
    # Skip if Metal is not available
    if MLX.metal.available?
      y = MLX.array([1, 2, 3], device: :gpu)
      assert_equal :gpu, y.device
      
      # Test moving between devices
      z = x.to(:gpu)
      assert_equal :gpu, z.device
      
      w = y.to(:cpu)
      assert_equal :cpu, w.device
    end
  end
end 