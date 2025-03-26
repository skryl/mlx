require_relative 'mlx_test_case'

class TestEval < MLXTestCase
  def setup
    # Set random seed for reproducibility
    MLX.random.seed(42)
  end
  
  def test_eval
    # Test basic evaluation
    arrs = Array.new(4) { MLX.ones([2, 2]) }
    MLX.eval(*arrs)
    
    arrs.each do |x|
      assert_equal [[1, 1], [1, 1]], x.to_a
    end
  end
  
  def test_retain_graph
    # Test that eval doesn't prune the computation graph for gradients
    fun = lambda do |x|
      y = 3 * x
      MLX.eval(y)
      2 * y
    end
    
    dfun_dx = MLX.grad(fun)
    y = dfun_dx.call(MLX.array(1.0))
    assert_equal 6.0, y.item
  end
  
  def test_eval_mixed
    # Test eval with mixed content (arrays and non-arrays)
    x = MLX.array(1) + 1 + 1
    y = 0
    z = "hello"
    state = [x, y, z]
    MLX.eval(state)
    
    assert_equal 3, x.item
  end
  
  def test_async_eval
    # Test asynchronous evaluation
    x = MLX.array(1) + MLX.array(1) + MLX.array(1)
    MLX.async_eval(x)
    assert_equal 3, x.item
    
    # It should be safe to call eval on the array which has been async eval'ed
    x = MLX.array(1) + MLX.array(1) + MLX.array(1)
    assert_equal 3, x.item
    
    x = MLX.array([1, 2, 3])
    y = 2 * x
    MLX.async_eval(y)
    
    z = 2 * y
    MLX.async_eval(z)
    
    assert MLX.array_equal(y, MLX.array([2, 4, 6]))
    assert MLX.array_equal(z, MLX.array([4, 8, 12]))
  end
  
  def test_async_eval_twice
    # Test multiple async_eval calls
    100.times do
      x = MLX.array(1) + MLX.array(1) + MLX.array(1)
      MLX.async_eval(x)
      
      y = x + 1
      MLX.async_eval(y)
      
      assert_equal 3, x.item
      assert_equal 4, y.item
    end
  end
  
  def test_async_eval_in_trace
    # Test async_eval inside transformation functions (should raise error)
    fun = lambda do |x|
      y = x + 1.0
      MLX.async_eval(y)
      MLX.exp(y)
    end
    
    # Raises error in grad
    assert_raises(ArgumentError) do
      MLX.grad(fun).call(MLX.array(1.0))
    end
    
    # Also raises error in vmap
    assert_raises(ArgumentError) do
      MLX.vmap(fun).call(MLX.ones([2, 2]))
    end
  end
  
  def test_async_eval_into_eval
    # Test async_eval followed by regular evaluation
    x = MLX.array(1)
    y = x + 1
    MLX.async_eval(y)
    
    a = y - 10
    b = MLX.abs(a)
    
    assert_equal 8, b.item
  end
  
  def test_async_eval_into_eval_diff_stream
    # Test async_eval with different streams
    s = MLX.new_stream(:cpu)
    
    x = MLX.array(0)
    y = x - 5
    MLX.async_eval(y)
    
    z = MLX.abs(y, stream: s)
    assert_equal 5, z.item
  end
  
  def test_eval_slow_fast_multi_stream
    # Test evaluation with operations of different speeds on multiple streams
    x = MLX.ones([8000])
    y = MLX.abs(MLX.array(-1.0))
    
    20.times do
      x = x + MLX.array(1.0)
    end
    
    z = MLX.add(x, y, stream: :cpu)
    assert MLX.allclose(z, MLX.full([8000], 22.0))
    
    # Switch eval order
    x = MLX.ones([8000])
    y = MLX.abs(MLX.array(-1.0))
    
    20.times do
      x = x + MLX.array(1.0)
    end
    
    z = MLX.add(y, x, stream: :cpu)
    assert MLX.allclose(z, MLX.full([8000], 22.0))
  end
  
  def test_multi_output_eval_during_transform
    # Test eval with multiple outputs during transformation
    x = MLX.random.uniform(shape: [1024])
    y = MLX.ones([1024])
    MLX.eval(x, y)
    
    fn = lambda do |x|
      a, b = MLX.divmod(x, x)
      MLX.eval(a)
      a
    end
    
    out = MLX.vjp(fn, [x], [y])
    out = MLX.vjp(fn, [x], [y])
    
    peak_mem = MLX.get_peak_memory
    out = MLX.vjp(fn, [x], [y])
    
    assert_equal peak_mem, MLX.get_peak_memory
  end
  
  def test_async_eval_with_multiple_streams
    # Test async_eval with multiple streams
    x = MLX.array([1.0])
    y = MLX.array([1.0])
    a = MLX.array([1.0])
    b = MLX.array([1.0])
    
    d = MLX.default_device
    s2 = MLX.new_stream(d)
    
    10.times do
      20.times do
        x = x + y
      end
      
      MLX.async_eval(x)
      MLX.eval(a + b)
    end
  end
  
  def test_donation_for_noops
    # Test memory donation with no-op operations
    fun1 = lambda do |x|
      s = x.shape
      10.times do
        x = MLX.abs(x)
        x = MLX.reshape(x, [-1])
        x = x.transpose.transpose
        x = MLX.stop_gradient(x)
        x = MLX.abs(x)
      end
      x
    end
    
    x = MLX.zeros([1024, 1024])
    MLX.eval(x)
    
    pre = MLX.get_peak_memory
    out = fun1.call(x)
    x = nil
    MLX.eval(out)
    post = MLX.get_peak_memory
    
    assert_equal pre, post
    
    fun2 = lambda do |x|
      10.times do
        x = MLX.abs(x)
        x = x[0..-2]  # Slice off last element
        x = MLX.abs(x)
      end
      x
    end
    
    x = MLX.zeros([1024 * 1024])
    MLX.eval(x)
    
    pre = MLX.get_peak_memory
    out = fun2.call(x)
    x = nil
    MLX.eval(out)
    post = MLX.get_peak_memory
    
    assert_equal pre, post
  end
  
  def test_multistream_deadlock
    # Test for deadlocks with multiple streams
    # Skip if Metal is not available
    skip "Metal is not available" unless MLX.metal.available?
    
    s1 = MLX.default_stream(:gpu)
    s2 = MLX.new_stream(:gpu)
    
    x = MLX.array(1.0)
    x = MLX.abs(x, stream: s1)
    
    100.times do
      x = MLX.abs(x, stream: s2)
    end
    
    MLX.eval(x)
    
    s1 = MLX.default_stream(:gpu)
    s2 = MLX.new_stream(:gpu)
    
    # Set temporary memory limit
    old_limit = MLX.set_memory_limit(1000)
    
    x = MLX.ones([128, 128], stream: s2)
    
    20.times do
      x = MLX.abs(x, stream: s1)
    end
    
    y = MLX.abs(x, stream: s2)
    z = MLX.abs(y, stream: s2)
    
    MLX.eval(z)
    
    # Restore memory limit
    MLX.set_memory_limit(old_limit)
  end
end 