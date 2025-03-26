require_relative 'mlx_test_case'

class TestMemory < MLXTestCase
  def setup
    # Set random seed for reproducibility
    MLX.random.seed(42)
  end
  
  def test_memory_info
    # Test memory management functions
    
    # Test cache limit
    old_limit = MLX.set_cache_limit(0)
    
    a = MLX.zeros([4096])
    MLX.eval(a)
    a = nil
    
    assert_equal 0, MLX.get_cache_memory
    assert_equal 0, MLX.set_cache_limit(old_limit)
    assert_equal old_limit, MLX.set_cache_limit(old_limit)
    
    # Test memory limit
    old_limit = MLX.set_memory_limit(10)
    assert_equal 10, MLX.set_memory_limit(old_limit)
    assert_equal old_limit, MLX.set_memory_limit(old_limit)
    
    # Query active and peak memory
    a = MLX.zeros([4096])
    MLX.eval(a)
    MLX.synchronize
    
    active_mem = MLX.get_active_memory
    assert active_mem >= 4096 * 4, "Active memory should account for at least array size"
    
    b = MLX.zeros([4096])
    MLX.eval(b)
    b = nil
    MLX.synchronize
    
    new_active_mem = MLX.get_active_memory
    assert_equal active_mem, new_active_mem, "Active memory should be consistent"
    
    peak_mem = MLX.get_peak_memory
    assert peak_mem >= 4096 * 8, "Peak memory should account for both arrays"
    
    # Test cache memory (only on Metal)
    if MLX.metal.available?
      cache_mem = MLX.get_cache_memory
      assert cache_mem >= 4096 * 4, "Cache memory should include released array"
    end
    
    # Test clearing cache
    MLX.clear_cache
    assert_equal 0, MLX.get_cache_memory, "Cache should be empty after clearing"
    
    # Test resetting peak memory
    MLX.reset_peak_memory
    assert_equal 0, MLX.get_peak_memory, "Peak memory should be reset"
  end
  
  def test_wired_memory
    # Test wired memory functions (Metal only)
    skip "Metal is not available" unless MLX.metal.available?
    
    old_limit = MLX.set_wired_limit(1000)
    old_limit = MLX.set_wired_limit(0)
    assert_equal 1000, old_limit, "Previous limit should be returned"
    
    # Test setting limit above maximum
    max_size = MLX.metal.device_info["max_recommended_working_set_size"]
    assert_raises(ArgumentError) do
      MLX.set_wired_limit(max_size + 10)
    end
  end
  
  def test_memory_donation
    # Test memory donation functionality
    
    # Create a reference function
    ref_fn = lambda do |x|
      # Keep a reference to the input
      y = x + 1
      z = y + 2
      z
    end
    
    # Create a donation function
    donate_fn = lambda do |x|
      # Input should be donated
      y = x + 1
      z = y + 2
      z
    end
    
    # Create a test array
    x = MLX.ones([1024, 1024])
    MLX.eval(x)
    
    # Run without donation
    MLX.reset_peak_memory
    ref_result = ref_fn.call(x)
    MLX.eval(ref_result)
    peak_without_donation = MLX.get_peak_memory
    
    # Run with donation
    MLX.reset_peak_memory
    x_copy = MLX.array(x.to_a)  # Create a fresh copy
    donate_result = donate_fn.call(x_copy.donate)
    MLX.eval(donate_result)
    peak_with_donation = MLX.get_peak_memory
    
    # Donation should use less memory
    assert peak_with_donation < peak_without_donation, "Donation should reduce peak memory usage"
    
    # But results should be the same
    assert MLX.array_equal(ref_result, donate_result), "Results should be identical"
  end
  
  def test_memory_reuse
    # Test memory reuse patterns
    
    # Function to create and operate on large arrays
    fn = lambda do
      # Large arrays that will be allocated and freed
      a = MLX.zeros([2048, 2048])
      b = MLX.zeros([2048, 2048])
      
      # Use the arrays and discard them
      c = a + b
      a = nil
      b = nil
      
      # Create new arrays - memory should be reused
      d = MLX.zeros([2048, 2048])
      e = MLX.zeros([2048, 2048])
      
      # Final result
      f = c + d + e
      MLX.eval(f)
      f
    end
    
    # Run once to warm up cache
    result1 = fn.call
    
    # Clear cache to force new allocations
    MLX.clear_cache
    
    # Reset peak counter
    MLX.reset_peak_memory
    
    # Run again with empty cache
    result2 = fn.call
    peak_no_cache = MLX.get_peak_memory
    
    # Run a third time - should reuse memory
    MLX.reset_peak_memory
    result3 = fn.call
    peak_with_cache = MLX.get_peak_memory
    
    # On Metal, memory reuse should reduce the peak
    if MLX.metal.available?
      assert peak_with_cache <= peak_no_cache, "Memory reuse should reduce peak usage"
    end
    
    # Results should be identical
    assert MLX.array_equal(result1, result2), "Results should be identical"
    assert MLX.array_equal(result2, result3), "Results should be identical"
  end
end 