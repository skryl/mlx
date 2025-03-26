require_relative 'mlx_test_case'

class TestInit < MLXTestCase
  def test_constant
    value = 5.0
    
    [MLX.float32, MLX.float16].each do |dtype|
      initializer = MLX.nn.init.constant(value, dtype)
      [[3], [3, 3], [3, 3, 3]].each do |shape|
        result = initializer.call(MLX.array(MLX.zeros(shape)))
        assert_equal shape, result.shape
        assert_equal dtype, result.dtype
      end
    end
  end
  
  def test_normal
    mean = 0.0
    std = 1.0
    
    [MLX.float32, MLX.float16].each do |dtype|
      initializer = MLX.nn.init.normal(mean, std, dtype: dtype)
      [[3], [3, 3], [3, 3, 3]].each do |shape|
        result = initializer.call(MLX.array(MLX.empty(shape)))
        assert_equal shape, result.shape
        assert_equal dtype, result.dtype
      end
    end
  end
  
  def test_uniform
    low = -1.0
    high = 1.0
    
    [MLX.float32, MLX.float16].each do |dtype|
      initializer = MLX.nn.init.uniform(low, high, dtype)
      [[3], [3, 3], [3, 3, 3]].each do |shape|
        result = initializer.call(MLX.array(MLX.empty(shape)))
        assert_equal shape, result.shape
        assert_equal dtype, result.dtype
        assert MLX.all(result >= low).item && MLX.all(result <= high).item
      end
    end
  end
  
  def test_identity
    [MLX.float32, MLX.float16].each do |dtype|
      initializer = MLX.nn.init.identity(dtype)
      
      # Test valid case
      result = initializer.call(MLX.zeros([3, 3]))
      assert MLX.array_equal(result, MLX.eye(3))
      assert_equal dtype, result.dtype
      
      # Test invalid case (non-square matrix)
      assert_raises(ValueError) do
        initializer.call(MLX.zeros([3, 2]))
      end
    end
  end
  
  def test_glorot_normal
    [MLX.float32, MLX.float16].each do |dtype|
      initializer = MLX.nn.init.glorot_normal(dtype)
      [[3, 3], [3, 3, 3]].each do |shape|
        result = initializer.call(MLX.array(MLX.empty(shape)))
        assert_equal shape, result.shape
        assert_equal dtype, result.dtype
      end
    end
  end
  
  def test_glorot_uniform
    [MLX.float32, MLX.float16].each do |dtype|
      initializer = MLX.nn.init.glorot_uniform(dtype)
      [[3, 3], [3, 3, 3]].each do |shape|
        result = initializer.call(MLX.array(MLX.empty(shape)))
        assert_equal shape, result.shape
        assert_equal dtype, result.dtype
      end
    end
  end
  
  def test_he_normal
    [MLX.float32, MLX.float16].each do |dtype|
      initializer = MLX.nn.init.he_normal(dtype)
      [[3, 3], [3, 3, 3]].each do |shape|
        result = initializer.call(MLX.array(MLX.empty(shape)))
        assert_equal shape, result.shape
        assert_equal dtype, result.dtype
      end
    end
  end
  
  def test_he_uniform
    [MLX.float32, MLX.float16].each do |dtype|
      initializer = MLX.nn.init.he_uniform(dtype)
      [[3, 3], [3, 3, 3]].each do |shape|
        result = initializer.call(MLX.array(MLX.empty(shape)))
        assert_equal shape, result.shape
        assert_equal dtype, result.dtype
      end
    end
  end
  
  def test_sparse
    mean = 0.0
    std = 1.0
    sparsity = 0.5
    
    [MLX.float32, MLX.float16].each do |dtype|
      initializer = MLX.nn.init.sparse(sparsity, mean, std, dtype: dtype)
      [[3, 2], [2, 2], [4, 3]].each do |shape|
        result = initializer.call(MLX.array(MLX.empty(shape)))
        assert_equal shape, result.shape
        assert_equal dtype, result.dtype
        
        # Check sparsity
        zero_count = MLX.sum(result == 0).item
        expected_min_zeros = 0.5 * shape[0] * shape[1]
        assert zero_count >= expected_min_zeros, "Not sparse enough: expected at least #{expected_min_zeros} zeros, got #{zero_count}"
      end
      
      # Test invalid case (1D array)
      assert_raises(ValueError) do
        initializer.call(MLX.zeros([1]))
      end
    end
  end
  
  def test_orthogonal
    initializer = MLX.nn.init.orthogonal(gain: 1.0, dtype: MLX.float32)
    
    # Test with a square matrix
    shape = [4, 4]
    result = initializer.call(MLX.zeros(shape, dtype: MLX.float32))
    assert_equal shape, result.shape
    assert_equal MLX.float32, result.dtype
    
    i_matrix = result.matmul(result.T)
    eye = MLX.eye(shape[0], dtype: MLX.float32)
    assert MLX.allclose(i_matrix, eye, atol: 1e-5), "Orthogonal init failed on a square matrix"
    
    # Test with a rectangular matrix: more rows than cols
    shape = [6, 4]
    result = initializer.call(MLX.zeros(shape, dtype: MLX.float32))
    assert_equal shape, result.shape
    assert_equal MLX.float32, result.dtype
    
    i_matrix = result.T.matmul(result)
    eye = MLX.eye(shape[1], dtype: MLX.float32)
    assert MLX.allclose(i_matrix, eye, atol: 1e-5), "Orthogonal init failed on a rectangular matrix"
  end
  
  def test_variance_scaling
    scale = 2.0
    mode = "fan_avg"
    distribution = "normal"
    
    [MLX.float32, MLX.float16].each do |dtype|
      initializer = MLX.nn.init.variance_scaling(scale, mode, distribution, dtype: dtype)
      [[3, 3], [4, 5]].each do |shape|
        result = initializer.call(MLX.array(MLX.empty(shape)))
        assert_equal shape, result.shape
        assert_equal dtype, result.dtype
      end
    end
    
    # Test invalid mode
    assert_raises(ValueError) do
      MLX.nn.init.variance_scaling(scale, "invalid_mode", distribution)
    end
    
    # Test invalid distribution
    assert_raises(ValueError) do
      MLX.nn.init.variance_scaling(scale, mode, "invalid_distribution")
    end
  end
  
  def test_truncated_normal
    mean = 0.0
    std = 1.0
    
    [MLX.float32, MLX.float16].each do |dtype|
      initializer = MLX.nn.init.truncated_normal(mean, std, dtype: dtype)
      [[3], [3, 3], [3, 3, 3]].each do |shape|
        result = initializer.call(MLX.array(MLX.empty(shape)))
        assert_equal shape, result.shape
        assert_equal dtype, result.dtype
        
        # Check bounds (truncated normal typically uses ±2 standard deviations)
        assert MLX.all(result >= mean - 2*std).item
        assert MLX.all(result <= mean + 2*std).item
      end
    end
  end
  
  def test_ones
    [MLX.float32, MLX.float16].each do |dtype|
      initializer = MLX.nn.init.ones(dtype)
      [[3], [3, 3], [3, 3, 3]].each do |shape|
        result = initializer.call(MLX.array(MLX.empty(shape)))
        assert_equal shape, result.shape
        assert_equal dtype, result.dtype
        assert MLX.all(result == 1.0).item
      end
    end
  end
  
  def test_zeros
    [MLX.float32, MLX.float16].each do |dtype|
      initializer = MLX.nn.init.zeros(dtype)
      [[3], [3, 3], [3, 3, 3]].each do |shape|
        result = initializer.call(MLX.array(MLX.empty(shape)))
        assert_equal shape, result.shape
        assert_equal dtype, result.dtype
        assert MLX.all(result == 0.0).item
      end
    end
  end
end 