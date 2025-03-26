require_relative 'mlx_test_case'

class TestRandom < MLXTestCase
  def test_global_rng
    MLX.random.seed(3)
    a = MLX.random.uniform
    b = MLX.random.uniform
    
    MLX.random.seed(3)
    x = MLX.random.uniform
    y = MLX.random.uniform
    
    assert_equal a.item, x.item
    assert_equal y.item, b.item
  end
  
  def test_key
    k1 = MLX.random.key(0)
    k2 = MLX.random.key(0)
    assert MLX.array_equal(k1, k2)
    
    k2 = MLX.random.key(1)
    refute MLX.array_equal(k1, k2)
  end
  
  def test_key_split
    key = MLX.random.key(0)
    
    k1, k2 = MLX.random.split(key)
    refute MLX.array_equal(k1, k2)
    
    r1, r2 = MLX.random.split(key)
    assert MLX.array_equal(k1, r1)
    assert MLX.array_equal(k2, r2)
    
    keys = MLX.random.split(key, 10)
    assert_equal [10, 2], keys.shape
  end
  
  def test_uniform
    key = MLX.random.key(0)
    a = MLX.random.uniform(key: key)
    assert_equal [], a.shape
    assert_equal MLX.float32, a.dtype
    
    b = MLX.random.uniform(key: key)
    assert_equal a.item, b.item
    
    a = MLX.random.uniform(shape: [2, 3])
    assert_equal [2, 3], a.shape
    
    a = MLX.random.uniform(shape: [1000], low: -1, high: 5)
    assert MLX.all((a > -1) & (a < 5)).item
    
    a = MLX.random.uniform(shape: [1000], low: MLX.array(-1), high: 5)
    assert MLX.all((a > -1) & (a < 5)).item
    
    a = MLX.random.uniform(low: -0.1, high: 0.1, shape: [1], dtype: MLX.bfloat16)
    assert_equal MLX.bfloat16, a.dtype
    
    assert_equal MLX.random.uniform.dtype, MLX.random.uniform(dtype: nil).dtype
  end
  
  def test_normal_and_laplace
    # Same tests for normal and laplace
    [MLX.random.method(:normal), MLX.random.method(:laplace)].each do |distribution_sampler|
      key = MLX.random.key(0)
      a = distribution_sampler.call(key: key)
      assert_equal [], a.shape
      assert_equal MLX.float32, a.dtype
      
      b = distribution_sampler.call(key: key)
      assert_equal a.item, b.item
      
      a = distribution_sampler.call(shape: [2, 3])
      assert_equal [2, 3], a.shape
      
      # Generate in float16 or bfloat16
      [MLX.float16, MLX.bfloat16].each do |t|
        a = distribution_sampler.call(dtype: t)
        assert_equal t, a.dtype
      end
      
      # Generate with a given mean and standard deviation
      loc = 1.0
      scale = 2.0
      
      a = distribution_sampler.call(shape: [3, 2], loc: loc, scale: scale, key: key)
      b = scale * distribution_sampler.call(shape: [3, 2], key: key) + loc
      assert MLX.allclose(a, b)
      
      a = distribution_sampler.call(shape: [3, 2], loc: loc, scale: scale, dtype: MLX.float16, key: key)
      b = scale * distribution_sampler.call(shape: [3, 2], dtype: MLX.float16, key: key) + loc
      assert MLX.allclose(a, b)
      
      assert_equal distribution_sampler.call.dtype, distribution_sampler.call(dtype: nil).dtype
      
      # Test not getting -inf or inf with half precision
      [MLX.float16, MLX.bfloat16].each do |hp|
        a = MLX.abs(distribution_sampler.call(shape: [10000], loc: 0, scale: 1, dtype: hp))
        assert MLX.all(a < MLX.inf)
      end
    end
  end
  
  def test_multivariate_normal
    key = MLX.random.key(0)
    mean = MLX.array([0, 0])
    cov = MLX.array([[1, 0], [0, 1]])
    
    a = MLX.random.multivariate_normal(mean, cov, key: key, stream: MLX.cpu)
    assert_equal [2], a.shape
    
    # Check dtypes
    [MLX.float32].each do |t|
      a = MLX.random.multivariate_normal(mean, cov, dtype: t, key: key, stream: MLX.cpu)
      assert_equal t, a.dtype
    end
    
    [
      MLX.int8, MLX.int32, MLX.int64,
      MLX.uint8, MLX.uint32, MLX.uint64,
      MLX.float16, MLX.bfloat16
    ].each do |t|
      assert_raises(ValueError) do
        MLX.random.multivariate_normal(mean, cov, dtype: t, key: key, stream: MLX.cpu)
      end
    end
    
    # Check incompatible shapes
    assert_raises(ValueError) do
      mean = MLX.zeros([2, 2])
      cov = MLX.zeros([2, 2])
      MLX.random.multivariate_normal(mean, cov, shape: [3], key: key, stream: MLX.cpu)
    end
    
    assert_raises(ValueError) do
      mean = MLX.zeros([2])
      cov = MLX.zeros([2, 2, 2])
      MLX.random.multivariate_normal(mean, cov, shape: [3], key: key, stream: MLX.cpu)
    end
    
    assert_raises(ValueError) do
      mean = MLX.zeros([3])
      cov = MLX.zeros([2, 2])
      MLX.random.multivariate_normal(mean, cov, key: key, stream: MLX.cpu)
    end
    
    assert_raises(ValueError) do
      mean = MLX.zeros([2])
      cov = MLX.zeros([2, 3])
      MLX.random.multivariate_normal(mean, cov, key: key, stream: MLX.cpu)
    end
    
    # Different shape of mean and cov
    mean = MLX.array([[0, 7], [1, 2], [3, 4]])
    cov = MLX.array([[1, 0.5], [0.5, 1]])
    a = MLX.random.multivariate_normal(mean, cov, shape: [4, 3], stream: MLX.cpu)
    assert_equal [4, 3, 2], a.shape
    
    # Check correctness of the mean and covariance
    n_test = 100_000
    
    def check_jointly_gaussian(data, mean, cov)
      empirical_mean = MLX.mean(data, axis: 0)
      empirical_cov = (data - empirical_mean).T.matmul(data - empirical_mean) / data.shape[0]
      n = data.shape[1]
      
      delta = 10 * n**2 / Math.sqrt(n_test)
      assert MLX.allclose(empirical_mean, mean, rtol: 0.0, atol: delta)
      assert MLX.allclose(empirical_cov, cov, rtol: 0.0, atol: delta)
    end
    
    mean = MLX.array([4.0, 7.0])
    cov = MLX.array([[2, 0.5], [0.5, 1]])
    data = MLX.random.multivariate_normal(mean, cov, shape: [n_test], key: key, stream: MLX.cpu)
    check_jointly_gaussian(data, mean, cov)
    
    mean = MLX.arange(3)
    cov = MLX.array([[1, -1, 0.5], [-1, 1, -0.5], [0.5, -0.5, 1]])
    data = MLX.random.multivariate_normal(mean, cov, shape: [n_test], key: key, stream: MLX.cpu)
    check_jointly_gaussian(data, mean, cov)
  end
  
  def test_randint
    key = MLX.random.key(0)
    a = MLX.random.randint(0, 10, key: key)
    assert_equal [], a.shape
    assert_equal MLX.int32, a.dtype
    
    b = MLX.random.randint(0, 10, key: key)
    assert_equal a.item, b.item
    
    a = MLX.random.randint(0, 10, shape: [2, 3])
    assert_equal [2, 3], a.shape
    
    a = MLX.random.randint(0, 10, shape: [1000])
    assert MLX.all((a >= 0) & (a < 10)).item
    
    # Test different dtypes
    [MLX.int32, MLX.int8, MLX.int16, MLX.int64, MLX.uint8, MLX.uint16, MLX.uint32, MLX.uint64].each do |dt|
      a = MLX.random.randint(0, 5, shape: [5], dtype: dt)
      assert_equal dt, a.dtype
    end
    
    # Test with different low/high types
    a = MLX.random.randint(MLX.array(0), MLX.array(10), shape: [5])
    assert_equal MLX.int32, a.dtype
    
    # Test with array inputs
    a = MLX.random.randint(MLX.array([0, 1]), MLX.array([10, 20]), shape: [5, 2])
    assert_equal [5, 2], a.shape
    assert MLX.all((a[:, 0] >= 0) & (a[:, 0] < 10)).item
    assert MLX.all((a[:, 1] >= 1) & (a[:, 1] < 20)).item
  end
  
  def test_bernoulli
    key = MLX.random.key(0)
    a = MLX.random.bernoulli(0.5, key: key)
    assert_equal [], a.shape
    assert_equal MLX.bool, a.dtype
    
    b = MLX.random.bernoulli(0.5, key: key)
    assert_equal a.item, b.item
    
    a = MLX.random.bernoulli(0.5, shape: [2, 3])
    assert_equal [2, 3], a.shape
    
    probs = 0.5 * MLX.ones([3, 4])
    a = MLX.random.bernoulli(probs)
    assert_equal [3, 4], a.shape
    
    # Test with different dtypes
    a = MLX.random.bernoulli(0.5, shape: [100], dtype: MLX.float32)
    assert_equal MLX.float32, a.dtype
    
    a = MLX.random.bernoulli(0.5, shape: [100], dtype: MLX.int32)
    assert_equal MLX.int32, a.dtype
  end
  
  def test_truncated_normal
    key = MLX.random.key(0)
    a = MLX.random.truncated_normal(-2, 2, key: key)
    assert_equal [], a.shape
    assert_equal MLX.float32, a.dtype
    
    b = MLX.random.truncated_normal(-2, 2, key: key)
    assert_equal a.item, b.item
    
    a = MLX.random.truncated_normal(-2, 2, shape: [2, 3])
    assert_equal [2, 3], a.shape
    
    # Test with different dtypes
    a = MLX.random.truncated_normal(-2, 2, shape: [5], dtype: MLX.float16)
    assert_equal MLX.float16, a.dtype
    
    # Test that values are within bounds
    a = MLX.random.truncated_normal(-2, 2, shape: [1000])
    assert MLX.all((a >= -2) & (a <= 2)).item
    
    # Test with means and scales
    a = MLX.random.truncated_normal(-2, 2, shape: [1000], loc: 1.0, scale: 2.0)
    assert MLX.all((a >= -2) & (a <= 2)).item
    
    # Test with array inputs
    lower = MLX.array([-1, 0, 1])
    upper = MLX.array([2, 3, 4])
    a = MLX.random.truncated_normal(lower, upper, shape: [5, 3])
    assert_equal [5, 3], a.shape
    
    # Check bounds
    assert MLX.all((a >= MLX.expand_dims(lower, 0)) & (a <= MLX.expand_dims(upper, 0))).item
  end
  
  def test_gumbel
    key = MLX.random.key(0)
    a = MLX.random.gumbel(key: key)
    assert_equal [], a.shape
    assert_equal MLX.float32, a.dtype
    
    b = MLX.random.gumbel(key: key)
    assert_equal a.item, b.item
    
    a = MLX.random.gumbel(shape: [2, 3])
    assert_equal [2, 3], a.shape
    
    # Test with different dtypes
    a = MLX.random.gumbel(shape: [5], dtype: MLX.float16)
    assert_equal MLX.float16, a.dtype
  end
  
  def test_categorical
    key = MLX.random.key(0)
    logits = MLX.array([0.1, 0.5, 0.4])
    a = MLX.random.categorical(logits, key: key)
    assert_equal [], a.shape
    assert_equal MLX.int32, a.dtype
    
    b = MLX.random.categorical(logits, key: key)
    assert_equal a.item, b.item
    
    # Test with multi-dimensional input
    logits = MLX.array([[0.1, 0.5, 0.4], [0.8, 0.1, 0.1]])
    a = MLX.random.categorical(logits)
    assert_equal [2], a.shape
    
    # Test with different num_samples
    a = MLX.random.categorical(logits, num_samples: 5)
    assert_equal [2, 5], a.shape
    
    # Test with different dtypes
    a = MLX.random.categorical(logits, dtype: MLX.int64)
    assert_equal MLX.int64, a.dtype
  end
  
  def test_permutation
    key = MLX.random.key(0)
    a = MLX.random.permutation(10, key: key)
    assert_equal [10], a.shape
    assert_equal MLX.int32, a.dtype
    
    # Test deterministic output with same key
    b = MLX.random.permutation(10, key: key)
    assert MLX.array_equal(a, b)
    
    # Check that result is a permutation
    a = MLX.random.permutation(1000)
    assert MLX.array_equal(MLX.sort(a), MLX.arange(1000))
    
    # Test with array input
    x = MLX.array([1, 2, 3, 4])
    a = MLX.random.permutation(x)
    assert MLX.array_equal(MLX.sort(a), x)
    
    # Test with multi-dimensional array
    x = MLX.reshape(MLX.arange(12), [3, 4])
    a = MLX.random.permutation(x)
    assert_equal [3, 4], a.shape
    assert MLX.array_equal(MLX.sort(a.reshape(12)), MLX.arange(12))
    
    # Test with axis parameter
    a = MLX.random.permutation(x, axis: 1)
    assert_equal [3, 4], a.shape
    assert MLX.array_equal(MLX.sort(a, axis: 1), x)
  end
end 