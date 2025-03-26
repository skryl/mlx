require_relative 'mlx_test_case'

class TestArray < MLXTestCase
  def test_array_creation
    # Test creating an array from a Ruby array
    arr = MLX.array([1, 2, 3, 4, 5])
    assert_equal 5, arr.size
    assert_equal [5], arr.shape
    assert_equal 1, arr.ndim
    
    # Test creating an array with specific dtype
    arr_int32 = MLX.array([1, 2, 3], dtype: MLX::INT32)
    assert_equal MLX::INT32, arr_int32.dtype
    
    # Test from nested arrays
    nested = MLX.array([[1, 2, 3], [4, 5, 6]])
    assert_equal [2, 3], nested.shape
    assert_equal 2, nested.ndim
    
    # Test creation with zeros, ones, full
    zeros = MLX.zeros([2, 3])
    assert_equal [2, 3], zeros.shape
    assert_equal MLX::FLOAT32, zeros.dtype
    
    ones = MLX.ones([3, 2], dtype: MLX::INT32)
    assert_equal [3, 2], ones.shape
    assert_equal MLX::INT32, ones.dtype
    
    full = MLX.full([2, 2], 7, dtype: MLX::FLOAT32)
    assert_equal [2, 2], full.shape
    assert_equal MLX::FLOAT32, full.dtype
  end

  def test_array_operations
    # Test array + array
    arr1 = MLX.array([1, 2, 3])
    arr2 = MLX.array([4, 5, 6])
    result = arr1 + arr2
    
    assert_array_equal(result, [5, 7, 9])
    
    # Test array - array
    diff = arr2 - arr1
    assert_array_equal(diff, [3, 3, 3])
    
    # Test array * array
    prod = arr1 * arr2
    assert_array_equal(prod, [4, 10, 18])
    
    # Test array / array
    div = arr2 / arr1
    assert_array_equal(div, [4, 2.5, 2])
    
    # Test array + scalar
    scalar_sum = arr1 + 10
    assert_array_equal(scalar_sum, [11, 12, 13])
    
    # Test scalar + array
    scalar_sum2 = 10 + arr1
    assert_array_equal(scalar_sum2, [11, 12, 13])
    
    # Test array - scalar
    scalar_diff = arr2 - 2
    assert_array_equal(scalar_diff, [2, 3, 4])
    
    # Test scalar - array
    scalar_diff2 = 10 - arr1
    assert_array_equal(scalar_diff2, [9, 8, 7])
    
    # Test array * scalar
    scalar_prod = arr1 * 2
    assert_array_equal(scalar_prod, [2, 4, 6])
    
    # Test array / scalar
    scalar_div = arr2 / 2
    assert_array_equal(scalar_div, [2, 2.5, 3])
  end
  
  def test_array_equality
    a = MLX.array([1, 2, 3])
    b = MLX.array([1, 2, 3])
    c = MLX.array([1, 2, 4])
    
    # Test equality operators
    assert MLX.all(a == b).item
    refute MLX.all(a == c).item
    assert MLX.any(a != c).item
    refute MLX.any(a != b).item
    
    # Test equality with scalar
    assert MLX.any(a == 1).item
    refute MLX.all(a == 1).item
    assert MLX.all(a != 5).item
  end
  
  def test_array_comparison
    a = MLX.array([1, 2, 3])
    b = MLX.array([0, 2, 4])
    
    # Test comparison operators
    assert MLX.any(a > b).item
    assert MLX.any(a < b).item
    assert MLX.any(a >= b).item
    assert MLX.any(a <= b).item
    
    # With scalars
    assert MLX.all(a > 0).item
    assert MLX.all(a < 10).item
    assert MLX.any(a == 2).item
    assert MLX.all(a >= 1).item
    assert MLX.all(a <= 3).item
  end
  
  def test_array_reshaping
    a = MLX.array([1, 2, 3, 4, 5, 6])
    
    # Test reshape
    b = MLX.reshape(a, [2, 3])
    assert_equal [2, 3], b.shape
    
    # Test transpose
    c = MLX.reshape(a, [3, 2])
    d = MLX.transpose(c)
    assert_equal [2, 3], d.shape
    
    # Test flatten
    e = MLX.reshape(a, [2, 3])
    f = MLX.reshape(e, [-1])
    assert_equal [6], f.shape
  end
  
  def test_array_indexing
    a = MLX.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    # Test basic indexing
    assert_equal 1, a[0, 0].item
    assert_equal 5, a[1, 1].item
    assert_equal 9, a[2, 2].item
    
    # Test slicing
    row = a[1]
    assert_array_equal(row, [4, 5, 6])
    
    col = a[:, 1]
    assert_array_equal(col, [2, 5, 8])
    
    # Test advanced indexing
    selected = a[[0, 2], [0, 2]]
    assert_array_equal(selected, [1, 9])
  end
  
  def test_array_math_functions
    a = MLX.array([1, 2, 3, 4])
    
    # Test abs
    assert_array_equal(MLX.abs(MLX.array([-1, -2, 3, -4])), [1, 2, 3, 4])
    
    # Test sqrt
    assert_array_equal(MLX.sqrt(a), [1, 1.4142, 1.7321, 2], atol: 1e-4)
    
    # Test exp
    exp_result = MLX.exp(MLX.array([0, 1, 2]))
    assert_array_equal(exp_result, [1, 2.7183, 7.3891], atol: 1e-4)
    
    # Test log
    log_result = MLX.log(MLX.array([1, 2, 10]))
    assert_array_equal(log_result, [0, 0.6931, 2.3026], atol: 1e-4)
    
    # Test sin, cos
    sin_result = MLX.sin(MLX.array([0, Math::PI/2, Math::PI]))
    assert_array_equal(sin_result, [0, 1, 0], atol: 1e-4)
    
    cos_result = MLX.cos(MLX.array([0, Math::PI/2, Math::PI]))
    assert_array_equal(cos_result, [1, 0, -1], atol: 1e-4)
  end
  
  def test_array_reduction
    a = MLX.array([[1, 2, 3], [4, 5, 6]])
    
    # Test sum
    assert_equal 21, MLX.sum(a).item
    assert_array_equal(MLX.sum(a, axis: 0), [5, 7, 9])
    assert_array_equal(MLX.sum(a, axis: 1), [6, 15])
    
    # Test mean
    assert_equal 3.5, MLX.mean(a).item
    assert_array_equal(MLX.mean(a, axis: 0), [2.5, 3.5, 4.5])
    assert_array_equal(MLX.mean(a, axis: 1), [2, 5])
    
    # Test min/max
    assert_equal 1, MLX.min(a).item
    assert_equal 6, MLX.max(a).item
    assert_array_equal(MLX.min(a, axis: 0), [1, 2, 3])
    assert_array_equal(MLX.max(a, axis: 1), [3, 6])
    
    # Test argmin/argmax
    assert_equal 0, MLX.argmin(a).item
    assert_equal 5, MLX.argmax(a).item
    assert_array_equal(MLX.argmin(a, axis: 0), [0, 0, 0])
    assert_array_equal(MLX.argmax(a, axis: 1), [2, 2])
  end
  
  def test_array_logical
    a = MLX.array([true, false, true])
    b = MLX.array([true, true, false])
    
    # Test logical operations
    assert_array_equal(MLX.logical_and(a, b), [true, false, false])
    assert_array_equal(MLX.logical_or(a, b), [true, true, true])
    assert_array_equal(MLX.logical_not(a), [false, true, false])
    assert_array_equal(MLX.logical_xor(a, b), [false, true, true])
    
    # Test all and any
    assert MLX.any(a).item
    refute MLX.all(a).item
    assert MLX.any(b).item
    refute MLX.all(b).item
  end
  
  def test_array_conversion
    # Test dtype conversion
    a = MLX.array([1, 2, 3], dtype: MLX::FLOAT32)
    
    b = MLX.astype(a, MLX::INT32)
    assert_equal MLX::INT32, b.dtype
    assert_array_equal(b, [1, 2, 3])
    
    c = MLX.astype(a, MLX::FLOAT16)
    assert_equal MLX::FLOAT16, c.dtype
    assert_array_equal(c, [1, 2, 3])
    
    # Test to_list - if implemented
    if a.respond_to?(:to_list)
      assert_equal [1.0, 2.0, 3.0], a.to_list
    end
  end
  
  def test_array_device
    # Only test if metal is available
    if MLX.metal.is_available
      a = MLX.array([1, 2, 3])
      
      # Test device transformation
      cpu_arr = MLX.to_cpu(a)  
      gpu_arr = MLX.to_gpu(a)
      
      assert_array_equal(cpu_arr, a)
      assert_array_equal(gpu_arr, a)
    end
  end
end 