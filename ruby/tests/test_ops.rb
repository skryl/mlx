require_relative 'mlx_test_case'

class TestOps < MLXTestCase
  def test_full_ones_zeros
    # Test full
    x = MLX.full(2, 3.0)
    assert_equal [2], x.shape
    assert_array_equal(x, [3.0, 3.0])
    
    x = MLX.full([2, 3], 2.0)
    assert_equal MLX::FLOAT32, x.dtype
    assert_equal [2, 3], x.shape
    assert_array_equal(x, [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
    
    x = MLX.full([3, 2], MLX.array([false, true]))
    assert_equal MLX::BOOL, x.dtype
    assert_array_equal(x, [[false, true], [false, true], [false, true]])
    
    x = MLX.full([3, 2], MLX.array([2.0, 3.0]))
    assert_array_equal(x, [[2.0, 3.0], [2.0, 3.0], [2.0, 3.0]])
    
    # Test zeros
    x = MLX.zeros(2)
    assert_equal [2], x.shape
    assert_array_equal(x, [0.0, 0.0])
    
    # Test ones
    x = MLX.ones(2)
    assert_equal [2], x.shape
    assert_array_equal(x, [1.0, 1.0])
    
    # Test with different dtypes
    [MLX::BOOL, MLX::INT32, MLX::FLOAT32].each do |dtype|
      x = MLX.zeros([2, 2], dtype: dtype)
      assert_equal dtype, x.dtype
      assert MLX.array_equal(x, MLX.array([[0, 0], [0, 0]], dtype: dtype))
      y = MLX.zeros_like(x)
      assert_equal dtype, y.dtype
      assert MLX.array_equal(y, x)
      
      x = MLX.ones([2, 2], dtype: dtype)
      assert_equal dtype, x.dtype
      assert MLX.array_equal(x, MLX.array([[1, 1], [1, 1]], dtype: dtype))
      y = MLX.ones_like(x)
      assert_equal dtype, y.dtype
      assert MLX.array_equal(y, x)
    end
  end
  
  def test_add
    # Test array + array
    x = MLX.array(1)
    y = MLX.array(1)
    z = MLX.add(x, y)
    assert_equal 2, z.item
    
    # Test bool + int
    x = MLX.array(false, dtype: MLX::BOOL)
    z = x + 1
    assert_equal MLX::INT32, z.dtype
    assert_equal 1, z.item
    
    z = 2 + x
    assert_equal MLX::INT32, z.dtype
    assert_equal 2, z.item
    
    # Test uint32 + int
    x = MLX.array(1, dtype: MLX::UINT32)
    z = x + 3
    assert_equal MLX::UINT32, z.dtype
    assert_equal 4, z.item
    
    z = 3 + x
    assert_equal MLX::UINT32, z.dtype
    assert_equal 4, z.item
    
    # Test uint32 + float
    z = x + 3.0
    assert_equal MLX::FLOAT32, z.dtype
    assert_equal 4.0, z.item
    
    z = 3.0 + x
    assert_equal MLX::FLOAT32, z.dtype
    assert_equal 4.0, z.item
    
    # Test int64 + int/float
    x = MLX.array(1, dtype: MLX::INT64)
    z = x + 3
    assert_equal MLX::INT64, z.dtype
    assert_equal 4, z.item
    
    z = 3 + x
    assert_equal MLX::INT64, z.dtype
    assert_equal 4, z.item
    
    z = x + 3.0
    assert_equal MLX::FLOAT32, z.dtype
    assert_equal 4.0, z.item
    
    z = 3.0 + x
    assert_equal MLX::FLOAT32, z.dtype
    assert_equal 4.0, z.item
    
    # Test float32 + int
    x = MLX.array(1, dtype: MLX::FLOAT32)
    z = x + 3
    assert_equal MLX::FLOAT32, z.dtype
    assert_equal 4, z.item
    
    z = 3 + x
    assert_equal MLX::FLOAT32, z.dtype
    assert_equal 4, z.item
  end
  
  def test_subtract
    x = MLX.array(4.0)
    y = MLX.array(3.0)
    
    z = MLX.subtract(x, y)
    assert_equal MLX::FLOAT32, z.dtype
    assert_equal 1.0, z.item
    
    z = x - 3.0
    assert_equal MLX::FLOAT32, z.dtype
    assert_equal 1.0, z.item
    
    z = 5.0 - x
    assert_equal MLX::FLOAT32, z.dtype
    assert_equal 1.0, z.item
  end
  
  def test_multiply
    x = MLX.array(2.0)
    y = MLX.array(3.0)
    
    z = MLX.multiply(x, y)
    assert_equal MLX::FLOAT32, z.dtype
    assert_equal 6.0, z.item
    
    z = x * 3.0
    assert_equal MLX::FLOAT32, z.dtype
    assert_equal 6.0, z.item
    
    z = 3.0 * x
    assert_equal MLX::FLOAT32, z.dtype
    assert_equal 6.0, z.item
  end
  
  def test_divide
    x = MLX.array(2.0)
    y = MLX.array(4.0)
    
    z = MLX.divide(x, y)
    assert_equal MLX::FLOAT32, z.dtype
    assert_equal 0.5, z.item
    
    z = x / 4.0
    assert_equal MLX::FLOAT32, z.dtype
    assert_equal 0.5, z.item
    
    z = 1.0 / x
    assert_equal MLX::FLOAT32, z.dtype
    assert_equal 0.5, z.item
    
    # Test with other dtypes
    x = MLX.array(2.0, dtype: MLX::FLOAT16)
    z = x / 4.0
    assert_equal MLX::FLOAT16, z.dtype
    
    z = 4.0 / x
    assert_equal MLX::FLOAT16, z.dtype
    
    # Test integer division
    x = MLX.array(5)
    y = MLX.array(2)
    z = x / y
    assert_equal MLX::FLOAT32, z.dtype
    assert_equal 2.5, z.item
  end
  
  def test_remainder
    x = MLX.array(10)
    y = MLX.array(3)
    
    z = MLX.remainder(x, y)
    assert_equal MLX::INT32, z.dtype
    assert_equal 1, z.item
    
    z = x % 3
    assert_equal MLX::INT32, z.dtype
    assert_equal 1, z.item
    
    z = x % 3.0
    assert_equal MLX::FLOAT32, z.dtype
    assert_equal 1.0, z.item
    
    # Floating point remainder
    x = MLX.array(10.5)
    y = MLX.array(3.0)
    z = MLX.remainder(x, y)
    assert_equal MLX::FLOAT32, z.dtype
    assert_equal 1.5, z.item
    
    z = x % 3.0
    assert_equal MLX::FLOAT32, z.dtype
    assert_equal 1.5, z.item
  end
  
  def test_comparisons
    x = MLX.array([1, 2, 3, 4])
    y = MLX.array([2, 2, 2, 2])
    
    # Test equal
    result = MLX.equal(x, y)
    assert_array_equal(result, [false, true, false, false])
    
    # Test not_equal
    result = MLX.not_equal(x, y)
    assert_array_equal(result, [true, false, true, true])
    
    # Test less
    result = MLX.less(x, y)
    assert_array_equal(result, [true, false, false, false])
    
    # Test less_equal
    result = MLX.less_equal(x, y)
    assert_array_equal(result, [true, true, false, false])
    
    # Test greater
    result = MLX.greater(x, y)
    assert_array_equal(result, [false, false, true, true])
    
    # Test greater_equal
    result = MLX.greater_equal(x, y)
    assert_array_equal(result, [false, true, true, true])
  end
  
  def test_array_equal
    x = MLX.array([1, 2, 3])
    y = MLX.array([1, 2, 3])
    z = MLX.array([1, 2, 4])
    
    # Same values, same shape
    assert MLX.array_equal(x, y)
    
    # Different values
    refute MLX.array_equal(x, z)
    
    # Different shape
    refute MLX.array_equal(x, MLX.array([1, 2, 3, 4]))
    
    # Different dtype
    refute MLX.array_equal(x, MLX.array([1.0, 2.0, 3.0]))
  end
  
  def test_isnan_isinf_isfinite
    # Create an array with special values
    x = MLX.array([0.0, Float::INFINITY, -Float::INFINITY, Float::NAN])
    
    # Test isnan
    assert_array_equal(MLX.isnan(x), [false, false, false, true])
    
    # Test isinf
    assert_array_equal(MLX.isinf(x), [false, true, true, false])
    
    # Test isfinite
    assert_array_equal(MLX.isfinite(x), [true, false, false, false])
  end
  
  def test_min_max
    x = MLX.array([1, 3, 2, 5, 4])
    
    # Test min
    assert_equal 1, MLX.min(x).item
    
    # Test max
    assert_equal 5, MLX.max(x).item
    
    # Test minimum (element-wise)
    y = MLX.array([2, 1, 3, 4, 6])
    assert_array_equal(MLX.minimum(x, y), [1, 1, 2, 4, 4])
    
    # Test maximum (element-wise)
    assert_array_equal(MLX.maximum(x, y), [2, 3, 3, 5, 6])
    
    # Test min/max with axis
    x = MLX.array([[1, 2, 3], [4, 5, 6]])
    assert_array_equal(MLX.min(x, axis: 0), [1, 2, 3])
    assert_array_equal(MLX.min(x, axis: 1), [1, 4])
    assert_array_equal(MLX.max(x, axis: 0), [4, 5, 6])
    assert_array_equal(MLX.max(x, axis: 1), [3, 6])
  end
  
  def test_argmin_argmax
    x = MLX.array([3, 1, 4, 1, 5, 9, 2, 6])
    
    # Test argmin
    assert_equal 1, MLX.argmin(x).item
    
    # Test argmax
    assert_equal 5, MLX.argmax(x).item
    
    # Test with axis
    x = MLX.array([[1, 2, 3], [4, 0, 6]])
    assert_array_equal(MLX.argmin(x, axis: 0), [0, 1, 0])
    assert_array_equal(MLX.argmin(x, axis: 1), [0, 1])
    assert_array_equal(MLX.argmax(x, axis: 0), [1, 0, 1])
    assert_array_equal(MLX.argmax(x, axis: 1), [2, 2])
  end
  
  def test_math_functions
    # Test basic math operations
    x = MLX.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # Test abs
    assert_array_equal(MLX.abs(x), [2.0, 1.0, 0.0, 1.0, 2.0])
    
    # Test negative
    assert_array_equal(MLX.negative(x), [2.0, 1.0, 0.0, -1.0, -2.0])
    
    # Test sign
    assert_array_equal(MLX.sign(x), [-1.0, -1.0, 0.0, 1.0, 1.0])
    
    # Test square
    assert_array_equal(MLX.square(x), [4.0, 1.0, 0.0, 1.0, 4.0])
    
    # Test sqrt (on positive numbers)
    assert_array_equal(MLX.sqrt(MLX.array([0.0, 1.0, 4.0, 9.0])), [0.0, 1.0, 2.0, 3.0])
    
    # Test exp
    exp_vals = MLX.exp(MLX.array([0.0, 1.0, 2.0]))
    assert_array_equal(exp_vals, [1.0, Math.exp(1), Math.exp(2)], atol: 1e-5)
    
    # Test log
    log_vals = MLX.log(MLX.array([1.0, 2.0, Math::E]))
    assert_array_equal(log_vals, [0.0, Math.log(2), 1.0], atol: 1e-5)
    
    # Test log2
    log2_vals = MLX.log2(MLX.array([1.0, 2.0, 4.0, 8.0]))
    assert_array_equal(log2_vals, [0.0, 1.0, 2.0, 3.0], atol: 1e-5)
    
    # Test log10
    log10_vals = MLX.log10(MLX.array([1.0, 10.0, 100.0, 1000.0]))
    assert_array_equal(log10_vals, [0.0, 1.0, 2.0, 3.0], atol: 1e-5)
  end
  
  def test_trig_functions
    x = MLX.array([0.0, Math::PI/4, Math::PI/2, Math::PI])
    
    # Test sin
    sin_vals = MLX.sin(x)
    expected_sin = [0.0, Math.sin(Math::PI/4), 1.0, 0.0]
    assert_array_equal(sin_vals, expected_sin, atol: 1e-5)
    
    # Test cos
    cos_vals = MLX.cos(x)
    expected_cos = [1.0, Math.cos(Math::PI/4), 0.0, -1.0]
    assert_array_equal(cos_vals, expected_cos, atol: 1e-5)
    
    # Test tan
    tan_vals = MLX.tan(x[0..2])  # Skip PI which gives infinity
    expected_tan = [0.0, Math.tan(Math::PI/4), Float::INFINITY]
    # Use element-wise comparison since infinity requires special handling
    tan_vals.to_list.zip(expected_tan).each do |actual, expected|
      if expected == Float::INFINITY
        assert actual > 1e5  # Very large number
      else
        assert_in_delta expected, actual, 1e-5
      end
    end
  end
  
  def test_reductions
    x = MLX.array([[1, 2, 3], [4, 5, 6]])
    
    # Test sum
    assert_equal 21, MLX.sum(x).item
    assert_array_equal(MLX.sum(x, axis: 0), [5, 7, 9])
    assert_array_equal(MLX.sum(x, axis: 1), [6, 15])
    
    # Test prod
    assert_equal 720, MLX.prod(x).item
    assert_array_equal(MLX.prod(x, axis: 0), [4, 10, 18])
    assert_array_equal(MLX.prod(x, axis: 1), [6, 120])
    
    # Test mean
    assert_equal 3.5, MLX.mean(x).item
    assert_array_equal(MLX.mean(x, axis: 0), [2.5, 3.5, 4.5])
    assert_array_equal(MLX.mean(x, axis: 1), [2.0, 5.0])
    
    # Test var
    assert_in_delta 2.9167, MLX.var(x).item, 1e-4
    assert_array_equal(MLX.var(x, axis: 0), [2.25, 2.25, 2.25], atol: 1e-4)
    assert_array_equal(MLX.var(x, axis: 1), [0.6667, 0.6667], atol: 1e-4)
    
    # Test std
    assert_in_delta 1.7078, MLX.std(x).item, 1e-4
    assert_array_equal(MLX.std(x, axis: 0), [1.5, 1.5, 1.5], atol: 1e-4)
    assert_array_equal(MLX.std(x, axis: 1), [0.8165, 0.8165], atol: 1e-4)
  end
  
  def test_logical_ops
    x = MLX.array([true, false, true])
    y = MLX.array([true, true, false])
    
    # Test logical_not
    assert_array_equal(MLX.logical_not(x), [false, true, false])
    
    # Test logical_and
    assert_array_equal(MLX.logical_and(x, y), [true, false, false])
    
    # Test logical_or
    assert_array_equal(MLX.logical_or(x, y), [true, true, true])
    
    # Test logical_xor
    assert_array_equal(MLX.logical_xor(x, y), [false, true, true])
  end
  
  def test_all_any
    # Test all
    assert MLX.all(MLX.array([true, true, true])).item
    refute MLX.all(MLX.array([true, false, true])).item
    
    # Test with non-boolean arrays (treated as boolean mask)
    assert MLX.all(MLX.array([1, 2, 3])).item
    refute MLX.all(MLX.array([1, 0, 3])).item
    
    # Test all with axis
    x = MLX.array([[true, true], [true, false]])
    assert_array_equal(MLX.all(x, axis: 0), [true, false])
    assert_array_equal(MLX.all(x, axis: 1), [true, false])
    
    # Test any
    assert MLX.any(MLX.array([false, true, false])).item
    refute MLX.any(MLX.array([false, false, false])).item
    
    # Test with non-boolean arrays (treated as boolean mask)
    assert MLX.any(MLX.array([0, 2, 0])).item
    refute MLX.any(MLX.array([0, 0, 0])).item
    
    # Test any with axis
    x = MLX.array([[false, true], [false, false]])
    assert_array_equal(MLX.any(x, axis: 0), [false, true])
    assert_array_equal(MLX.any(x, axis: 1), [true, false])
  end
  
  def test_array_manipulation
    # Test reshape
    x = MLX.array([1, 2, 3, 4, 5, 6])
    assert_array_equal(MLX.reshape(x, [2, 3]), [[1, 2, 3], [4, 5, 6]])
    assert_array_equal(MLX.reshape(x, [3, 2]), [[1, 2], [3, 4], [5, 6]])
    
    # Test transpose
    x = MLX.array([[1, 2, 3], [4, 5, 6]])
    assert_array_equal(MLX.transpose(x), [[1, 4], [2, 5], [3, 6]])
    
    # Test permute_dims
    x = MLX.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # Shape: 2x2x2
    assert_array_equal(
      MLX.permute_dims(x, [2, 0, 1]),  # New order: depth, batch, height
      [[[1, 3], [5, 7]], [[2, 4], [6, 8]]]
    )
    
    # Test concatenate
    a = MLX.array([[1, 2], [3, 4]])
    b = MLX.array([[5, 6]])
    assert_array_equal(MLX.concatenate([a, b], axis: 0), [[1, 2], [3, 4], [5, 6]])
    
    # Test stack
    a = MLX.array([1, 2, 3])
    b = MLX.array([4, 5, 6])
    assert_array_equal(MLX.stack([a, b]), [[1, 2, 3], [4, 5, 6]])
    assert_array_equal(MLX.stack([a, b], axis: 1), [[1, 4], [2, 5], [3, 6]])
  end
  
  def test_softmax
    # Test softmax with 1D
    x = MLX.array([1.0, 2.0, 3.0])
    sm = MLX.softmax(x)
    
    # Calculate expected values
    max_x = 3.0
    exp_x = [Math.exp(1.0 - max_x), Math.exp(2.0 - max_x), Math.exp(3.0 - max_x)]
    sum_exp = exp_x.sum
    expected = exp_x.map { |v| v / sum_exp }
    
    assert_array_equal(sm, expected, atol: 1e-5)
    
    # Test softmax with 2D (axis=1)
    x = MLX.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    sm = MLX.softmax(x, axis: 1)
    
    # Expected values for first row
    max_row1 = 3.0
    exp_row1 = [Math.exp(1.0 - max_row1), Math.exp(2.0 - max_row1), Math.exp(3.0 - max_row1)]
    sum_exp_row1 = exp_row1.sum
    expected_row1 = exp_row1.map { |v| v / sum_exp_row1 }
    
    # Expected values for second row
    max_row2 = 6.0
    exp_row2 = [Math.exp(4.0 - max_row2), Math.exp(5.0 - max_row2), Math.exp(6.0 - max_row2)]
    sum_exp_row2 = exp_row2.sum
    expected_row2 = exp_row2.map { |v| v / sum_exp_row2 }
    
    assert_array_equal(sm, [expected_row1, expected_row2], atol: 1e-5)
  end
end 