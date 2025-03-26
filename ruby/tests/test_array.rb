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

  # Array transform tests
  def test_reshape
    # Test reshape on 1D array
    x = MLX.arange(6)
    y = MLX.reshape(x, [2, 3])
    assert_equal [2, 3], y.shape
    assert MLX.array_equal(y, MLX.array([[0, 1, 2], [3, 4, 5]]))
    
    # Test reshape on 2D array
    x = MLX.array([[1, 2], [3, 4], [5, 6]])
    y = MLX.reshape(x, [2, 3])
    assert_equal [2, 3], y.shape
    assert MLX.array_equal(y, MLX.array([[1, 2, 3], [4, 5, 6]]))
    
    # Test with -1 dimension
    x = MLX.arange(12)
    y = MLX.reshape(x, [-1, 4])
    assert_equal [3, 4], y.shape
    
    # Test with multiple -1 dimensions (should raise error)
    assert_raises(ValueError) do
      MLX.reshape(x, [-1, -1])
    end
    
    # Test preserving dtype
    x = MLX.array([1.0, 2.0, 3.0, 4.0])
    y = MLX.reshape(x, [2, 2])
    assert_equal x.dtype, y.dtype
  end
  
  def test_transpose
    # Test basic transpose
    x = MLX.array([[1, 2, 3], [4, 5, 6]])
    y = MLX.transpose(x)
    assert_equal [3, 2], y.shape
    assert MLX.array_equal(y, MLX.array([[1, 4], [2, 5], [3, 6]]))
    
    # Test with specific axes
    x = MLX.reshape(MLX.arange(24), [2, 3, 4])
    y = MLX.transpose(x, axes: [2, 0, 1])
    assert_equal [4, 2, 3], y.shape
    
    # Test multiple permutations
    z = MLX.transpose(y, axes: [1, 0, 2])
    assert_equal [2, 4, 3], z.shape
    
    # Test that transposing twice with same axes gives original
    x = MLX.array([[1, 2], [3, 4]])
    y = MLX.transpose(MLX.transpose(x))
    assert MLX.array_equal(x, y)
  end
  
  def test_concatenate
    # Test concatenation along axis 0
    x = MLX.array([[1, 2], [3, 4]])
    y = MLX.array([[5, 6]])
    z = MLX.concatenate([x, y])
    assert_equal [3, 2], z.shape
    assert MLX.array_equal(z, MLX.array([[1, 2], [3, 4], [5, 6]]))
    
    # Test concatenation along axis 1
    x = MLX.array([[1, 2], [3, 4]])
    y = MLX.array([[5], [6]])
    z = MLX.concatenate([x, y], axis: 1)
    assert_equal [2, 3], z.shape
    assert MLX.array_equal(z, MLX.array([[1, 2, 5], [3, 4, 6]]))
    
    # Test with multiple arrays
    x = MLX.array([1, 2])
    y = MLX.array([3, 4])
    z = MLX.array([5, 6])
    result = MLX.concatenate([x, y, z])
    assert_equal [6], result.shape
    assert MLX.array_equal(result, MLX.array([1, 2, 3, 4, 5, 6]))
    
    # Test error case: inconsistent shapes
    assert_raises(ValueError) do
      MLX.concatenate([MLX.array([1, 2]), MLX.array([[3, 4]])])
    end
  end
  
  def test_stack
    # Test stack along axis 0
    x = MLX.array([1, 2, 3])
    y = MLX.array([4, 5, 6])
    z = MLX.stack([x, y])
    assert_equal [2, 3], z.shape
    assert MLX.array_equal(z, MLX.array([[1, 2, 3], [4, 5, 6]]))
    
    # Test stack along axis 1
    z = MLX.stack([x, y], axis: 1)
    assert_equal [3, 2], z.shape
    assert MLX.array_equal(z, MLX.array([[1, 4], [2, 5], [3, 6]]))
    
    # Test with multiple arrays
    x = MLX.array([1, 2])
    y = MLX.array([3, 4])
    z = MLX.array([5, 6])
    result = MLX.stack([x, y, z])
    assert_equal [3, 2], result.shape
    
    # Test error case: inconsistent shapes
    assert_raises(ValueError) do
      MLX.stack([MLX.array([1, 2]), MLX.array([3, 4, 5])])
    end
  end
  
  def test_split
    # Test split into equal parts
    x = MLX.arange(9)
    splits = MLX.split(x, 3)
    assert_equal 3, splits.length
    assert_equal [3], splits[0].shape
    assert MLX.array_equal(splits[0], MLX.array([0, 1, 2]))
    assert MLX.array_equal(splits[1], MLX.array([3, 4, 5]))
    assert MLX.array_equal(splits[2], MLX.array([6, 7, 8]))
    
    # Test split with sections
    x = MLX.arange(8)
    splits = MLX.split(x, [2, 5])
    assert_equal 3, splits.length
    assert_equal [2], splits[0].shape
    assert_equal [3], splits[1].shape
    assert_equal [3], splits[2].shape
    assert MLX.array_equal(splits[0], MLX.array([0, 1]))
    assert MLX.array_equal(splits[1], MLX.array([2, 3, 4]))
    assert MLX.array_equal(splits[2], MLX.array([5, 6, 7]))
    
    # Test split along specific axis
    x = MLX.reshape(MLX.arange(12), [4, 3])
    splits = MLX.split(x, 2, axis: 0)
    assert_equal 2, splits.length
    assert_equal [2, 3], splits[0].shape
    assert MLX.array_equal(splits[0], MLX.array([[0, 1, 2], [3, 4, 5]]))
    assert MLX.array_equal(splits[1], MLX.array([[6, 7, 8], [9, 10, 11]]))
  end
  
  def test_squeeze
    # Test basic squeeze
    x = MLX.array([[[1], [2], [3]]])
    y = MLX.squeeze(x)
    assert_equal [3], y.shape
    assert MLX.array_equal(y, MLX.array([1, 2, 3]))
    
    # Test squeeze with specific axis
    x = MLX.array([[[1], [2], [3]]])
    y = MLX.squeeze(x, axis: 2)
    assert_equal [1, 3], y.shape
    
    # Test squeeze on non-singleton dimension (should raise error)
    assert_raises(ValueError) do
      MLX.squeeze(MLX.array([[1, 2], [3, 4]]), axis: 0)
    end
    
    # Test on array with no singleton dimensions
    x = MLX.array([[1, 2], [3, 4]])
    y = MLX.squeeze(x)
    assert_equal x.shape, y.shape
    assert MLX.array_equal(x, y)
  end
  
  def test_expand_dims
    # Test basic expand_dims
    x = MLX.array([1, 2, 3, 4])
    y = MLX.expand_dims(x, axis: 0)
    assert_equal [1, 4], y.shape
    assert MLX.array_equal(y, MLX.array([[1, 2, 3, 4]]))
    
    # Test expand_dims with axis=1
    y = MLX.expand_dims(x, axis: 1)
    assert_equal [4, 1], y.shape
    assert MLX.array_equal(y, MLX.array([[1], [2], [3], [4]]))
    
    # Test expand_dims with 2D input
    x = MLX.array([[1, 2], [3, 4]])
    y = MLX.expand_dims(x, axis: 2)
    assert_equal [2, 2, 1], y.shape
    
    # Test expand_dims with negative axis
    y = MLX.expand_dims(x, axis: -1)
    assert_equal [2, 2, 1], y.shape
    
    # Test multiple calls
    z = MLX.expand_dims(MLX.expand_dims(x, axis: 0), axis: -1)
    assert_equal [1, 2, 2, 1], z.shape
  end
  
  def test_permute_dims
    # Test basic permute_dims (equivalent to transpose)
    x = MLX.array([[1, 2, 3], [4, 5, 6]])
    y = MLX.permute_dims(x, axes: [1, 0])
    assert_equal [3, 2], y.shape
    assert MLX.array_equal(y, MLX.array([[1, 4], [2, 5], [3, 6]]))
    
    # Test with higher dimensions
    x = MLX.reshape(MLX.arange(24), [2, 3, 4])
    y = MLX.permute_dims(x, axes: [2, 0, 1])
    assert_equal [4, 2, 3], y.shape
  end
  
  def test_flip
    # Test 1D flip
    x = MLX.array([1, 2, 3, 4])
    y = MLX.flip(x)
    assert MLX.array_equal(y, MLX.array([4, 3, 2, 1]))
    
    # Test 2D flip along axis 0
    x = MLX.array([[1, 2], [3, 4]])
    y = MLX.flip(x, axis: 0)
    assert MLX.array_equal(y, MLX.array([[3, 4], [1, 2]]))
    
    # Test 2D flip along axis 1
    y = MLX.flip(x, axis: 1)
    assert MLX.array_equal(y, MLX.array([[2, 1], [4, 3]]))
    
    # Test higher dimensions
    x = MLX.reshape(MLX.arange(8), [2, 2, 2])
    y = MLX.flip(x, axis: 0)
    assert MLX.array_equal(y[0], x[1])
    assert MLX.array_equal(y[1], x[0])
  end
  
  def test_broadcast_to
    # Test basic broadcasting
    x = MLX.array([1, 2, 3])
    y = MLX.broadcast_to(x, [3, 3])
    assert_equal [3, 3], y.shape
    assert MLX.array_equal(y, MLX.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]))
    
    # Test with scalar
    x = MLX.array(5)
    y = MLX.broadcast_to(x, [2, 3])
    assert_equal [2, 3], y.shape
    assert MLX.array_equal(y, MLX.full([2, 3], 5))
    
    # Test with higher dimensions
    x = MLX.array([[1, 2]])
    y = MLX.broadcast_to(x, [3, 2])
    assert_equal [3, 2], y.shape
    assert MLX.array_equal(y, MLX.array([[1, 2], [1, 2], [1, 2]]))
  end
  
  def test_tile
    # Test basic tiling
    x = MLX.array([1, 2, 3])
    y = MLX.tile(x, [2])
    assert_equal [6], y.shape
    assert MLX.array_equal(y, MLX.array([1, 2, 3, 1, 2, 3]))
    
    # Test 2D tiling
    x = MLX.array([[1, 2], [3, 4]])
    y = MLX.tile(x, [2, 1])
    assert_equal [4, 2], y.shape
    assert MLX.array_equal(y, MLX.array([[1, 2], [3, 4], [1, 2], [3, 4]]))
    
    # Test tiling in both dimensions
    y = MLX.tile(x, [2, 2])
    assert_equal [4, 4], y.shape
    assert MLX.array_equal(y, MLX.array([
      [1, 2, 1, 2],
      [3, 4, 3, 4],
      [1, 2, 1, 2],
      [3, 4, 3, 4]
    ]))
    
    # Test with scalar
    x = MLX.array(5)
    y = MLX.tile(x, [3])
    assert_equal [3], y.shape
    assert MLX.array_equal(y, MLX.array([5, 5, 5]))
  end
  
  def test_pad
    # Test basic padding
    x = MLX.array([[1, 2], [3, 4]])
    y = MLX.pad(x, [[1, 1], [1, 1]])
    assert_equal [4, 4], y.shape
    assert MLX.array_equal(y, MLX.array([
      [0, 0, 0, 0],
      [0, 1, 2, 0],
      [0, 3, 4, 0],
      [0, 0, 0, 0]
    ]))
    
    # Test asymmetric padding
    y = MLX.pad(x, [[1, 0], [0, 1]])
    assert_equal [3, 3], y.shape
    assert MLX.array_equal(y, MLX.array([
      [0, 0, 0],
      [1, 2, 0],
      [3, 4, 0]
    ]))
    
    # Test with custom value
    y = MLX.pad(x, [[1, 1], [1, 1]], constant_value: 9)
    assert_equal [4, 4], y.shape
    assert MLX.array_equal(y, MLX.array([
      [9, 9, 9, 9],
      [9, 1, 2, 9],
      [9, 3, 4, 9],
      [9, 9, 9, 9]
    ]))
    
    # Test 1D padding
    x = MLX.array([1, 2, 3])
    y = MLX.pad(x, [[2, 2]])
    assert_equal [7], y.shape
    assert MLX.array_equal(y, MLX.array([0, 0, 1, 2, 3, 0, 0]))
  end
  
  def test_swapaxes
    # Test basic swapaxes
    x = MLX.array([[1, 2, 3], [4, 5, 6]])
    y = MLX.swapaxes(x, 0, 1)
    assert_equal [3, 2], y.shape
    assert MLX.array_equal(y, MLX.array([[1, 4], [2, 5], [3, 6]]))
    
    # Test with higher dimensions
    x = MLX.reshape(MLX.arange(24), [2, 3, 4])
    y = MLX.swapaxes(x, 0, 2)
    assert_equal [4, 3, 2], y.shape
    
    # Test that swapping twice gives original
    z = MLX.swapaxes(MLX.swapaxes(x, 0, 1), 0, 1)
    assert MLX.array_equal(x, z)
  end
  
  def test_moveaxis
    # Test basic moveaxis
    x = MLX.reshape(MLX.arange(24), [2, 3, 4])
    y = MLX.moveaxis(x, 0, 2)
    assert_equal [3, 4, 2], y.shape
    
    # Test with multiple axes
    y = MLX.moveaxis(x, [0, 1], [2, 0])
    assert_equal [3, 2, 4], y.shape
    
    # Test moving to the same position
    y = MLX.moveaxis(x, 0, 0)
    assert MLX.array_equal(x, y)
  end
  
  def test_diag
    # Test 1D to 2D diagonal
    x = MLX.array([1, 2, 3])
    y = MLX.diag(x)
    assert_equal [3, 3], y.shape
    assert MLX.array_equal(y, MLX.array([
      [1, 0, 0],
      [0, 2, 0],
      [0, 0, 3]
    ]))
    
    # Test with offset
    y = MLX.diag(x, k: 1)
    assert_equal [4, 4], y.shape
    assert MLX.array_equal(y, MLX.array([
      [0, 1, 0, 0],
      [0, 0, 2, 0],
      [0, 0, 0, 3],
      [0, 0, 0, 0]
    ]))
    
    # Test 2D to 1D diagonal extraction
    x = MLX.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = MLX.diag(x)
    assert_equal [3], y.shape
    assert MLX.array_equal(y, MLX.array([1, 5, 9]))
    
    # Test with offset
    y = MLX.diag(x, k: -1)
    assert_equal [2], y.shape
    assert MLX.array_equal(y, MLX.array([4, 8]))
  end
end 