require_relative 'mlx_test_case'

class TestArray < MLXTestCase
  def test_array_creation
    # Test creating an array from a Ruby array
    arr = MLX.array([1, 2, 3, 4, 5])
    assert_equal 5, arr.size
    assert_equal [5], arr.shape
    assert_equal 1, arr.ndim
    
    # Test creating an array with specific dtype
    arr_int32 = MLX.array([1, 2, 3], dtype: MLX::Core::INT32)
    assert_equal MLX::Core::INT32, arr_int32.dtype
    
    # Test from nested arrays
    nested = MLX.array([[1, 2, 3], [4, 5, 6]])
    assert_equal [2, 3], nested.shape
    assert_equal 2, nested.ndim
    
    # Test creation with zeros, ones, full
    zeros = MLX.zeros([2, 3])
    assert_equal [2, 3], zeros.shape
    assert_equal MLX::Core::FLOAT32, zeros.dtype
    
    ones = MLX.ones([3, 2], dtype: MLX::Core::INT32)
    assert_equal [3, 2], ones.shape
    assert_equal MLX::Core::INT32, ones.dtype
    
    full = MLX.full([2, 2], 7, dtype: MLX::Core::FLOAT32)
    assert_equal [2, 2], full.shape
    assert_equal MLX::Core::FLOAT32, full.dtype
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
    assert a.all((a == b)).item
    refute a.all((a == c)).item
    assert a.any((a != c)).item
    refute a.any((a != b)).item
    
    # Test equality with scalar
    assert a.any((a == 1)).item
    refute a.all((a == 1)).item
    assert a.all((a != 5)).item
  end
  
  def test_array_comparison
    a = MLX.array([1, 2, 3])
    b = MLX.array([0, 2, 4])
    
    # Test comparison operators
    assert a.any((a > b)).item
    assert a.any((a < b)).item
    assert a.any((a >= b)).item
    assert a.any((a <= b)).item
    
    # With scalars
    assert a.all((a > 0)).item
    assert a.all((a < 10)).item
    assert a.any((a == 2)).item
    assert a.all((a >= 1)).item
    assert a.all((a <= 3)).item
  end
  
  def test_array_reshaping
    a = MLX.array([1, 2, 3, 4, 5, 6])
    
    # Test reshape
    b = a.reshape([2, 3])
    assert_equal [2, 3], b.shape
    
    # Test transpose
    c = a.reshape([3, 2])
    d = c.transpose
    assert_equal [2, 3], d.shape
    
    # Test flatten
    e = a.reshape([2, 3])
    f = e.flatten
    assert_equal [6], f.shape
  end
  
  def test_array_indexing
    a = MLX.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    # Test basic indexing - adjust based on your actual array indexing implementation
    assert_equal 1, a[[0, 0]].item
    assert_equal 5, a[[1, 1]].item
    assert_equal 9, a[[2, 2]].item
    
    # Test slicing
    row = a[1]
    assert_array_equal(row, [4, 5, 6])
    
    # Test column access - using proper indexing API for MLX
    # Instead of Python-style a[:, 1]
    col = MLX::Core::Indexing.index_select(a, dim: 1, index: MLX.array(1))
    assert_array_equal(col, [[2], [5], [8]])
    
    # Test advanced indexing
    # Instead of a[[0, 2], [0, 2]], use proper MLX API
    selected = MLX::Core::Indexing.take(a, MLX.array([0, 8]))  # Indices 0 and 8 in flattened array
    assert_array_equal(selected, [1, 9])
  end
  
  def test_array_math_functions
    a = MLX.array([1, 2, 3, 4])
    
    # Test abs
    assert_array_equal(MLX.array([-1, -2, 3, -4]).abs, [1, 2, 3, 4])
    
    # Test sqrt
    assert_array_equal(a.sqrt, [1, 1.4142, 1.7321, 2], atol: 1e-4)
    
    # Test exp
    exp_result = MLX.array([0, 1, 2]).exp
    assert_array_equal(exp_result, [1, 2.7183, 7.3891], atol: 1e-4)
    
    # Test log
    log_result = MLX.array([1, 2, 10]).log
    assert_array_equal(log_result, [0, 0.6931, 2.3026], atol: 1e-4)
    
    # Test sin, cos
    sin_result = MLX.array([0, Math::PI/2, Math::PI]).sin
    assert_array_equal(sin_result, [0, 1, 0], atol: 1e-4)
    
    cos_result = MLX.array([0, Math::PI/2, Math::PI]).cos
    assert_array_equal(cos_result, [1, 0, -1], atol: 1e-4)
  end
  
  def test_array_reduction
    a = MLX.array([[1, 2, 3], [4, 5, 6]])
    
    # Test sum (adjust expected values based on actual implementation)
    assert_equal 21, a.sum.item
    assert_array_equal(a.sum(axis: 0), [5, 7, 9])
    assert_array_equal(a.sum(axis: 1), [6, 15])
    
    # Test mean
    assert_equal 3.5, a.mean.item
    assert_array_equal(a.mean(axis: 0), [2.5, 3.5, 4.5])
    assert_array_equal(a.mean(axis: 1), [2, 5])
    
    # Test min/max
    assert_equal 1, a.min.item
    assert_equal 6, a.max.item
    assert_array_equal(a.min(axis: 0), [1, 2, 3])
    assert_array_equal(a.max(axis: 1), [3, 6])
    
    # Test argmin/argmax
    assert_equal 0, a.argmin.item
    assert_equal 5, a.argmax.item
    assert_array_equal(a.argmin(axis: 0), [0, 0, 0])
    assert_array_equal(a.argmax(axis: 1), [2, 2])
  end
  
  def test_array_logical
    a = MLX.array([true, false, true])
    b = MLX.array([true, true, false])
    
    # Test logical operations
    assert_array_equal(a & b, [true, false, false])
    assert_array_equal(a | b, [true, true, true])
    assert_array_equal(~a, [false, true, false])
    assert_array_equal(a ^ b, [false, true, true])
    
    # Test all and any
    assert a.any.item
    refute a.all.item
    assert b.any.item
    refute b.all.item
  end
  
  def test_array_conversion
    # Test dtype conversion
    a = MLX.array([1, 2, 3], dtype: MLX::Core::FLOAT32)
    
    b = a.astype(MLX::Core::INT32)
    assert_equal MLX::Core::INT32, b.dtype
    assert_array_equal(b, [1, 2, 3])
    
    c = a.astype(MLX::Core::FLOAT16)
    assert_equal MLX::Core::FLOAT16, c.dtype
    assert_array_equal(c, [1, 2, 3])
    
    # Test to_list - if implemented
    if a.respond_to?(:tolist)
      assert_equal [1.0, 2.0, 3.0], a.tolist
    end
  end
  
  def test_array_device
    # Only test if metal is available
    if MLX::Core::Metal.metal_is_available
      a = MLX.array([1, 2, 3])
      
      # Test device transformation - update with actual method names
      cpu_arr = MLX::Device.to_cpu(a)  
      gpu_arr = MLX::Device.to_gpu(a)
      
      assert_array_equal(cpu_arr, a)
      assert_array_equal(gpu_arr, a)
    end
  end

  # Array transform tests
  def test_reshape
    # Test reshape on 1D array
    x = MLX.arange(0, 6)
    y = x.reshape([2, 3])
    assert_equal [2, 3], y.shape
    assert_array_equal(y, [[0, 1, 2], [3, 4, 5]])
    
    # Test reshape on 2D array
    x = MLX.array([[1, 2], [3, 4], [5, 6]])
    y = x.reshape([2, 3])
    assert_equal [2, 3], y.shape
    assert_array_equal(y, [[1, 2, 3], [4, 5, 6]])
    
    # Test with -1 dimension
    x = MLX.arange(0, 12)
    y = x.reshape([-1, 4])
    assert_equal [3, 4], y.shape
    
    # Test with multiple -1 dimensions (should raise error)
    assert_raises(ArgumentError) do
      x.reshape([-1, -1])
    end
    
    # Test preserving dtype
    x = MLX.array([1.0, 2.0, 3.0, 4.0])
    y = x.reshape([2, 2])
    assert_equal x.dtype, y.dtype
  end
  
  def test_transpose
    # Test basic transpose
    x = MLX.array([[1, 2, 3], [4, 5, 6]])
    y = x.transpose
    assert_equal [3, 2], y.shape
    assert_array_equal(y, [[1, 4], [2, 5], [3, 6]])
    
    # Test with specific axes
    x = MLX.reshape(MLX.arange(0, 24), [2, 3, 4])
    y = x.transpose(axes: [2, 0, 1])
    assert_equal [4, 2, 3], y.shape
    
    # Test multiple permutations
    z = y.transpose(axes: [1, 0, 2])
    assert_equal [2, 4, 3], z.shape
    
    # Test that transposing twice with same axes gives original
    x = MLX.array([[1, 2], [3, 4]])
    y = x.transpose.transpose
    assert_array_equal(x, y)
  end
  
  def test_concatenate
    # Test concatenation along axis 0
    x = MLX.array([[1, 2], [3, 4]])
    y = MLX.array([[5, 6]])
    z = MLX.concatenate([x, y])
    assert_equal [3, 2], z.shape
    assert_array_equal(z, [[1, 2], [3, 4], [5, 6]])
    
    # Test concatenation along axis 1
    x = MLX.array([[1, 2], [3, 4]])
    y = MLX.array([[5], [6]])
    z = MLX.concatenate([x, y], axis: 1)
    assert_equal [2, 3], z.shape
    assert_array_equal(z, [[1, 2, 5], [3, 4, 6]])
    
    # Test with multiple arrays
    x = MLX.array([1, 2])
    y = MLX.array([3, 4])
    z = MLX.array([5, 6])
    result = MLX.concatenate([x, y, z])
    assert_equal [6], result.shape
    assert_array_equal(result, [1, 2, 3, 4, 5, 6])
    
    # Test error case: inconsistent shapes
    assert_raises(ArgumentError) do
      MLX.concatenate([MLX.array([1, 2]), MLX.array([[3, 4]])])
    end
  end
  
  def test_stack
    # Test stack along axis 0
    x = MLX.array([1, 2, 3])
    y = MLX.array([4, 5, 6])
    z = MLX.stack([x, y])
    assert_equal [2, 3], z.shape
    assert_array_equal(z, [[1, 2, 3], [4, 5, 6]])
    
    # Test stack along axis 1
    z = MLX.stack([x, y], axis: 1)
    assert_equal [3, 2], z.shape
    assert_array_equal(z, [[1, 4], [2, 5], [3, 6]])
    
    # Test with multiple arrays
    x = MLX.array([1, 2])
    y = MLX.array([3, 4])
    z = MLX.array([5, 6])
    result = MLX.stack([x, y, z])
    assert_equal [3, 2], result.shape
    
    # Test error case: inconsistent shapes
    assert_raises(ArgumentError) do
      MLX.stack([MLX.array([1, 2]), MLX.array([3, 4, 5])])
    end
  end
  
  def test_split
    # Test split into equal parts
    x = MLX.arange(0, 9)
    splits = MLX.split(x, 3)
    assert_equal 3, splits.length
    assert_equal [3], splits[0].shape
    assert_array_equal(splits[0], [0, 1, 2])
    assert_array_equal(splits[1], [3, 4, 5])
    assert_array_equal(splits[2], [6, 7, 8])
    
    # Test split with sections
    x = MLX.arange(0, 8)
    splits = MLX.split(x, [2, 5])
    assert_equal 3, splits.length
    assert_equal [2], splits[0].shape
    assert_equal [3], splits[1].shape
    assert_equal [3], splits[2].shape
    assert_array_equal(splits[0], [0, 1])
    assert_array_equal(splits[1], [2, 3, 4])
    assert_array_equal(splits[2], [5, 6, 7])
    
    # Test split along specific axis
    x = MLX.reshape(MLX.arange(0, 12), [4, 3])
    splits = MLX.split(x, 2, axis: 0)
    assert_equal 2, splits.length
    assert_equal [2, 3], splits[0].shape
    assert_array_equal(splits[0], [[0, 1, 2], [3, 4, 5]])
    assert_array_equal(splits[1], [[6, 7, 8], [9, 10, 11]])
  end
  
  def test_squeeze
    skip "Implementing in a future version"
    # Test basic squeeze
    x = MLX.array([[[1], [2], [3]]])
    y = x.squeeze
    assert_equal [3], y.shape
    assert_array_equal(y, [1, 2, 3])
    
    # Test squeeze with specific axis
    x = MLX.array([[[1], [2], [3]]])
    y = x.squeeze(axes: 2)
    assert_equal [1, 3], y.shape
    
    # Test squeeze on non-singleton dimension (should raise error)
    assert_raises(ArgumentError) do
      MLX.array([[1, 2], [3, 4]]).squeeze(axes: 0)
    end
    
    # Test on array with no singleton dimensions
    x = MLX.array([[1, 2], [3, 4]])
    y = x.squeeze
    assert_equal x.shape, y.shape
    assert_array_equal(x, y)
  end
  
  def test_expand_dims
    skip "Implementing in a future version"
    # Test basic expand_dims
    x = MLX.array([1, 2, 3, 4])
    y = x.expand_dims(axis: 0)
    assert_equal [1, 4], y.shape
    assert_array_equal(y, [[1, 2, 3, 4]])
    
    # Test expand_dims with axis=1
    y = x.expand_dims(axis: 1)
    assert_equal [4, 1], y.shape
    assert_array_equal(y, [[1], [2], [3], [4]])
    
    # Test expand_dims with 2D input
    x = MLX.array([[1, 2], [3, 4]])
    y = x.expand_dims(axis: 2)
    assert_equal [2, 2, 1], y.shape
    
    # Test expand_dims with negative axis
    y = x.expand_dims(axis: -1)
    assert_equal [2, 2, 1], y.shape
    
    # Test multiple calls
    z = x.expand_dims(axis: 0).expand_dims(axis: -1)
    assert_equal [1, 2, 2, 1], z.shape
  end
  
  def test_permute_dims
    skip "Implementing in a future version"
    # Test basic permute_dims (equivalent to transpose)
    x = MLX.array([[1, 2, 3], [4, 5, 6]])
    y = MLX.transpose(x, axes: [1, 0])
    assert_equal [3, 2], y.shape
    assert_array_equal(y, [[1, 4], [2, 5], [3, 6]])
    
    # Test with higher dimensions
    x = MLX.reshape(MLX.arange(0, 24), [2, 3, 4])
    y = MLX.transpose(x, axes: [2, 0, 1])
    assert_equal [4, 2, 3], y.shape
  end
  
  def test_flip
    skip "Implementing in a future version"
    # Test 1D flip
    x = MLX.array([1, 2, 3, 4])
    y = x.flip
    assert_array_equal(y, [4, 3, 2, 1])
    
    # Test 2D flip along axis 0
    x = MLX.array([[1, 2], [3, 4]])
    y = x.flip(axis: 0)
    assert_array_equal(y, [[3, 4], [1, 2]])
    
    # Test 2D flip along axis 1
    y = x.flip(axis: 1)
    assert_array_equal(y, [[2, 1], [4, 3]])
    
    # Test higher dimensions
    x = MLX.reshape(MLX.arange(0, 8), [2, 2, 2])
    y = x.flip(axis: 0)
    assert_array_equal(y[0], x[1])
    assert_array_equal(y[1], x[0])
  end
  
  def test_broadcast_to
    skip "Implementing in a future version"
    # Test basic broadcasting
    x = MLX.array([1, 2, 3])
    y = MLX.broadcast_to(x, [3, 3])
    assert_equal [3, 3], y.shape
    assert_array_equal(y, [[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    
    # Test with scalar
    x = MLX.array(5)
    y = MLX.broadcast_to(x, [2, 3])
    assert_equal [2, 3], y.shape
    assert_array_equal(y, MLX.full([2, 3], 5))
    
    # Test with higher dimensions
    x = MLX.array([[1, 2]])
    y = MLX.broadcast_to(x, [3, 2])
    assert_equal [3, 2], y.shape
    assert_array_equal(y, [[1, 2], [1, 2], [1, 2]])
  end
  
  def test_tile
    skip "Implementing in a future version"
    # Test basic tiling
    x = MLX.array([1, 2, 3])
    y = MLX.tile(x, [2])
    assert_equal [6], y.shape
    assert_array_equal(y, [1, 2, 3, 1, 2, 3])
    
    # Test 2D tiling
    x = MLX.array([[1, 2], [3, 4]])
    y = MLX.tile(x, [2, 1])
    assert_equal [4, 2], y.shape
    assert_array_equal(y, [[1, 2], [3, 4], [1, 2], [3, 4]])
    
    # Test tiling in both dimensions
    y = MLX.tile(x, [2, 2])
    assert_equal [4, 4], y.shape
    assert_array_equal(y, [
      [1, 2, 1, 2],
      [3, 4, 3, 4],
      [1, 2, 1, 2],
      [3, 4, 3, 4]
    ])
    
    # Test with scalar
    x = MLX.array(5)
    y = MLX.tile(x, [3])
    assert_equal [3], y.shape
    assert_array_equal(y, [5, 5, 5])
  end
  
  def test_pad
    skip "Implementing in a future version"
    # Test basic padding
    x = MLX.array([[1, 2], [3, 4]])
    y = MLX.pad(x, [[1, 1], [1, 1]])
    assert_equal [4, 4], y.shape
    assert_array_equal(y, [
      [0, 0, 0, 0],
      [0, 1, 2, 0],
      [0, 3, 4, 0],
      [0, 0, 0, 0]
    ])
    
    # Test asymmetric padding
    y = MLX.pad(x, [[1, 0], [0, 1]])
    assert_equal [3, 3], y.shape
    assert_array_equal(y, [
      [0, 0, 0],
      [1, 2, 0],
      [3, 4, 0]
    ])
    
    # Test with custom value
    y = MLX.pad(x, [[1, 1], [1, 1]], constant_values: 9)
    assert_equal [4, 4], y.shape
    assert_array_equal(y, [
      [9, 9, 9, 9],
      [9, 1, 2, 9],
      [9, 3, 4, 9],
      [9, 9, 9, 9]
    ])
    
    # Test 1D padding
    x = MLX.array([1, 2, 3])
    y = MLX.pad(x, [[2, 2]])
    assert_equal [7], y.shape
    assert_array_equal(y, [0, 0, 1, 2, 3, 0, 0])
  end
  
  def test_swapaxes
    skip "Implementing in a future version"
    # Test basic swapaxes
    x = MLX.array([[1, 2, 3], [4, 5, 6]])
    y = x.swapaxes(0, 1)
    assert_equal [3, 2], y.shape
    assert_array_equal(y, [[1, 4], [2, 5], [3, 6]])
    
    # Test with higher dimensions
    x = MLX.reshape(MLX.arange(0, 24), [2, 3, 4])
    y = x.swapaxes(0, 2)
    assert_equal [4, 3, 2], y.shape
    
    # Test that swapping twice gives original
    z = x.swapaxes(0, 1).swapaxes(0, 1)
    assert_array_equal(x, z)
  end
  
  def test_moveaxis
    skip "Implementing in a future version"
    # Test basic moveaxis
    x = MLX.reshape(MLX.arange(0, 24), [2, 3, 4])
    y = x.moveaxis(0, 2)
    assert_equal [3, 4, 2], y.shape
    
    # Test with multiple axes
    y = MLX.moveaxis(x, [0, 1], [2, 0])
    assert_equal [3, 2, 4], y.shape
    
    # Test moving to the same position
    y = x.moveaxis(0, 0)
    assert_array_equal(x, y)
  end
  
  def test_diag
    skip "Implementing in a future version"
    # Test 1D to 2D diagonal
    x = MLX.array([1, 2, 3])
    y = x.diag
    assert_equal [3, 3], y.shape
    assert_array_equal(y, [
      [1, 0, 0],
      [0, 2, 0],
      [0, 0, 3]
    ])
    
    # Test with offset
    y = x.diag(k: 1)
    assert_equal [4, 4], y.shape
    assert_array_equal(y, [
      [0, 1, 0, 0],
      [0, 0, 2, 0],
      [0, 0, 0, 3],
      [0, 0, 0, 0]
    ])
    
    # Test 2D to 1D diagonal extraction
    x = MLX.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = x.diagonal
    assert_equal [3], y.shape
    assert_array_equal(y, [1, 5, 9])
    
    # Test with offset
    y = x.diagonal(offset: -1)
    assert_equal [2], y.shape
    assert_array_equal(y, [4, 8])
  end
end 