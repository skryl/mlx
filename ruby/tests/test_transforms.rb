require_relative 'mlx_test_case'

class TestTransforms < MLXTestCase
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
    
    # Test inversion property
    z = MLX.permute_dims(y, axes: [1, 2, 0])
    assert MLX.array_equal(x, z)
    
    # Test with invalid axes (should raise error)
    assert_raises(ValueError) do
      MLX.permute_dims(x, axes: [0, 1])  # Missing axis
    end
    
    assert_raises(ValueError) do
      MLX.permute_dims(x, axes: [0, 1, 1])  # Repeated axis
    end
  end
  
  def test_flip
    # Test 1D flip
    x = MLX.array([1, 2, 3, 4])
    y = MLX.flip(x)
    assert_equal x.shape, y.shape
    assert MLX.array_equal(y, MLX.array([4, 3, 2, 1]))
    
    # Test 2D flip along axis 0
    x = MLX.array([[1, 2], [3, 4]])
    y = MLX.flip(x, axis: 0)
    assert_equal x.shape, y.shape
    assert MLX.array_equal(y, MLX.array([[3, 4], [1, 2]]))
    
    # Test 2D flip along axis 1
    y = MLX.flip(x, axis: 1)
    assert_equal x.shape, y.shape
    assert MLX.array_equal(y, MLX.array([[2, 1], [4, 3]]))
    
    # Test 3D flip along multiple axes
    x = MLX.reshape(MLX.arange(8), [2, 2, 2])
    y = MLX.flip(x, axis: [0, 2])
    assert_equal x.shape, y.shape
    expected = MLX.array([[[5, 4], [7, 6]], [[1, 0], [3, 2]]])
    assert MLX.array_equal(y, expected)
  end
  
  def test_broadcast_to
    # Test basic broadcasting
    x = MLX.array([1, 2, 3])
    y = MLX.broadcast_to(x, [3, 3])
    assert_equal [3, 3], y.shape
    expected = MLX.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    assert MLX.array_equal(y, expected)
    
    # Test broadcasting scalar
    x = MLX.array(5)
    y = MLX.broadcast_to(x, [2, 2])
    assert_equal [2, 2], y.shape
    assert MLX.all(y == 5).item
    
    # Test broadcasting to same shape
    x = MLX.array([[1, 2], [3, 4]])
    y = MLX.broadcast_to(x, [2, 2])
    assert MLX.array_equal(x, y)
    
    # Test broadcasting to incompatible shape (should raise error)
    assert_raises(ValueError) do
      MLX.broadcast_to(MLX.array([1, 2, 3]), [2, 2])
    end
  end
  
  def test_tile
    # Test basic tiling
    x = MLX.array([1, 2, 3])
    y = MLX.tile(x, [2])
    assert_equal [6], y.shape
    assert MLX.array_equal(y, MLX.array([1, 2, 3, 1, 2, 3]))
    
    # Test tiling 2D array
    x = MLX.array([[1, 2], [3, 4]])
    y = MLX.tile(x, [2, 2])
    assert_equal [4, 4], y.shape
    expected = MLX.array([
      [1, 2, 1, 2],
      [3, 4, 3, 4],
      [1, 2, 1, 2],
      [3, 4, 3, 4]
    ])
    assert MLX.array_equal(y, expected)
    
    # Test tiling with 1s (should return copy of original)
    y = MLX.tile(x, [1, 1])
    assert_equal x.shape, y.shape
    assert MLX.array_equal(x, y)
    
    # Test tiling with scalar
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
    expected = MLX.array([
      [0, 0, 0, 0],
      [0, 1, 2, 0],
      [0, 3, 4, 0],
      [0, 0, 0, 0]
    ])
    assert MLX.array_equal(y, expected)
    
    # Test padding with custom value
    y = MLX.pad(x, [[1, 1], [1, 1]], value: 9)
    expected = MLX.array([
      [9, 9, 9, 9],
      [9, 1, 2, 9],
      [9, 3, 4, 9],
      [9, 9, 9, 9]
    ])
    assert MLX.array_equal(y, expected)
    
    # Test asymmetric padding
    y = MLX.pad(x, [[0, 1], [1, 0]])
    assert_equal [3, 3], y.shape
    expected = MLX.array([
      [0, 1, 2],
      [0, 3, 4],
      [0, 0, 0]
    ])
    assert MLX.array_equal(y, expected)
    
    # Test with different edge padding modes (if supported)
    # For example, "reflect", "symmetric", "edge" modes
    modes = ["constant", "reflect", "symmetric", "edge"]
    modes.each do |mode|
      begin
        y = MLX.pad(x, [[1, 1], [1, 1]], mode: mode)
        assert_equal [4, 4], y.shape
      rescue NotImplementedError
        # Skip if mode not implemented
        next
      end
    end
  end
  
  def test_swapaxes
    # Test basic swapaxes
    x = MLX.array([[1, 2, 3], [4, 5, 6]])
    y = MLX.swapaxes(x, 0, 1)
    assert_equal [3, 2], y.shape
    assert MLX.array_equal(y, MLX.array([[1, 4], [2, 5], [3, 6]]))
    
    # Test swapaxes with higher dimensions
    x = MLX.reshape(MLX.arange(24), [2, 3, 4])
    y = MLX.swapaxes(x, 0, 2)
    assert_equal [4, 3, 2], y.shape
    
    # Test swapaxes with negative indices
    y = MLX.swapaxes(x, -3, -1)
    assert_equal [4, 3, 2], y.shape
    
    # Test swapping same axis (should return copy)
    y = MLX.swapaxes(x, 1, 1)
    assert_equal x.shape, y.shape
    assert MLX.array_equal(x, y)
  end
  
  def test_moveaxis
    # Test basic moveaxis
    x = MLX.reshape(MLX.arange(24), [2, 3, 4])
    y = MLX.moveaxis(x, 0, 2)
    assert_equal [3, 4, 2], y.shape
    
    # Test moveaxis with negative indices
    y = MLX.moveaxis(x, -3, -1)
    assert_equal [3, 4, 2], y.shape
    
    # Test moving axis to same position (should return copy)
    y = MLX.moveaxis(x, 1, 1)
    assert_equal x.shape, y.shape
    assert MLX.array_equal(x, y)
    
    # Test multiple axis movement
    y = MLX.moveaxis(x, [0, 1], [2, 0])
    assert_equal [3, 2, 4], y.shape
  end
  
  def test_diag
    # Test 1D to 2D diagonal
    x = MLX.array([1, 2, 3])
    y = MLX.diag(x)
    assert_equal [3, 3], y.shape
    expected = MLX.array([
      [1, 0, 0],
      [0, 2, 0],
      [0, 0, 3]
    ])
    assert MLX.array_equal(y, expected)
    
    # Test 2D to 1D diagonal extraction
    x = MLX.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = MLX.diag(x)
    assert_equal [3], y.shape
    assert MLX.array_equal(y, MLX.array([1, 5, 9]))
    
    # Test with offset
    x = MLX.array([1, 2, 3])
    y = MLX.diag(x, k: 1)
    assert_equal [4, 4], y.shape
    expected = MLX.array([
      [0, 1, 0, 0],
      [0, 0, 2, 0],
      [0, 0, 0, 3],
      [0, 0, 0, 0]
    ])
    assert MLX.array_equal(y, expected)
    
    # Test with negative offset
    y = MLX.diag(x, k: -1)
    assert_equal [4, 4], y.shape
    expected = MLX.array([
      [0, 0, 0, 0],
      [1, 0, 0, 0],
      [0, 2, 0, 0],
      [0, 0, 3, 0]
    ])
    assert MLX.array_equal(y, expected)
  end
end 