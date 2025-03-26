require_relative 'mlx_test_case'

class TestVmap < MLXTestCase
  def setup
    # Set random seed for reproducibility
    MLX.random.seed(42)
  end
  
  def test_basics
    # Can't vmap over scalars
    assert_raises(ArgumentError) do
      MLX.vmap(lambda {|x| MLX.exp(x)}).call(MLX.array(1.0))
    end
    
    # Invalid input
    assert_raises(TypeError) do
      MLX.vmap(lambda {|x| MLX.exp(x)}).call("hello")
    end
    
    # Invalid in_axes
    assert_raises(ArgumentError) do
      MLX.vmap(lambda {|x| MLX.exp(x)}, in_axes: "hello").call(MLX.array([0, 1]))
    end
    
    assert_raises(ArgumentError) do
      MLX.vmap(lambda {|x| MLX.exp(x)}, in_axes: 2).call(MLX.array([0, 1]))
    end
    
    # Invalid out_axes
    assert_raises(ArgumentError) do
      MLX.vmap(lambda {|x| MLX.exp(x)}, out_axes: "hello").call(MLX.array([0, 1]))
    end
    
    assert_raises(ArgumentError) do
      MLX.vmap(lambda {|x| MLX.exp(x)}, out_axes: 2).call(MLX.array([0, 1]))
    end
  end
  
  def test_unary
    ops = [
      "abs",
      "cos",
      "erf",
      "exp",
      "log",
      "log1p",
      "log2",
      "log10",
      "logical_not",
      "negative",
      "reciprocal",
      "rsqrt",
      "sigmoid",
      "sign",
      "sin",
      "sqrt",
      "square"
    ]
    
    ops.each do |opname|
      # Skip if operation not available in MLX Ruby
      next unless MLX.respond_to?(opname)
      
      op = MLX.method(opname)
      
      # Test on 1D array
      x = MLX.arange(5)
      y = MLX.vmap(lambda {|a| op.call(a)}).call(x)
      assert MLX.array_equal(y, op.call(x))
      
      # Test on 2D array
      x = MLX.arange(8).reshape([2, 4])
      y = MLX.vmap(lambda {|a| op.call(a)}).call(x)
      assert MLX.array_equal(y, op.call(x))
      
      # Test with different in_axes and out_axes
      y = MLX.vmap(lambda {|a| op.call(a)}, in_axes: 1, out_axes: 1).call(x)
      assert MLX.array_equal(y, op.call(x))
    end
  end
  
  def test_binary
    ops = [
      "add",
      "divide",
      "equal",
      "greater",
      "greater_equal",
      "less",
      "less_equal",
      "maximum",
      "minimum",
      "multiply",
      "power",
      "subtract",
      "logical_or",
      "logical_and"
    ]
    
    ops.each do |opname|
      # Skip if operation not available in MLX Ruby
      next unless MLX.respond_to?(opname)
      
      op = MLX.method(opname)
      
      # Test on 1D arrays
      x = MLX.random.uniform(shape: [5])
      y = MLX.random.uniform(shape: [5])
      out = MLX.vmap(lambda {|a, b| op.call(a, b)}).call(x, y)
      assert MLX.array_equal(out, op.call(x, y))
      
      # Test on 2D arrays
      x = MLX.random.uniform(shape: [2, 4])
      y = MLX.random.uniform(shape: [2, 4])
      out = MLX.vmap(lambda {|a, b| op.call(a, b)}).call(x, y)
      assert MLX.array_equal(out, op.call(x, y))
      
      # Test with specified in_axes and out_axes
      out = MLX.vmap(lambda {|a, b| op.call(a, b)}, in_axes: [0, 0], out_axes: 0).call(x, y)
      assert MLX.array_equal(out, op.call(x, y))
      
      # Test with different in_axes
      y = MLX.random.uniform(shape: [4, 2])
      out = MLX.vmap(lambda {|a, b| op.call(a, b)}, in_axes: [0, 1], out_axes: 0).call(x, y)
      assert MLX.array_equal(out, op.call(x, MLX.transpose(y)))
      
      # Test with different out_axes
      out = MLX.vmap(lambda {|a, b| op.call(a, b)}, in_axes: [0, 1], out_axes: 1).call(x, y)
      assert MLX.array_equal(out, MLX.transpose(op.call(x, MLX.transpose(y))))
    end
  end
  
  def test_tree
    # Define a function that operates on a tree structure
    my_fun = lambda do |tree|
      (tree[:a] + tree[:b][0]) * tree[:b][1]
    end
    
    # Create a tree with arrays
    tree = {
      a: MLX.random.uniform(shape: [2, 4]),
      b: [
        MLX.random.uniform(shape: [2, 4]),
        MLX.random.uniform(shape: [2, 4]),
      ]
    }
    
    # Test vmap with the tree
    out = MLX.vmap(my_fun).call(tree)
    expected = my_fun.call(tree)
    assert MLX.array_equal(out, expected)
    
    # Test with specified in_axes
    out = MLX.vmap(my_fun, in_axes: {a: 0, b: 0}).call(tree)
    assert MLX.array_equal(out, expected)
    
    out = MLX.vmap(my_fun, in_axes: {a: 0, b: [0, 0]}).call(tree)
    assert MLX.array_equal(out, expected)
    
    # Test with different in_axes
    tree = {
      a: MLX.random.uniform(shape: [2, 4]),
      b: [
        MLX.random.uniform(shape: [4, 2]),
        MLX.random.uniform(shape: [4, 2]),
      ]
    }
    
    out = MLX.vmap(my_fun, in_axes: {a: 0, b: [1, 1]}, out_axes: 0).call(tree)
    expected = (tree[:a] + MLX.transpose(tree[:b][0])) * MLX.transpose(tree[:b][1])
    assert MLX.array_equal(out, expected)
    
    # Test with a function that returns a tree
    my_fun2 = lambda do |x, y|
      {a: x + y, b: x * y}
    end
    
    x = MLX.random.uniform(shape: [2, 4])
    y = MLX.random.uniform(shape: [2, 4])
    out = MLX.vmap(my_fun2, in_axes: 0, out_axes: 0).call(x, y)
    expected = my_fun2.call(x, y)
    
    assert MLX.array_equal(out[:a], expected[:a])
    assert MLX.array_equal(out[:b], expected[:b])
    
    # Test with different out_axes for different parts of the tree
    out = MLX.vmap(my_fun2, in_axes: 0, out_axes: {a: 1, b: 0}).call(x, y)
    expected = my_fun2.call(x, y)
    
    assert MLX.array_equal(MLX.transpose(out[:a]), expected[:a])
    assert MLX.array_equal(out[:b], expected[:b])
  end
  
  def test_vmap_indexing
    x = MLX.arange(16).reshape([2, 2, 2, 2])
    inds = MLX.array([[0, 1, 0], [1, 1, 0]])
    
    # Test indexing with vmapped function
    out = MLX.vmap(lambda {|x, y| x[y]}, in_axes: [0, 0]).call(x, inds)
    expected = MLX.array([
      [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[0, 1], [2, 3]]],
      [[[12, 13], [14, 15]], [[12, 13], [14, 15]], [[8, 9], [10, 11]]]
    ])
    assert MLX.array_equal(out, expected)
    
    # Test with None for one axis
    out = MLX.vmap(lambda {|x, y| x[y]}, in_axes: [0, nil]).call(x, inds)
    expected = MLX.array([
      [
        [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[0, 1], [2, 3]]],
        [[[4, 5], [6, 7]], [[4, 5], [6, 7]], [[0, 1], [2, 3]]]
      ],
      [
        [[[8, 9], [10, 11]], [[12, 13], [14, 15]], [[8, 9], [10, 11]]],
        [[[12, 13], [14, 15]], [[12, 13], [14, 15]], [[8, 9], [10, 11]]]
      ]
    ])
    assert MLX.array_equal(out, expected)
    
    # Test with None for the other axis
    out = MLX.vmap(lambda {|x, y| x[y]}, in_axes: [nil, 0]).call(x, inds)
    expected = MLX.array([
      [
        [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
        [[[8, 9], [10, 11]], [[12, 13], [14, 15]]],
        [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
      ],
      [
        [[[8, 9], [10, 11]], [[12, 13], [14, 15]]],
        [[[8, 9], [10, 11]], [[12, 13], [14, 15]]],
        [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
      ]
    ])
    assert MLX.array_equal(out, expected)
    
    # Test with multiple indices
    inds2 = MLX.array([[0, 1, 0], [0, 1, 0]])
    out = MLX.vmap(lambda {|x, y, z| x[y, z]}, in_axes: [nil, 0, 0]).call(x, inds, inds2)
    expected = MLX.array([
      [[[0, 1], [2, 3]], [[12, 13], [14, 15]], [[0, 1], [2, 3]]],
      [[[8, 9], [10, 11]], [[12, 13], [14, 15]], [[0, 1], [2, 3]]]
    ])
    assert MLX.array_equal(out, expected)
  end
  
  def test_vmap_reduce
    a = MLX.ones([5, 5], dtype: MLX.int32)
    
    # Test sum reduction
    out = MLX.vmap(lambda {|x| MLX.sum(x)}).call(a)
    assert MLX.array_equal(out, MLX.full([5], 5))
    
    # Test with keepdims
    out = MLX.vmap(lambda {|x| MLX.sum(x, keepdims: true)}).call(a)
    assert MLX.array_equal(out, MLX.full([5, 1], 5))
    
    # Test with axis
    out = MLX.vmap(lambda {|x| MLX.sum(x, axis: 0)}).call(a)
    assert MLX.array_equal(out, MLX.full([5], 5))
    
    # Test with multiple axes
    a = MLX.ones([5, 3, 2], dtype: MLX.int32)
    out = MLX.vmap(lambda {|x| MLX.sum(x, axis: [0, 1])}).call(a)
    assert MLX.array_equal(out, MLX.full([5], 6))
    
    # Test with different in_axes
    a = MLX.ones([5, 3, 2], dtype: MLX.int32)
    out = MLX.vmap(lambda {|x| MLX.sum(x, axis: [0, 1])}, in_axes: 1).call(a)
    assert MLX.array_equal(out, MLX.full([3], 10))
    
    a = MLX.ones([5, 3, 2], dtype: MLX.int32)
    out = MLX.vmap(lambda {|x| MLX.sum(x, axis: [0, 1])}, in_axes: 2).call(a)
    assert MLX.array_equal(out, MLX.full([2], 15))
  end
  
  def test_vmap_argreduce
    a = MLX.array([[1, 2, 3], [2, 3, 1]])
    
    # Test argmin
    out = MLX.vmap(lambda {|x| MLX.argmin(x)}).call(a)
    expected = MLX.array([0, 2])
    assert MLX.array_equal(out, expected)
    
    # Test argmax
    out = MLX.vmap(lambda {|x| MLX.argmax(x)}).call(a)
    expected = MLX.array([2, 1])
    assert MLX.array_equal(out, expected)
  end
  
  def test_vmap_mean
    a = MLX.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    
    # Test mean along axis
    out = MLX.vmap(lambda {|x| MLX.mean(x)}).call(a)
    expected = MLX.array([2.0, 5.0])
    assert MLX.array_equal(out, expected)
    
    # Test with keepdims
    out = MLX.vmap(lambda {|x| MLX.mean(x, keepdims: true)}).call(a)
    expected = MLX.array([[2.0], [5.0]])
    assert MLX.array_equal(out, expected)
  end
  
  def test_vmap_matmul
    # Test matmul with vmap
    batch, m, n, k = 3, 4, 5, 6
    
    a = MLX.random.normal(shape: [batch, m, k])
    b = MLX.random.normal(shape: [batch, k, n])
    
    # Manual batch matmul
    c_expected = MLX.zeros([batch, m, n])
    (0...batch).each do |i|
      c_expected[i] = MLX.matmul(a[i], b[i])
    end
    
    # Vmap batch matmul
    batched_matmul = lambda {|a, b| MLX.matmul(a, b)}
    c_vmap = MLX.vmap(batched_matmul).call(a, b)
    
    assert MLX.allclose(c_vmap, c_expected)
    
    # Test with different in_axes
    a = MLX.random.normal(shape: [m, batch, k])
    b = MLX.random.normal(shape: [batch, k, n])
    
    c_vmap = MLX.vmap(batched_matmul, in_axes: [1, 0]).call(a, b)
    
    c_expected = MLX.zeros([batch, m, n])
    (0...batch).each do |i|
      c_expected[i] = MLX.matmul(a[:, i, :], b[i])
    end
    
    assert MLX.allclose(c_vmap, c_expected)
  end
  
  def test_vmap_concatenate
    # Test concatenation inside vmap
    cat_fun = lambda {|x, y| MLX.concatenate([x, y])}
    
    x = MLX.random.normal(shape: [3, 4])
    y = MLX.random.normal(shape: [3, 5])
    
    result = MLX.vmap(cat_fun).call(x, y)
    assert_equal [3, 9], result.shape
    
    # Test with constants
    cat_constant = lambda {|x| MLX.concatenate([x, MLX.ones([4])])}
    
    result = MLX.vmap(cat_constant).call(x)
    assert_equal [3, 8], result.shape
  end
  
  def test_vmap_gather
    # Test gather operations with vmap
    gather = lambda do |a, idx|
      MLX.take(a, idx, axis: 0)
    end
    
    a = MLX.arange(20).reshape([5, 4])
    idx = MLX.array([[0, 1], [2, 3]])
    
    # Vmap over first axis of idx
    result = MLX.vmap(gather, in_axes: [nil, 0]).call(a, idx)
    expected = MLX.array([
      [[0, 1, 2, 3], [4, 5, 6, 7]],
      [[8, 9, 10, 11], [12, 13, 14, 15]]
    ])
    
    assert MLX.array_equal(result, expected)
    
    # Vmap over both inputs
    a_batch = MLX.arange(24).reshape([2, 3, 4])
    idx_batch = MLX.array([[[0, 1], [1, 2]], [[1, 2], [0, 1]]])
    
    gather_two_idx = lambda do |a, idxa, idxb|
      MLX.take(a, idxa, axis: 0)[idxb]
    end
    
    result = MLX.vmap(gather_two_idx, in_axes: [0, 0, 0]).call(a_batch, idx_batch[:, 0, :], idx_batch[:, 1, :])
    
    assert_equal [2, 2], result.shape
  end
  
  def test_vmap_scatter
    # Test scatter operations with vmap
    scatter = lambda do |a|
      result = MLX.zeros([5, 5])
      indices = MLX.array([0, 2, 4])
      MLX.scatter(result, indices, a, axis: 0)
    end
    
    a = MLX.arange(12).reshape([3, 4])
    
    # Vmap over first axis of a
    result = MLX.vmap(scatter, in_axes: 0).call(a)
    assert_equal [4, 5, 5], result.shape
    
    # Test scatter_add
    scatter_add = lambda do |a|
      result = MLX.ones([5, 5])
      indices = MLX.array([0, 2, 4])
      MLX.scatter_add(result, indices, a, axis: 0)
    end
    
    result = MLX.vmap(scatter_add, in_axes: 0).call(a)
    assert_equal [4, 5, 5], result.shape
    
    # Test with custom indices
    scatter = lambda do |a, idx|
      result = MLX.zeros([5])
      MLX.scatter(result, idx, a)
    end
    
    a = MLX.arange(6).reshape([2, 3])
    idx = MLX.array([[0, 2, 4], [1, 3, 4]])
    
    result = MLX.vmap(scatter, in_axes: [0, 0]).call(a, idx)
    assert_equal [2, 5], result.shape
  end
  
  def test_vmap_take_along_axis
    # Test take_along_axis with vmap
    fun = lambda do |a, idx|
      MLX.take_along_axis(a, idx, axis: 1)
    end
    
    a = MLX.arange(20).reshape([5, 4])
    idx = MLX.array([[0, 2], [1, 3], [2, 0], [3, 1], [0, 2]])
    
    result = MLX.vmap(fun).call(a, idx)
    assert_equal [5, 2], result.shape
    
    # Check first row
    expected_first_row = MLX.array([0, 2])
    assert MLX.array_equal(result[0], expected_first_row)
  end
  
  def test_split_vmap
    # Test splitting operations inside vmap
    fun = lambda do |x|
      split = MLX.split(x, 2, axis: 0)
      MLX.concatenate(split, axis: 0)
    end
    
    a = MLX.arange(12).reshape([2, 6])
    result = MLX.vmap(fun, in_axes: 1).call(a)
    
    # The result should be the same as the input
    assert MLX.array_equal(result, a)
  end
  
  def test_const_func
    # Test a function that ignores some inputs
    const_func = lambda do |a, b|
      # Ignore b, just return a
      a
    end
    
    a = MLX.array([[1, 2], [3, 4]])
    b = MLX.array([[5, 6], [7, 8]])
    
    # Vmap over first axis
    result = MLX.vmap(const_func).call(a, b)
    assert MLX.array_equal(result, a)
    
    # Vmap with different in_axes
    result = MLX.vmap(const_func, in_axes: [0, 1]).call(a, b)
    assert MLX.array_equal(result, a)
  end
end 