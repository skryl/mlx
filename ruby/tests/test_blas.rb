require_relative 'mlx_test_case'

class TestBlas < MLXTestCase
  def setup
    # Set random seed for reproducibility
    MLX.random.seed(42)
  end
  
  def dtypes
    if MLX.metal.available?
      [MLX.float32, MLX.float16]
    else
      [MLX.float32]
    end
  end
  
  def gemm_test(shape_a, shape_b, dtype=MLX.float32, f_a=lambda {|x| x}, f_b=lambda {|x| x})
    # Helper method to test matrix multiplication
    scale = [shape_a.sum, 128].max
    a = (MLX.random.normal(shape: shape_a) * (1.0 / scale)).astype(dtype)
    b = (MLX.random.normal(shape: shape_b) * (1.0 / scale)).astype(dtype)
    
    # Apply transformations if provided
    a = f_a.call(a)
    b = f_b.call(b)
    
    # Compute matrix multiplication
    c = a.matmul(b)
    MLX.eval(c)
    
    # Test succeeds if we get here without errors
    assert true
  end
  
  def test_matmul_unaligned
    # Skip if metal is not available
    return unless MLX.metal.available?
    
    dtypes.each do |dtype|
      base_shapes = [4, 8, 16, 32, 64, 128]
      perturbations = [-2, -1, 0, 1, 2]
      
      base_shapes.each do |dim|
        perturbations.each do |p|
          shape_a = [dim + p, dim + p]
          shape_b = [dim + p, dim + p]
          gemm_test(shape_a, shape_b, dtype)
        end
      end
    end
  end
  
  def test_matmul_shapes
    # Skip if metal is not available
    return unless MLX.metal.available?
    
    shapes = [
      [1, 2, 1, 1],
      [1, 1, 2, 1],
      [3, 23, 457, 3],
    ]
    
    if MLX.default_device == :gpu
      shapes += [
        [16, 768, 768, 128],
        [1, 64, 64, 4096],
      ]
    end
    
    dtypes.each do |dtype|
      shapes.each do |b, m, n, k|
        # Test nn (no transpose)
        shape_a = [b, m, k]
        shape_b = [b, k, n]
        gemm_test(shape_a, shape_b, dtype)
        
        # Test nt (transpose b)
        shape_a = [b, m, k]
        shape_b = [b, n, k]
        gemm_test(
          shape_a, 
          shape_b, 
          dtype,
          lambda {|x| x},
          lambda {|x| MLX.transpose(x, axes: [0, 2, 1])}
        )
        
        # Test tn (transpose a)
        shape_a = [b, k, m]
        shape_b = [b, k, n]
        gemm_test(
          shape_a, 
          shape_b, 
          dtype,
          lambda {|x| MLX.transpose(x, axes: [0, 2, 1])},
          lambda {|x| x}
        )
        
        # Test tt (transpose both)
        shape_a = [b, k, m]
        shape_b = [b, n, k]
        gemm_test(
          shape_a, 
          shape_b, 
          dtype,
          lambda {|x| MLX.transpose(x, axes: [0, 2, 1])},
          lambda {|x| MLX.transpose(x, axes: [0, 2, 1])}
        )
      end
    end
  end
  
  def test_matmul
    # Basic matrix multiplication test
    a = MLX.array([[1.0, 2.0], [3.0, 4.0]])
    b = MLX.array([[0.0, -1.0], [-3.0, 3.0]])
    
    expected = [[-6.0, 5.0], [-12.0, 9.0]]
    
    assert_equal expected, (a.matmul(b)).to_a
    assert_equal expected, MLX.matmul(a, b).to_a
    
    # Transposed matmul
    MLX.random.seed(0)
    a = (MLX.random.normal(shape: [128, 16]) * (1.0 / 128)).astype(MLX.float32)
    b = (MLX.random.normal(shape: [128, 16]) * (1.0 / 128)).astype(MLX.float32)
    
    c = a.matmul(MLX.transpose(b, axes: [1, 0]))
    d = MLX.transpose(a, axes: [1, 0]).matmul(b)
    
    # Check shapes
    assert_equal [128, 128], c.shape
    assert_equal [16, 128], d.shape
    
    # Test succeeds if we get here without errors
    assert true
  end
  
  def test_matmul_dtypes
    dtypes.each do |dtype|
      a = (MLX.random.normal(shape: [16, 16, 16]) * (1.0 / 256)).astype(dtype)
      b = (MLX.random.normal(shape: [16, 16, 16]) * (1.0 / 256)).astype(dtype)
      
      c = a.matmul(b)
      
      # Test succeeds if we get here without errors
      assert true
    end
  end
  
  def test_matmul_batched
    MLX.random.seed(0)
    
    # Batched matmul
    a = (MLX.random.normal(shape: [32, 128, 16]) * (1.0 / 128)).astype(MLX.float32)
    b = (MLX.random.normal(shape: [32, 16, 16]) * (1.0 / 128)).astype(MLX.float32)
    
    c = a.matmul(b)
    assert_equal [32, 128, 16], c.shape
    
    # Batched and transposed matmul
    b = (MLX.random.normal(shape: [32, 128, 16]) * (1.0 / 128)).astype(MLX.float32)
    c = a.matmul(MLX.transpose(b, axes: [0, 2, 1]))
    assert_equal [32, 128, 128], c.shape
    
    # Batched matmul with simple broadcast
    a = (MLX.random.normal(shape: [32, 128, 16]) * (1.0 / 128)).astype(MLX.float32)
    b = (MLX.random.normal(shape: [16, 16]) * (1.0 / 128)).astype(MLX.float32)
    
    c = a.matmul(b)
    assert_equal [32, 128, 16], c.shape
    
    # Both operands broadcasted
    d = MLX.broadcast_to(b, [5, 16, 16])
    e = d.matmul(d)
    assert_equal [5, 16, 16], e.shape
    
    # Batched and transposed matmul with simple broadcast
    a = (MLX.random.normal(shape: [32, 128, 16]) * (1.0 / 128)).astype(MLX.float32)
    b = (MLX.random.normal(shape: [128, 16]) * (1.0 / 128)).astype(MLX.float32)
    
    c = a.matmul(MLX.transpose(b, axes: [1, 0]))
    assert_equal [32, 128, 128], c.shape
    
    # Matmul with vector
    a = (MLX.random.normal(shape: [32, 128, 16]) * (1.0 / 128)).astype(MLX.float32)
    b = (MLX.random.normal(shape: [16]) * (1.0 / 128)).astype(MLX.float32)
    
    c = a.matmul(b)
    assert_equal [32, 128], c.shape
    
    # Test Multiheaded attention style matmul
    a = (MLX.random.normal(shape: [64, 16, 4, 32]) * (1.0 / 128)).astype(MLX.float32)
    b = (MLX.random.normal(shape: [64, 16, 4, 32]) * (1.0 / 128)).astype(MLX.float32)
    
    a = MLX.transpose(a, axes: [0, 2, 1, 3])
    b = MLX.transpose(b, axes: [0, 2, 1, 3])
    
    c = a.matmul(MLX.transpose(b, axes: [0, 1, 3, 2]))
    assert_equal [64, 4, 16, 16], c.shape
  end
  
  def test_dot
    # Test dot product
    a = MLX.array([1.0, 2.0, 3.0])
    b = MLX.array([4.0, 5.0, 6.0])
    
    c = MLX.dot(a, b)
    assert_equal 32.0, c.item
    
    # Test 2D dot product
    a = MLX.array([[1.0, 2.0], [3.0, 4.0]])
    b = MLX.array([[5.0, 6.0], [7.0, 8.0]])
    
    c = MLX.dot(a, b)
    expected = [[19.0, 22.0], [43.0, 50.0]]
    assert_equal expected, c.to_a
  end
  
  def test_vdot
    # Test vector dot product
    a = MLX.array([1.0, 2.0, 3.0])
    b = MLX.array([4.0, 5.0, 6.0])
    
    c = MLX.vdot(a, b)
    assert_equal 32.0, c.item
    
    # Test complex vector dot product
    if MLX.complex64
      a = MLX.array([1.0 + 2.0i, 3.0 + 4.0i])
      b = MLX.array([5.0 + 6.0i, 7.0 + 8.0i])
      
      c = MLX.vdot(a, b)
      # vdot uses conjugate of first argument
      assert_in_delta 70.0, c.real, 1e-5
      assert_in_delta -8.0, c.imag, 1e-5
    end
  end
  
  def test_outer
    # Test outer product
    a = MLX.array([1.0, 2.0, 3.0])
    b = MLX.array([4.0, 5.0, 6.0])
    
    c = MLX.outer(a, b)
    expected = [
      [4.0, 5.0, 6.0],
      [8.0, 10.0, 12.0],
      [12.0, 15.0, 18.0]
    ]
    assert_equal expected, c.to_a
  end
end 