require_relative 'mlx_test_case'

class TestDouble < MLXTestCase
  def setup
    # Set random seed for reproducibility
    MLX.random.seed(42)
  end
  
  def test_unary_ops
    # Test unary operations with double precision
    shape = [3, 3]
    x = MLX.random.normal(shape: shape)
    
    # Test for GPU exception (double not supported on GPU)
    if MLX.default_device == :gpu
      assert_raises(ArgumentError) do
        x.astype(MLX.float64)
      end
    end
    
    # Convert to double on CPU
    x_double = x.astype(MLX.float64, stream: :cpu)
    
    # Test various unary operations
    ops = [
      :abs,
      :arccos,
      :arccosh,
      :arcsin,
      :arcsinh,
      :arctan,
      :arctanh,
      :ceil,
      :erf,
      :erfinv,
      :exp,
      :expm1,
      :floor,
      :log,
      :logical_not,
      :negative,
      :round,
      :sin,
      :sinh,
      :sqrt,
      :rsqrt,
      :tan,
      :tanh
    ]
    
    ops.each do |op|
      # Skip if operation not available in MLX Ruby
      next unless MLX.respond_to?(op)
      
      if MLX.default_device == :gpu
        # Double precision not supported on GPU
        assert_raises(ArgumentError) do
          MLX.send(op, x_double)
        end
        next
      end
      
      # Test on CPU
      y = MLX.send(op, x)
      y_double = MLX.send(op, x_double)
      
      # Compare results (convert double back to float32 for comparison)
      assert MLX.allclose(y, y_double.astype(MLX.float32, :cpu), equal_nan: true)
    end
  end
  
  def test_binary_ops
    # Test binary operations with double precision
    shape = [3, 3]
    a = MLX.random.normal(shape: shape)
    b = MLX.random.normal(shape: shape)
    
    # Convert to double on CPU
    a_double = a.astype(MLX.float64, stream: :cpu)
    b_double = b.astype(MLX.float64, stream: :cpu)
    
    # Test various binary operations
    ops = [
      :add,
      :arctan2,
      :divide,
      :multiply,
      :subtract,
      :logical_and,
      :logical_or,
      :remainder,
      :maximum,
      :minimum,
      :power,
      :equal,
      :greater,
      :greater_equal,
      :less,
      :less_equal,
      :not_equal,
      :logaddexp
    ]
    
    ops.each do |op|
      # Skip if operation not available in MLX Ruby
      next unless MLX.respond_to?(op)
      
      if MLX.default_device == :gpu
        # Double precision not supported on GPU
        assert_raises(ArgumentError) do
          MLX.send(op, a_double, b_double)
        end
        next
      end
      
      # Test on CPU
      y = MLX.send(op, a, b)
      y_double = MLX.send(op, a_double, b_double)
      
      # Compare results (convert double back to float32 for comparison)
      assert MLX.allclose(y, y_double.astype(MLX.float32, :cpu), equal_nan: true)
    end
  end
  
  def test_where
    # Test where operation with double precision
    shape = [3, 3]
    cond = MLX.random.uniform(shape: shape) > 0.5
    a = MLX.random.normal(shape: shape)
    b = MLX.random.normal(shape: shape)
    
    # Convert to double on CPU
    a_double = a.astype(MLX.float64, stream: :cpu)
    b_double = b.astype(MLX.float64, stream: :cpu)
    
    if MLX.default_device == :gpu
      # Double precision not supported on GPU
      assert_raises(ArgumentError) do
        MLX.where(cond, a_double, b_double)
      end
      return
    end
    
    # Test on CPU
    y = MLX.where(cond, a, b)
    y_double = MLX.where(cond, a_double, b_double)
    
    # Compare results (convert double back to float32 for comparison)
    assert MLX.allclose(y, y_double.astype(MLX.float32, :cpu))
  end
  
  def test_reductions
    # Test reduction operations with double precision
    shape = [32, 32]
    a = MLX.random.normal(shape: shape)
    
    # Convert to double on CPU
    a_double = a.astype(MLX.float64, stream: :cpu)
    
    # Test various reduction operations with different axes
    axes = [0, 1, [0, 1]]
    ops = [:sum, :prod, :min, :max, :any, :all]
    
    ops.each do |op|
      # Skip if operation not available in MLX Ruby
      next unless MLX.respond_to?(op)
      
      axes.each do |ax|
        if MLX.default_device == :gpu
          # Double precision not supported on GPU
          assert_raises(ArgumentError) do
            MLX.send(op, a_double, axis: ax)
          end
          next
        end
        
        # Test on CPU
        y = MLX.send(op, a)
        y_double = MLX.send(op, a_double)
        
        # Compare results (convert double back to float32 for comparison)
        assert MLX.allclose(y, y_double.astype(MLX.float32, :cpu))
      end
    end
  end
  
  def test_get_and_set_item
    # Test indexing and assignment with double precision
    shape = [3, 3]
    a = MLX.random.normal(shape: shape)
    b = MLX.random.normal(shape: [2])
    
    # Convert to double on CPU
    a_double = a.astype(MLX.float64, stream: :cpu)
    b_double = b.astype(MLX.float64, stream: :cpu)
    
    # Create index arrays
    idx_i = MLX.array([0, 2])
    idx_j = MLX.array([0, 2])
    
    if MLX.default_device == :gpu
      # Double precision not supported on GPU
      assert_raises(ArgumentError) do
        a_double[idx_i, idx_j]
      end
    else
      # Test indexing on CPU
      y = a[idx_i, idx_j]
      y_double = a_double[idx_i, idx_j]
      
      # Compare results (convert double back to float32 for comparison)
      assert MLX.allclose(y, y_double.astype(MLX.float32, :cpu))
    end
    
    if MLX.default_device == :gpu
      # Double precision not supported on GPU
      assert_raises(ArgumentError) do
        a_double[idx_i, idx_j] = b_double
      end
    else
      # Test assignment on CPU
      a[idx_i, idx_j] = b
      a_double[idx_i, idx_j] = b_double
      
      # Compare results (convert double back to float32 for comparison)
      assert MLX.allclose(a, a_double.astype(MLX.float32, :cpu))
    end
  end
  
  def test_gemm
    # Test matrix multiplication with double precision
    shape = [8, 8]
    a = MLX.random.normal(shape: shape)
    b = MLX.random.normal(shape: shape)
    
    # Convert to double on CPU
    a_double = a.astype(MLX.float64, stream: :cpu)
    b_double = b.astype(MLX.float64, stream: :cpu)
    
    if MLX.default_device == :gpu
      # Double precision not supported on GPU
      assert_raises(ArgumentError) do
        a_double.matmul(b_double)
      end
      return
    end
    
    # Test on CPU
    y = a.matmul(b)
    y_double = a_double.matmul(b_double)
    
    # Compare results (convert double back to float32 for comparison)
    assert MLX.allclose(y, y_double.astype(MLX.float32, :cpu), equal_nan: true)
  end
  
  def test_type_promotion
    # Test type promotion with double precision
    a = MLX.array([4, 8], dtype: MLX.float64)
    b = MLX.array([4, 8], dtype: MLX.int32)
    
    MLX.stream(:cpu) do
      c = a + b
      assert_equal MLX.float64, c.dtype
    end
  end
  
  def test_lapack
    # Test linear algebra operations with double precision
    MLX.stream(:cpu) do
      # QR factorization
      a = MLX.array([[2.0, 3.0], [1.0, 2.0]], dtype: MLX.float64)
      q, r = MLX.linalg.qr(a)
      
      # Check QR factorization results
      out = q.matmul(r)
      assert MLX.allclose(out, a)
      
      out = q.transpose.matmul(q)
      assert MLX.allclose(out, MLX.eye(2))
      
      assert MLX.allclose(MLX.tril(r, -1), MLX.zeros_like(r))
      assert_equal MLX.float64, q.dtype
      assert_equal MLX.float64, r.dtype
      
      # Singular Value Decomposition (SVD)
      a = MLX.array(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        dtype: MLX.float64
      )
      u, s, vt = MLX.linalg.svd(a)
      
      # Check SVD results
      s_diag = MLX.diag(s)
      reconstructed = u[:, 0...s.size].matmul(s_diag).matmul(vt)
      assert MLX.allclose(reconstructed, a)
      
      # Matrix Inverse
      a = MLX.array([[1, 2, 3], [6, -5, 4], [-9, 8, 7]], dtype: MLX.float64)
      a_inv = MLX.linalg.inv(a)
      
      # Check inverse results
      assert MLX.allclose(a.matmul(a_inv), MLX.eye(a.shape[0]))
      
      # Triangular matrix inverse
      a = MLX.array([[1, 0, 0], [6, -5, 0], [-9, 8, 7]], dtype: MLX.float64)
      b = MLX.array([[7, 0, 0], [3, -2, 0], [1, 8, 3]], dtype: MLX.float64)
      ab = MLX.stack([a, b])
      
      invs = MLX.linalg.tri_inv(ab, upper: false)
      
      # Check triangular inverse results
      ab.each_with_index do |m, i|
        m_inv = invs[i]
        assert MLX.allclose(m.matmul(m_inv), MLX.eye(m.shape[0]))
      end
      
      # Cholesky decomposition
      sqrt_a = MLX.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype: MLX.float64
      )
      a = sqrt_a.transpose.matmul(sqrt_a) / 81.0
      
      l = MLX.linalg.cholesky(a)
      u = MLX.linalg.cholesky(a, upper: true)
      
      # Check Cholesky decomposition results
      assert MLX.allclose(l.matmul(l.transpose), a)
      assert MLX.allclose(u.transpose.matmul(u), a)
      
      # Pseudoinverse
      a = MLX.array([[1, 2, 3], [6, -5, 4], [-9, 8, 7]], dtype: MLX.float64)
      a_plus = MLX.linalg.pinv(a)
      
      # Check pseudoinverse properties
      assert MLX.allclose(a.matmul(a_plus).matmul(a), a)
      
      # Eigenvalue decomposition for symmetric matrices
      a = MLX.array([[1.0, 2.0], [2.0, 4.0]], dtype: MLX.float64)
      eig_vals, eig_vecs = MLX.linalg.eigh(a)
      
      # Check eigenvectors and eigenvalues
      eigen_eq = a.matmul(eig_vecs)
      eigen_scaled = eig_vals.reshape(-1, 1) * eig_vecs
      assert MLX.allclose(eigen_eq, eigen_scaled)
      
      # Test eigenvalues only
      eig_vals_only = MLX.linalg.eigvalsh(a)
      assert MLX.allclose(eig_vals, eig_vals_only)
    end
  end
  
  def test_conversion
    # Test conversion between different floating-point precisions
    x = MLX.array([1.0, 2.0, 3.0], dtype: MLX.float32)
    
    if MLX.default_device == :gpu
      # Double precision not supported on GPU
      assert_raises(ArgumentError) do
        x.astype(MLX.float64)
      end
      
      # Convert to CPU first
      x_cpu = x.to(:cpu)
      x_double = x_cpu.astype(MLX.float64)
      assert_equal MLX.float64, x_double.dtype
      
      # Convert back to float32
      x_float32 = x_double.astype(MLX.float32)
      assert_equal MLX.float32, x_float32.dtype
      assert MLX.allclose(x_cpu, x_float32)
    else
      # On CPU, double conversion should work directly
      x_double = x.astype(MLX.float64)
      assert_equal MLX.float64, x_double.dtype
      
      # Convert back to float32
      x_float32 = x_double.astype(MLX.float32)
      assert_equal MLX.float32, x_float32.dtype
      assert MLX.allclose(x, x_float32)
    end
  end
end 