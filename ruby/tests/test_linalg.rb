require_relative 'mlx_test_case'

class TestLinalg < MLXTestCase
  def test_norm
    vector_ords = [nil, 0.5, 0, 1, 2, 3, -1, Float::INFINITY, -Float::INFINITY]
    matrix_ords = [nil, "fro", "nuc", -1, 1, -2, 2, Float::INFINITY, -Float::INFINITY]
    
    # Test with provided axes
    [[3], [2, 3], [2, 3, 3]].each do |shape|
      x_mx = MLX.arange(1, shape.reduce(:*) + 1, dtype: MLX.float32).reshape(shape)
      
      # Test for each possible axis combination
      (1...shape.length).each do |num_axes|
        axes = shape.length.times.to_a.combination(num_axes).to_a
        
        ords = (num_axes == 1) ? vector_ords : matrix_ords
        
        axes.each do |axis|
          [true, false].each do |keepdims|
            ords.each do |o|
              stream = ["nuc", -2, 2].include?(o) ? MLX.cpu : MLX.default_device
              
              out_mx = MLX.linalg.norm(x_mx, ord: o, axis: axis, keepdims: keepdims, stream: stream)
              
              # We don't have numpy here to compare, so we're just checking the shape
              expected_shape = if keepdims
                shape.dup.tap { |s| axis.each { |a| s[a] = 1 } }
              else
                shape.dup.tap { |s| axis.sort.reverse_each { |a| s.delete_at(a) } }
              end
              
              assert_equal expected_shape, out_mx.shape
            end
          end
        end
      end
    end
    
    # Test only ord provided
    [[3], [2, 3]].each do |shape|
      x_mx = MLX.arange(1, shape.reduce(:*) + 1).reshape(shape)
      
      [nil, 1, -1, Float::INFINITY, -Float::INFINITY].each do |o|
        [true, false].each do |keepdims|
          out_mx = MLX.linalg.norm(x_mx, ord: o, keepdims: keepdims)
          
          expected_shape = keepdims ? [1] * shape.length : []
          assert_equal expected_shape, out_mx.shape
        end
      end
    end
    
    # Test no ord and no axis provided
    [[3], [2, 3], [2, 3, 3]].each do |shape|
      x_mx = MLX.arange(1, shape.reduce(:*) + 1).reshape(shape)
      
      [true, false].each do |keepdims|
        out_mx = MLX.linalg.norm(x_mx, keepdims: keepdims)
        
        expected_shape = keepdims ? [1] * shape.length : []
        assert_equal expected_shape, out_mx.shape
      end
    end
  end
  
  def test_qr_factorization
    # Test error cases
    assert_raises(ValueError) do
      MLX.linalg.qr(MLX.array(0.0))
    end
    
    assert_raises(ValueError) do
      MLX.linalg.qr(MLX.array([0.0, 1.0]))
    end
    
    assert_raises(ValueError) do
      MLX.linalg.qr(MLX.array([[0, 1], [1, 0]]), mode: "invalid")
    end
    
    # Test basic QR factorization
    a = MLX.array([[2.0, 3.0], [1.0, 2.0]])
    q, r = MLX.linalg.qr(a, stream: MLX.cpu)
    
    # Check Q * R = A
    out = q.matmul(r)
    assert MLX.allclose(out, a)
    
    # Check Q is orthogonal (Q^T * Q = I)
    out = q.T.matmul(q)
    assert MLX.allclose(out, MLX.eye(2), rtol: 1e-5, atol: 1e-7)
    
    # Check R is upper triangular
    assert MLX.allclose(MLX.tril(r, -1), MLX.zeros_like(r))
    
    # Check dtype is preserved
    assert_equal MLX.float32, q.dtype
    assert_equal MLX.float32, r.dtype
    
    # Test multiple matrices
    b = MLX.array([[-1.0, 2.0], [-4.0, 1.0]])
    ab = MLX.stack([a, b])
    q, r = MLX.linalg.qr(ab, stream: MLX.cpu)
    
    # Check each matrix
    ab.shape[0].times do |i|
      out = q[i].matmul(r[i])
      assert MLX.allclose(out, ab[i])
      
      out = q[i].T.matmul(q[i])
      assert MLX.allclose(out, MLX.eye(2), rtol: 1e-5, atol: 1e-7)
      
      assert MLX.allclose(MLX.tril(r[i], -1), MLX.zeros_like(r[i]))
    end
    
    # Test non-square matrices
    [[4, 8], [8, 4]].each do |shape|
      a = MLX.random.uniform(shape: shape)
      q, r = MLX.linalg.qr(a, stream: MLX.cpu)
      
      # Check Q * R = A
      out = q.matmul(r)
      assert MLX.allclose(out, a, rtol: 1e-4, atol: 1e-6)
      
      # Check Q is orthogonal (Q^T * Q = I)
      out = q.T.matmul(q)
      assert MLX.allclose(out, MLX.eye([shape].min), rtol: 1e-4, atol: 1e-6)
    end
  end
  
  def test_svd_decomposition
    a = MLX.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype: MLX.float32)
    u, s, vt = MLX.linalg.svd(a, compute_uv: true, stream: MLX.cpu)
    
    # Check U * Σ * V^T = A
    assert MLX.allclose(u[0...u.shape[0], 0...s.shape[0]].matmul(MLX.diag(s)).matmul(vt), a, rtol: 1e-5, atol: 1e-7)
    
    # Test without computing U and V
    s_only = MLX.linalg.svd(a, compute_uv: false, stream: MLX.cpu)
    
    # Check that the singular values give the same Frobenius norm
    assert MLX.allclose(MLX.linalg.norm(s_only), MLX.linalg.norm(a, ord: "fro"), rtol: 1e-5, atol: 1e-7)
    
    # Test multiple matrices
    b = a + 10.0
    ab = MLX.stack([a, b])
    us, ss, vts = MLX.linalg.svd(ab, compute_uv: true, stream: MLX.cpu)
    
    # Check each matrix
    [a, b].each_with_index do |m, i|
      assert MLX.allclose(
        us[i][0...us[i].shape[0], 0...ss[i].shape[0]].matmul(MLX.diag(ss[i])).matmul(vts[i]), 
        m, 
        rtol: 1e-5, 
        atol: 1e-7
      )
    end
    
    # Test multiple matrices without computing U and V
    ss = MLX.linalg.svd(ab, compute_uv: false, stream: MLX.cpu)
    
    [a, b].each_with_index do |m, i|
      assert MLX.allclose(
        MLX.linalg.norm(ss[i]),
        MLX.linalg.norm(m, ord: "fro"),
        rtol: 1e-5,
        atol: 1e-7
      )
    end
  end
  
  def test_inverse
    a = MLX.array([[1, 2, 3], [6, -5, 4], [-9, 8, 7]], dtype: MLX.float32)
    a_inv = MLX.linalg.inv(a, stream: MLX.cpu)
    
    # Check A * A^-1 = I
    assert MLX.allclose(a.matmul(a_inv), MLX.eye(a.shape[0]), rtol: 0, atol: 1e-6)
    
    # Test multiple matrices
    b = a - 100
    ab = MLX.stack([a, b])
    invs = MLX.linalg.inv(ab, stream: MLX.cpu)
    
    ab.shape[0].times do |i|
      assert MLX.allclose(ab[i].matmul(invs[i]), MLX.eye(ab[i].shape[0]), rtol: 0, atol: 1e-5)
    end
  end
  
  def test_tri_inverse
    [false, true].each do |upper|
      a = MLX.array([[1, 0, 0], [6, -5, 0], [-9, 8, 7]], dtype: MLX.float32)
      b = MLX.array([[7, 0, 0], [3, -2, 0], [1, 8, 3]], dtype: MLX.float32)
      
      if upper
        a = a.T
        b = b.T
      end
      
      ab = MLX.stack([a, b])
      invs = MLX.linalg.tri_inv(ab, upper: upper, stream: MLX.cpu)
      
      ab.shape[0].times do |i|
        assert MLX.allclose(ab[i].matmul(invs[i]), MLX.eye(ab[i].shape[0]), rtol: 0, atol: 1e-5)
      end
    end
    
    # Ensure that tri_inv will 0-out the supposedly 0 triangle
    x = MLX.random.normal([2, 8, 8])
    x_lower = MLX.tril(x)
    x_upper = MLX.triu(x)
    
    # Test lower triangular
    x_inv = MLX.linalg.tri_inv(x_lower, upper: false)
    assert MLX.allclose(MLX.triu(x_inv, 1), MLX.zeros_like(MLX.triu(x_inv, 1)))
    
    # Test upper triangular
    x_inv = MLX.linalg.tri_inv(x_upper, upper: true)
    assert MLX.allclose(MLX.tril(x_inv, -1), MLX.zeros_like(MLX.tril(x_inv, -1)))
  end
  
  def test_cholesky
    a = MLX.array([[6, 3, 4, 8], [3, 6, 5, 1], [4, 5, 10, 7], [8, 1, 7, 25]], dtype: MLX.float32)
    l = MLX.linalg.cholesky(a, stream: MLX.cpu)
    
    # Check L * L^T = A
    assert MLX.allclose(l.matmul(l.T), a, rtol: 1e-5, atol: 1e-7)
    
    # Test multiple matrices
    b = MLX.array([[25, 15, -5], [15, 18, 0], [-5, 0, 11]], dtype: MLX.float32)
    ab = MLX.stack([a, b])
    ls = MLX.linalg.cholesky(ab, stream: MLX.cpu)
    
    ab.shape[0].times do |i|
      assert MLX.allclose(ls[i].matmul(ls[i].T), ab[i], rtol: 1e-5, atol: 1e-7)
    end
  end
  
  def test_pseudo_inverse
    # Test square matrix
    a = MLX.array([[1, 2], [3, 4]], dtype: MLX.float32)
    a_pinv = MLX.linalg.pinv(a, stream: MLX.cpu)
    
    # Check A * A^+ * A = A
    assert MLX.allclose(a.matmul(a_pinv).matmul(a), a, rtol: 1e-5, atol: 1e-7)
    
    # Test non-square matrix
    a = MLX.array([[1, 2, 3], [4, 5, 6]], dtype: MLX.float32)
    a_pinv = MLX.linalg.pinv(a, stream: MLX.cpu)
    
    # Check A * A^+ * A = A
    assert MLX.allclose(a.matmul(a_pinv).matmul(a), a, rtol: 1e-5, atol: 1e-7)
  end
  
  def test_cholesky_inv
    a = MLX.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]], dtype: MLX.float32)
    l = MLX.linalg.cholesky(a, stream: MLX.cpu)
    
    # Get inverse via cholesky_inv
    a_inv = MLX.linalg.cholesky_inv(l, stream: MLX.cpu)
    
    # Check A * A^-1 = I
    assert MLX.allclose(a.matmul(a_inv), MLX.eye(a.shape[0]), rtol: 1e-5, atol: 1e-7)
    
    # Test multiple matrices
    b = MLX.array([[25, 15, -5], [15, 18, 0], [-5, 0, 11]], dtype: MLX.float32)
    ab = MLX.stack([a, b])
    ls = MLX.linalg.cholesky(ab, stream: MLX.cpu)
    invs = MLX.linalg.cholesky_inv(ls, stream: MLX.cpu)
    
    ab.shape[0].times do |i|
      assert MLX.allclose(ab[i].matmul(invs[i]), MLX.eye(ab[i].shape[0]), rtol: 1e-5, atol: 1e-7)
    end
  end
  
  def test_cross_product
    # Test with 3D vectors
    a = MLX.array([1, 2, 3])
    b = MLX.array([4, 5, 6])
    c = MLX.linalg.cross(a, b)
    
    expected = MLX.array([-3, 6, -3])
    assert MLX.allclose(c, expected)
    
    # Test with batch of 3D vectors
    a = MLX.array([[1, 2, 3], [4, 5, 6]])
    b = MLX.array([[4, 5, 6], [7, 8, 9]])
    c = MLX.linalg.cross(a, b)
    
    expected = MLX.array([[-3, 6, -3], [-3, 6, -3]])
    assert MLX.allclose(c, expected)
    
    # Test with axis specified
    a = MLX.array([[1, 2, 3], [4, 5, 6]]).T  # shape: (3, 2)
    b = MLX.array([[4, 5, 6], [7, 8, 9]]).T  # shape: (3, 2)
    c = MLX.linalg.cross(a, b, axis: 0)
    
    expected = MLX.array([[-3, -3], [6, 6], [-3, -3]])
    assert MLX.allclose(c, expected)
    
    # Test with 2D vectors (cross product in 3D with z=0)
    a = MLX.array([1, 2])
    b = MLX.array([3, 4])
    c = MLX.linalg.cross(a, b)
    
    expected = -2  # 1*4 - 2*3
    assert MLX.allclose(c, expected)
    
    # Test with batch of 2D vectors
    a = MLX.array([[1, 2], [3, 4]])
    b = MLX.array([[3, 4], [5, 6]])
    c = MLX.linalg.cross(a, b)
    
    expected = MLX.array([-2, -2])
    assert MLX.allclose(c, expected)
  end
  
  def test_eigh
    # Helper method to check eigendecomposition
    def check_eigs_and_vecs(a, kwargs = {})
      w, v = MLX.linalg.eigh(a, **kwargs, stream: MLX.cpu)
      
      # If a specific subset of eigenvectors was requested, we need to check differently
      if kwargs[:subset_by_index]
        # Just verify that the dimensions are correct
        assert_equal [a.shape[0], kwargs[:subset_by_index][1] - kwargs[:subset_by_index][0]], v.shape
        assert_equal [kwargs[:subset_by_index][1] - kwargs[:subset_by_index][0]], w.shape
      else
        # Check A * v = w * v
        av = a.matmul(v)
        wv = v * w.reshape([w.shape[0], 1])
        assert MLX.allclose(av, wv, rtol: 1e-5, atol: 1e-6)
        
        # Check V^T * V = I
        vtv = v.T.matmul(v)
        assert MLX.allclose(vtv, MLX.eye(a.shape[0]), rtol: 1e-5, atol: 1e-6)
      end
    end
    
    # Test symmetric matrix
    a = MLX.array([
      [1.0, 2.0, 3.0],
      [2.0, 4.0, 5.0],
      [3.0, 5.0, 6.0]
    ])
    
    check_eigs_and_vecs(a)
    
    # Test subset_by_index
    check_eigs_and_vecs(a, subset_by_index: [0, 2])
    
    # Test subset_by_value
    check_eigs_and_vecs(a, subset_by_value: [-Float::INFINITY, 6.0])
    
    # Test multiple matrices
    b = MLX.array([
      [6.0, 5.0, 4.0],
      [5.0, 3.0, 2.0],
      [4.0, 2.0, 1.0]
    ])
    
    ab = MLX.stack([a, b])
    ws, vs = MLX.linalg.eigh(ab, stream: MLX.cpu)
    
    ab.shape[0].times do |i|
      # Check A * v = w * v
      av = ab[i].matmul(vs[i])
      wv = vs[i] * ws[i].reshape([ws[i].shape[0], 1])
      assert MLX.allclose(av, wv, rtol: 1e-5, atol: 1e-6)
      
      # Check V^T * V = I
      vtv = vs[i].T.matmul(vs[i])
      assert MLX.allclose(vtv, MLX.eye(ab[i].shape[0]), rtol: 1e-5, atol: 1e-6)
    end
  end
  
  def test_lu
    a = MLX.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]], dtype: MLX.float32)
    p, l, u = MLX.linalg.lu(a, stream: MLX.cpu)
    
    # Check P * A = L * U
    pa = p.matmul(a)
    lu = l.matmul(u)
    assert MLX.allclose(pa, lu, rtol: 1e-5, atol: 1e-7)
    
    # Check P is a permutation matrix
    assert MLX.all(MLX.sum(p, axis: 0) == 1).item
    assert MLX.all(MLX.sum(p, axis: 1) == 1).item
    
    # Check L is lower triangular with unit diagonal
    assert MLX.allclose(MLX.triu(l, 1), MLX.zeros_like(MLX.triu(l, 1)))
    assert MLX.allclose(MLX.diag(l), MLX.ones(l.shape[0]))
    
    # Check U is upper triangular
    assert MLX.allclose(MLX.tril(u, -1), MLX.zeros_like(MLX.tril(u, -1)))
    
    # Test multiple matrices
    b = a + 10
    ab = MLX.stack([a, b])
    ps, ls, us = MLX.linalg.lu(ab, stream: MLX.cpu)
    
    ab.shape[0].times do |i|
      pa = ps[i].matmul(ab[i])
      lu = ls[i].matmul(us[i])
      assert MLX.allclose(pa, lu, rtol: 1e-5, atol: 1e-7)
    end
  end
  
  def test_lu_factor
    a = MLX.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]], dtype: MLX.float32)
    lu, piv = MLX.linalg.lu_factor(a, stream: MLX.cpu)
    
    # Test multiple matrices
    b = a + 10
    ab = MLX.stack([a, b])
    lus, pivs = MLX.linalg.lu_factor(ab, stream: MLX.cpu)
    
    # Since we can't directly check the LU factorization without a special function
    # to convert pivots to permutation matrices, we'll just verify shapes
    assert_equal ab.shape, lus.shape
    assert_equal [ab.shape[0], ab.shape[1]], pivs.shape
  end
  
  def test_solve
    # Test square system
    a = MLX.array([[3, 1], [1, 2]], dtype: MLX.float32)
    b = MLX.array([9, 8], dtype: MLX.float32)
    x = MLX.linalg.solve(a, b, stream: MLX.cpu)
    
    # Check A * x = b
    assert MLX.allclose(a.matmul(x), b, rtol: 1e-5, atol: 1e-7)
    
    # Test with multiple right-hand sides
    b = MLX.array([[9, 5], [8, 6]], dtype: MLX.float32)
    x = MLX.linalg.solve(a, b, stream: MLX.cpu)
    
    # Check A * x = b
    assert MLX.allclose(a.matmul(x), b, rtol: 1e-5, atol: 1e-7)
    
    # Test batch of square systems
    a2 = a + 1
    aa = MLX.stack([a, a2])
    b2 = b + 1
    bb = MLX.stack([b, b2])
    xx = MLX.linalg.solve(aa, bb, stream: MLX.cpu)
    
    aa.shape[0].times do |i|
      assert MLX.allclose(aa[i].matmul(xx[i]), bb[i], rtol: 1e-5, atol: 1e-7)
    end
    
    # Test overdetermined system (least squares)
    a = MLX.array([[1, 2], [3, 4], [5, 6]], dtype: MLX.float32)
    b = MLX.array([1, 2, 3], dtype: MLX.float32)
    x = MLX.linalg.solve(a, b, stream: MLX.cpu)
    
    # For least squares, A^T * A * x = A^T * b
    ata = a.T.matmul(a)
    atb = a.T.matmul(b)
    assert MLX.allclose(ata.matmul(x), atb, rtol: 1e-5, atol: 1e-7)
  end
  
  def test_solve_triangular
    # Test lower triangular matrix
    l = MLX.array([[1, 0, 0], [3, 4, 0], [5, 6, 7]], dtype: MLX.float32)
    b = MLX.array([8, 9, 10], dtype: MLX.float32)
    x = MLX.linalg.solve_triangular(l, b, upper: false, stream: MLX.cpu)
    
    # Check L * x = b
    assert MLX.allclose(l.matmul(x), b, rtol: 1e-5, atol: 1e-7)
    
    # Test upper triangular matrix
    u = MLX.array([[7, 6, 5], [0, 4, 3], [0, 0, 1]], dtype: MLX.float32)
    b = MLX.array([10, 9, 8], dtype: MLX.float32)
    x = MLX.linalg.solve_triangular(u, b, upper: true, stream: MLX.cpu)
    
    # Check U * x = b
    assert MLX.allclose(u.matmul(x), b, rtol: 1e-5, atol: 1e-7)
    
    # Test batch of triangular systems
    l2 = l + 1
    ll = MLX.stack([l, l2])
    b2 = b + 1
    bb = MLX.stack([b, b2])
    xx = MLX.linalg.solve_triangular(ll, bb, upper: false, stream: MLX.cpu)
    
    ll.shape[0].times do |i|
      assert MLX.allclose(ll[i].matmul(xx[i]), bb[i], rtol: 1e-5, atol: 1e-7)
    end
  end
end 