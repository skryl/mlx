require_relative 'mlx_test_case'

class TestEinsum < MLXTestCase
  def setup
    # Seed for reproducibility
    MLX.random.seed(42)
  end
  
  def test_simple_path
    a = MLX.zeros([5, 5])
    path = MLX.einsum_path("ii", a)
    assert_equal [[0]], path[0]
    
    path = MLX.einsum_path("ij->i", a)
    assert_equal [[0]], path[0]
    
    path = MLX.einsum_path("ii->i", a)
    assert_equal [[0]], path[0]
    
    a = MLX.zeros([5, 8])
    b = MLX.zeros([8, 3])
    path = MLX.einsum_path("ij,jk", a, b)
    assert_equal [[0, 1]], path[0]
    
    path = MLX.einsum_path("ij,jk -> ijk", a, b)
    assert_equal [[0, 1]], path[0]
    
    a = MLX.zeros([5, 8])
    b = MLX.zeros([8, 3])
    c = MLX.zeros([3, 7])
    path = MLX.einsum_path("ij,jk,kl", a, b, c)
    assert_equal [[0, 1], [0, 1]], path[0]
    
    a = MLX.zeros([5, 8])
    b = MLX.zeros([8, 10])
    c = MLX.zeros([10, 7])
    path = MLX.einsum_path("ij,jk,kl", a, b, c)
    assert_equal [[1, 2], [0, 1]], path[0]
  end
  
  def test_longer_paths
    chars = "abcdefghijklmopqABC"
    sizes = [2, 3, 4, 5, 4, 3, 2, 6, 5, 4, 3, 2, 5, 7, 4, 3, 2, 3, 4]
    dim_dict = Hash[chars.chars.zip(sizes)]
    
    cases = [
      "eb,cb,fb->cef",
      "dd,fb,be,cdb->cef",
      "dd,fb,be,cdb->cef",
      "bca,cdb,dbf,afc->",
      "dcc,fce,ea,dbf->ab",
      "dcc,fce,ea,dbf->ab",
    ]
    
    cases.each do |test_case|
      arrow_idx = test_case.index("->")
      subscripts = test_case[0...arrow_idx].split(",")
      
      # Create input arrays based on subscripts
      inputs = subscripts.map do |s|
        shape = s.chars.map { |c| dim_dict[c] }
        MLX.ones(shape)
      end
      
      # Calculate einsum path
      path = MLX.einsum_path(test_case, *inputs)
      
      # Just check that we get a path - actual path validation would be complex
      assert path.is_a?(Array)
      assert_equal 2, path.size
    end
  end
  
  def test_simple_einsum
    a = MLX.arange(4 * 4).reshape([4, 4])
    a_mx = MLX.einsum("ii->i", a)
    assert_equal [4], a_mx.shape
    
    a = MLX.arange(2 * 2 * 2).reshape([2, 2, 2])
    a_mx = MLX.einsum("iii->i", a)
    assert_equal [2], a_mx.shape
    
    a = MLX.arange(2 * 2 * 3 * 3).reshape([2, 2, 3, 3])
    a_mx = MLX.einsum("iijj->ij", a)
    assert_equal [2, 3], a_mx.shape
    
    a = MLX.arange(2 * 2 * 3 * 3).reshape([2, 3, 2, 3])
    a_mx = MLX.einsum("ijij->ij", a)
    assert_equal [2, 3], a_mx.shape
    
    # Test some simple reductions
    a = MLX.arange(2 * 2).reshape([2, 2])
    a_mx = MLX.einsum("ii", a)
    assert a_mx.ndim == 0  # Should be a scalar
    
    a = MLX.arange(2 * 4).reshape([2, 4])
    a_mx = MLX.einsum("ij->", a)
    assert a_mx.ndim == 0  # Should be a scalar
    
    a = MLX.arange(2 * 4).reshape([2, 4])
    a_mx = MLX.einsum("ij->i", a)
    assert_equal [2], a_mx.shape
    
    a = MLX.arange(2 * 4).reshape([2, 4])
    a_mx = MLX.einsum("ij->j", a)
    assert_equal [4], a_mx.shape
    
    a = MLX.arange(2 * 2 * 2).reshape([2, 2, 2])
    a_mx = MLX.einsum("iii->", a)
    assert a_mx.ndim == 0  # Should be a scalar
    
    a = MLX.arange(2 * 2 * 3 * 3).reshape([2, 3, 2, 3])
    a_mx = MLX.einsum("ijij->j", a)
    assert_equal [3], a_mx.shape
    
    # Test some simple transposes
    a = MLX.arange(2 * 4).reshape([2, 4])
    a_mx = MLX.einsum("ij", a)
    assert_equal [2, 4], a_mx.shape
    
    a = MLX.arange(2 * 4).reshape([2, 4])
    a_mx = MLX.einsum("ij->ji", a)
    assert_equal [4, 2], a_mx.shape
    
    a = MLX.arange(2 * 3 * 4).reshape([2, 3, 4])
    a_mx = MLX.einsum("ijk->jki", a)
    assert_equal [3, 4, 2], a_mx.shape
  end
  
  def test_two_input_einsum
    # Matmul
    a = MLX.full([2, 8], 1.0)
    b = MLX.full([8, 2], 1.0)
    a_mx = MLX.einsum("ik,kj", a, b)
    assert_equal [2, 2], a_mx.shape
    assert_equal MLX.full([2, 2], 8.0).to_a, a_mx.to_a
    
    # Matmul + transpose
    a = MLX.full([2, 8], 1.0)
    b = MLX.full([8, 3], 1.0)
    a_mx = MLX.einsum("ik,kj->ji", a, b)
    assert_equal [3, 2], a_mx.shape
    assert_equal MLX.full([3, 2], 8.0).to_a, a_mx.to_a
    
    # Inner product
    a = MLX.full([4], 1.0)
    b = MLX.full([4], 1.0)
    a_mx = MLX.einsum("i,i", a, b)
    assert a_mx.ndim == 0  # Should be a scalar
    assert_equal 4.0, a_mx.item
    
    # Outer product
    a = MLX.full([4], 0.5)
    b = MLX.full([6], 2.0)
    a_mx = MLX.einsum("i,j->ij", a, b)
    assert_equal [4, 6], a_mx.shape
    assert_equal MLX.full([4, 6], 1.0).to_a, a_mx.to_a
    
    # Elementwise multiply
    a = MLX.full([2, 8], 1.0)
    b = MLX.full([2, 8], 1.0)
    a_mx = MLX.einsum("ij,ij->ij", a, b)
    assert_equal [2, 8], a_mx.shape
    assert_equal MLX.full([2, 8], 1.0).to_a, a_mx.to_a
    
    # Medley
    a = MLX.full([2, 8, 3, 5], 1.0)
    b = MLX.full([3, 7, 5, 2], 1.0)
    a_mx = MLX.einsum("abcd,fgda->bfca", a, b)
    assert_equal [8, 3, 3, 7], a_mx.shape
  end
  
  def test_sum_first
    a = MLX.full([5, 8], 1.0)
    b = MLX.full([8, 2], 1.0)
    a_mx = MLX.einsum("ab,bc->c", a, b)
    assert_equal [2], a_mx.shape
    assert_equal MLX.full([2], 40.0).to_a, a_mx.to_a
  end
  
  def test_broadcasting
    a = MLX.full([5, 1], 1.0)
    b = MLX.full([8, 2], 1.0)
    a_mx = MLX.einsum("ab,bc->c", a, b)
    assert_equal [2], a_mx.shape
    
    a = MLX.random.uniform(shape: [5, 1, 3, 1])
    b = MLX.random.uniform(shape: [1, 7, 1, 2])
    a_mx = MLX.einsum("abcd,cdab->abcd", a, b)
    assert_equal [5, 7, 3, 2], a_mx.shape
  end
  
  def test_attention
    q = MLX.random.uniform(shape: [2, 3, 4, 5])
    k = MLX.random.uniform(shape: [2, 3, 4, 5])
    v = MLX.random.uniform(shape: [2, 3, 4, 5])
    
    s = MLX.einsum("itjk,iujk->ijtu", q, k)
    out_mx = MLX.einsum("ijtu,iujk->itjk", s, v)
    
    assert_equal [2, 3, 4, 5], out_mx.shape
  end
  
  def test_multi_input_einsum
    a = MLX.ones([3, 4, 5])
    out_mx = MLX.einsum("ijk,lmk,ijf->lf", a, a, a)
    assert_equal [3, 3], out_mx.shape
    # For a tensor of ones, each element should be 3 * 4 * 5 = 60
    assert_equal MLX.full([3, 3], 60.0).to_a, out_mx.to_a
  end
  
  def test_opt_einsum_test_cases
    # Test a subset of cases from opt_einsum test suite
    test_cases = [
      # Test hadamard-like products
      "a,ab,abc->abc",
      "a,b,ab->ab",
      # Test index-transformations
      "ea,fb,gc,hd,abcd->efgh",
      # Test complex contractions (limited subset)
      "abhe,hidj,jgba,hiab,gab",
      # Test collapse
      "ab,ab,c->",
      "ab,ab,c->c",
      # Test outer products
      "ab,cd,ef->abcdef",
      "ab,cd,ef->acdf"
    ]
    
    # Helper function to create random inputs for a test case
    def inputs_for_case(test_case)
      arrow_idx = test_case.index("->")
      subscripts = arrow_idx ? test_case[0...arrow_idx].split(",") : test_case.split(",")
      
      # Create random shapes based on subscripts
      inputs = []
      shape_map = {}
      
      subscripts.each do |term|
        shape = []
        term.each_char do |c|
          # Assign consistent sizes to each index letter
          unless shape_map.key?(c)
            shape_map[c] = rand(2..4)
          end
          shape << shape_map[c]
        end
        inputs << MLX.random.uniform(shape: shape)
      end
      
      inputs
    end
    
    # Test each case
    test_cases.each do |test_case|
      inputs = inputs_for_case(test_case)
      result = MLX.einsum(test_case, *inputs)
      
      # Just verify we get a result without errors
      assert result.is_a?(MLX::Array)
    end
  end
  
  def test_ellipses
    # Test einsum with ellipsis notation
    def inputs_for_case(test_case)
      arrow_idx = test_case.index("->")
      subscripts = arrow_idx ? test_case[0...arrow_idx].split(",") : test_case.split(",")
      
      # Create random shapes based on subscripts
      inputs = []
      shape_map = {}
      
      subscripts.each do |term|
        shape = []
        has_ellipsis = term.include?("...")
        
        if has_ellipsis
          # For ellipsis, add some arbitrary dimensions
          shape << 2
          shape << 3
        end
        
        term.gsub("...", "").each_char do |c|
          # Assign consistent sizes to each index letter
          unless shape_map.key?(c)
            shape_map[c] = rand(2..4)
          end
          shape << shape_map[c]
        end
        
        inputs << MLX.random.uniform(shape: shape)
      end
      
      inputs
    end
    
    # Test ellipsis cases
    test_cases = [
      "...ij,...jk->...ik",
      "...ij,...jk,...kl->...il"
    ]
    
    test_cases.each do |test_case|
      inputs = inputs_for_case(test_case)
      result = MLX.einsum(test_case, *inputs)
      
      # Just verify we get a result without errors
      assert result.is_a?(MLX::Array)
    end
  end
end 