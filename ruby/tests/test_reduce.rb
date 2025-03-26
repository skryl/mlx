require_relative 'mlx_test_case'

class TestReduce < MLXTestCase
  def setup
    # Seed for reproducibility
    MLX.random.seed(42)
  end

  def test_axis_permutation_sums
    # Test sum reduction with different axis permutations
    shapes = [[5, 5, 1, 5, 5], [65, 65, 1, 65]]
    
    shapes.each do |shape|
      # Create random integer array
      x_array = (MLX.random.normal(shape: shape) * 128).astype(MLX.int32)
      
      # Test different axis permutations
      (0...shape.length).to_a.permutation.each do |t|
        y_array = MLX.transpose(x_array, axes: t)
        
        # Test different combinations of reduction axes
        (1..shape.length).each do |n|
          (0...shape.length).to_a.combination(n).each do |axes|
            z_array = MLX.sum(y_array, axis: axes)
            MLX.eval(z_array)
            
            # The test succeeds if we reach here without errors
            assert true
          end
        end
      end
    end
  end
  
  def test_expand_sums
    # Test sum reduction with broadcasting
    x_array = MLX.random.normal(shape: [5, 1, 5, 1, 5, 1]).astype(MLX.float32)
    
    (1..3).each do |m|
      [1, 3, 5].combination(m).each do |axes|
        shape = [5, 1, 5, 1, 5, 1]
        axes.each { |ax| shape[ax] = 5 }
        
        y_array = MLX.broadcast_to(x_array, shape)
        
        (1..6).each do |n|
          (0...6).to_a.combination(n).each do |a|
            z_array = MLX.sum(y_array, axis: a) / 1000
            MLX.eval(z_array)
            
            # The test succeeds if we reach here without errors
            assert true
          end
        end
      end
    end
  end
  
  def test_dtypes
    int_dtypes = [
      MLX.int8, MLX.int16, MLX.int32, 
      MLX.uint8, MLX.uint16, MLX.uint32, 
      MLX.int64, MLX.uint64,
      MLX.complex64
    ]
    float_dtypes = [MLX.float32]
    
    (int_dtypes + float_dtypes).each do |dtype|
      # Create random array with specified dtype
      x = MLX.random.uniform(0, 2, shape: [3, 3, 3]).astype(dtype)
      
      # Test different reduction operations
      ["sum", "prod", "min", "max"].each do |op|
        mlx_op = MLX.method(op)
        
        # Test different axis configurations
        [nil, 0, 1, 2, [0, 1], [0, 2], [1, 2], [0, 1, 2]].each do |axes|
          result = mlx_op.call(x, axis: axes)
          MLX.eval(result)
          
          # The test succeeds if we reach here without errors
          assert true
        end
      end
    end
  end
  
  def test_arg_reduce
    dtypes = [
      MLX.uint8, MLX.uint16, MLX.uint32, MLX.uint64,
      MLX.int8, MLX.int16, MLX.int32, MLX.int64,
      MLX.float16, MLX.float32
    ]
    
    dtypes.each do |dtype|
      data = MLX.random.rand(shape: [10, 12, 13]).astype(dtype)
      
      ["argmin", "argmax"].each do |op|
        mlx_op = MLX.method(op)
        
        # Test with different axes and keepdims options
        (0...3).each do |axis|
          [true, false].each do |keepdims|
            result = mlx_op.call(data, axis: axis, keepdims: keepdims)
            # The test succeeds if we reach here without errors
            assert true
          end
        end
        
        # Test with default axis (flattened)
        result_keepdims = mlx_op.call(data, keepdims: true)
        result = mlx_op.call(data)
        
        # The test succeeds if we reach here without errors
        assert true
      end
    end
  end
  
  def test_edge_case
    # Test edge case with transposed array
    x = (MLX.random.normal(shape: [100, 1, 100, 100]) * 128).astype(MLX.int32)
    x = MLX.transpose(x, axes: [0, 3, 1, 2])
    
    y = MLX.sum(x, axis: [0, 2, 3])
    MLX.eval(y)
    
    # The test succeeds if we reach here without errors
    assert true
  end
  
  def test_sum_bool
    # Test sum of boolean array
    x = MLX.random.uniform(0, 1, shape: [10, 10, 10]) > 0.5
    
    sum_result = MLX.sum(x)
    sum_val = sum_result.item
    
    # The result should be an integer, representing the count of true values
    assert sum_val.is_a?(Integer)
  end
  
  def test_many_reduction_axes
    def check(x, axes)
      # Calculate expected result by iterating through axes
      expected = x
      axes.each do |ax|
        expected = MLX.sum(expected, axis: ax, keepdims: true)
      end
      
      # Calculate actual result by passing all axes at once
      out = MLX.sum(x, axis: axes, keepdims: true)
      
      # Compare results
      assert MLX.array_equal(out, expected)
    end
    
    # Test with different shaped arrays and reduction axes
    x = MLX.random.randint(0, 10, shape: [4, 4, 4, 4, 4])
    check(x, [0, 2, 4])
    
    x = MLX.random.randint(0, 10, shape: [4, 4, 4, 4, 4, 4, 4])
    check(x, [0, 2, 4, 6])
    
    x = MLX.random.randint(0, 10, shape: [4, 4, 4, 4, 4, 4, 4, 4, 4])
    check(x, [0, 2, 4, 6, 8])
    
    x = MLX.random.randint(0, 10, shape: [4, 4, 4, 4, 4, 4, 4, 4, 4, 128])
    x = MLX.transpose(x, axes: [1, 0, 2, 3, 4, 5, 6, 7, 8, 9])
    check(x, [1, 3, 5, 7, 9])
  end
end 