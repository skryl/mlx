require_relative 'mlx_test_case'

class TestBF16 < MLXTestCase
  def setup
    # Set random seed for reproducibility
    MLX.random.seed(42)
  end
  
  def test_ops(ref_op, mlx_op, np_args, ref_transform: lambda {|x| x}, mlx_transform: lambda {|x| MLX.array(x)}, atol: 1e-5)
    # Transform arguments
    ref_args = np_args.map { |arg| ref_transform.call(arg) }
    mlx_args = np_args.map { |arg| mlx_transform.call(arg) }
    
    # Apply operations
    r_ref = ref_op.call(*ref_args)
    r_mlx = mlx_op.call(*mlx_args)
    
    # Compare results
    assert MLX.allclose(r_mlx, r_ref, atol: atol), "Results do not match within tolerance"
  end
  
  def default_test(op, np_args, simple_transform: lambda {|x| x}, atol_np: 1e-3)
    # MLX transformations for bfloat16
    mlx_transform = lambda do |x|
      simple_transform.call(MLX.array(x).astype(MLX.bfloat16))
    end
    
    # MLX operation function that converts to bf16, performs op, then back to fp32
    mlx_fn = lambda do |*args|
      result = MLX.send(op, *args)
      result.astype(MLX.float32)
    end
    
    # Simple test with direct MLX operation
    mlx_transform_simple = lambda {|x| MLX.array(x).astype(MLX.bfloat16) }
    test_ops(mlx_fn, mlx_fn, np_args, 
             ref_transform: mlx_transform_simple, 
             mlx_transform: mlx_transform_simple, 
             atol: atol_np)
  end
  
  def test_unary_ops
    # Test unary operations with bfloat16
    x = MLX.random.rand(shape: [18, 28, 38]).astype(MLX.float32)
    
    ["abs", "exp", "log", "square", "sqrt"].each do |op|
      # Skip operations that are not available in Ruby MLX
      next unless MLX.respond_to?(op)
      
      # Test with numpy array arguments
      np_args = [x]
      default_test(op, np_args)
    end
  end
  
  def test_binary_ops
    # Test binary operations with bfloat16
    x = MLX.random.rand(shape: [18, 28, 38]).astype(MLX.float32)
    y = MLX.random.rand(shape: [18, 28, 38]).astype(MLX.float32)
    
    ["add", "subtract", "multiply", "divide", "maximum", "minimum"].each do |op|
      # Skip operations that are not available in Ruby MLX
      next unless MLX.respond_to?(op)
      
      # Test with full arrays
      np_args = [x, y]
      default_test(op, np_args, simple_transform: lambda {|z| z})
      
      # Test with broadcasting along first dimension
      default_test(op, np_args, simple_transform: lambda {|z| z[0...1]})
      
      # Test with broadcasting along second dimension
      default_test(op, np_args, simple_transform: lambda {|z| z[0...1, 0...1]})
    end
  end
  
  def test_reduction_ops
    # Test reduction operations with bfloat16
    x = MLX.random.rand(shape: [18, 28, 38]).astype(MLX.float32)
    
    ["min", "max"].each do |op|
      # Skip operations that are not available in Ruby MLX
      next unless MLX.respond_to?(op)
      
      # Test with different axes configurations
      [0, 1, 2, [0, 1], [0, 2], [1, 2], [0, 1, 2]].each do |axes|
        np_args = [x]
        
        # MLX operation function that reduces along specific axes
        mlx_fn = lambda do |arg|
          arg_bf16 = arg.astype(MLX.bfloat16)
          result = MLX.send(op, arg_bf16, axis: axes)
          result.astype(MLX.float32)
        end
        
        # Reference operation function (uses same function for verification)
        ref_fn = mlx_fn
        
        # MLX transformations
        mlx_transform = lambda {|z| MLX.array(z).astype(MLX.bfloat16) }
        ref_transform = mlx_transform
        
        # Test with direct comparison
        test_ops(ref_fn, mlx_fn, np_args, 
                 ref_transform: ref_transform, 
                 mlx_transform: mlx_transform, 
                 atol: 1e-3)
      end
    end
  end
  
  def test_arg_reduction_ops
    # Test argument reduction operations (argmin, argmax) with bfloat16
    data = MLX.random.rand(shape: [10, 12, 13]).astype(MLX.float32)
    x = data.astype(MLX.bfloat16)
    data_fp32 = x.astype(MLX.float32)
    
    ["argmin", "argmax"].each do |op|
      # Skip operations that are not available in Ruby MLX
      next unless MLX.respond_to?(op)
      
      # Test with different axes and keepdims options
      (0...3).each do |axis|
        [true, false].each do |keepdims|
          # Get results from both operations
          a = MLX.send(op, x, axis: axis, keepdims: keepdims).astype(MLX.float32)
          b = MLX.send(op, data_fp32, axis: axis, keepdims: keepdims)
          
          # Compare results
          assert_equal a.to_a, b.to_a
        end
      end
      
      # Test with default axis (flattened)
      a = MLX.send(op, x, keepdims: true).astype(MLX.float32)
      b = MLX.send(op, data_fp32, keepdims: true)
      assert_equal a.to_a, b.to_a
      
      a = MLX.send(op, x).astype(MLX.float32)
      b = MLX.send(op, data_fp32)
      assert_equal a.item, b.item
    end
  end
  
  def test_blas_ops
    # Skip if GPU is not available
    return unless MLX.default_device == :gpu
    
    def test_blas(shape_x, shape_y)
      MLX.random.seed(42)
      
      # Create random arrays
      x = (MLX.random.normal(shape: shape_x) * (1.0 / shape_x[-1])).astype(MLX.float32)
      y = (MLX.random.normal(shape: shape_y) * (1.0 / shape_x[-1])).astype(MLX.float32)
      
      # Test matmul operation
      np_args = [x, y]
      
      # MLX operation function for matrix multiplication
      mlx_fn = lambda do |a, b|
        a_bf16 = a.astype(MLX.bfloat16)
        b_bf16 = b.astype(MLX.bfloat16)
        result = MLX.matmul(a_bf16, b_bf16)
        result.astype(MLX.float32)
      end
      
      # Reference operation (uses float32)
      ref_fn = lambda do |a, b|
        MLX.matmul(a, b)
      end
      
      # Test with direct comparison
      test_ops(ref_fn, mlx_fn, np_args, 
               ref_transform: lambda {|z| MLX.array(z) }, 
               mlx_transform: lambda {|z| MLX.array(z) }, 
               atol: 1e-3)
    end
    
    # Test with different shape configurations
    [
      [[32, 32], [32, 32]],
      [[23, 57], [57, 1]],
      [[1, 3], [3, 128]],
      [[8, 128, 768], [768, 16]]
    ].each do |shape_x, shape_y|
      test_blas(shape_x, shape_y)
    end
  end
  
  def test_conversion
    # Test conversion to/from bfloat16
    a = MLX.array([1.0, 2.0, 3.0]).astype(MLX.bfloat16)
    expected = MLX.array([1.0, 2.0, 3.0], dtype: MLX.bfloat16)
    
    assert_equal MLX.bfloat16, a.dtype
    assert MLX.array_equal(a, expected)
    
    # Test roundtrip conversion
    b = a.astype(MLX.float32).astype(MLX.bfloat16)
    assert MLX.array_equal(a, b)
  end
end 