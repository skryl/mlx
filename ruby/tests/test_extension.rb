require_relative 'mlx_test_case'

class TestExtension < MLXTestCase
  def test_custom_op_creation
    # Create a custom operation that squares an array
    square_op = MLX::Extension.custom_op(
      "square",
      ->(x) { MLX.multiply(x, x) }
    )
    
    # Register the operation
    square = square_op.register
    
    # Test the operation
    x = MLX.array([1.0, 2.0, 3.0, 4.0])
    result = square.call(x)
    
    # Check result
    assert_array_equal(result, [1.0, 4.0, 9.0, 16.0])
  end
  
  def test_custom_op_with_vjp
    # Create a custom operation with custom gradient
    cube_op = MLX::Extension.custom_op(
      "cube",
      ->(x) { MLX.multiply(MLX.multiply(x, x), x) }
    ).def_vjp(
      ->(x, dy, _) { MLX.multiply(MLX.multiply(3.0, MLX.square(x)), dy) }
    )
    
    # Register the operation
    cube = cube_op.register
    
    # Test the operation
    x = MLX.array([1.0, 2.0, 3.0])
    result = cube.call(x)
    
    # Check result
    assert_array_equal(result, [1.0, 8.0, 27.0])
    
    # Test gradients
    grad_fn = MLX.grad(cube)
    grad_x = grad_fn.call(x)
    
    # Gradient of x^3 is 3x^2
    assert_array_equal(grad_x, [3.0, 12.0, 27.0])
  end
  
  def test_custom_op_with_jvp
    # Create a custom operation with forward-mode gradient
    square_root_op = MLX::Extension.custom_op(
      "sqrt_custom",
      ->(x) { MLX.sqrt(x) }
    ).def_jvp(
      ->(x, v, _) { MLX.multiply(v, MLX.divide(0.5, MLX.sqrt(x))) }
    )
    
    # Register the operation
    sqrt_custom = square_root_op.register
    
    # Test the operation
    x = MLX.array([4.0, 9.0, 16.0])
    result = sqrt_custom.call(x)
    
    # Check result
    assert_array_equal(result, [2.0, 3.0, 4.0])
    
    # Test gradients
    grad_fn = MLX.grad(sqrt_custom)
    grad_x = grad_fn.call(x)
    
    # Gradient of sqrt(x) is 0.5/sqrt(x)
    assert_array_equal(grad_x, [0.25, 1.0/6.0, 0.125], atol: 1e-5)
  end
  
  def test_custom_implementation_type
    # Skip test if implementation selection isn't available
    skip unless MLX::Extension.respond_to?(:current_impl=)
    
    # Create a custom operation with CPU and GPU implementations
    abs_op = MLX::Extension.custom_op(
      "custom_abs",
      ->(x) { MLX.abs(x) }
    ).def_impl(:cpu)
    
    # Register the operation
    custom_abs = abs_op.register
    
    # Test on CPU
    MLX::Extension.with_impl(->(x) { custom_abs.call(x) }, MLX.array([-1.0, -2.0, 3.0]), impl: :cpu)
    
    # Also test the current_impl setter
    old_impl = MLX::Extension.current_impl
    begin
      MLX::Extension.current_impl = :cpu
      result = custom_abs.call(MLX.array([-1.0, -2.0, 3.0]))
      assert_array_equal(result, [1.0, 2.0, 3.0])
    ensure
      MLX::Extension.current_impl = old_impl
    end
  end
  
  def test_gradient_checkpointing
    skip unless defined?(MLX::Extension.checkpoint)
    
    # Define a function to checkpoint
    fn = ->(x) { MLX.multiply(MLX.multiply(x, x), x) }  # x^3
    
    # Create checkpointed version
    checkpointed = MLX::Extension.checkpoint(fn)
    
    # Test forward pass
    x = MLX.array([2.0, 3.0])
    y1 = fn.call(x)
    y2 = checkpointed.call(x)
    
    # Results should be identical
    assert_array_equal(y1, y2)
    
    # Test gradients
    grad_fn1 = MLX.grad(fn)
    grad_fn2 = MLX.grad(checkpointed)
    
    g1 = grad_fn1.call(x)
    g2 = grad_fn2.call(x)
    
    # Gradients should be identical
    assert_array_equal(g1, g2)
  end
  
  def test_custom_function_with_multiple_outputs
    # Create a custom function that returns multiple outputs
    minmax_op = MLX::Extension.custom_op(
      "minmax",
      ->(x) { [MLX.min(x), MLX.max(x)] }
    )
    
    # Register the operation
    minmax = minmax_op.register
    
    # Test the operation
    x = MLX.array([1.0, 5.0, 3.0, 2.0])
    min_val, max_val = minmax.call(x)
    
    # Check results
    assert_equal 1.0, min_val.item
    assert_equal 5.0, max_val.item
  end
  
  def test_register_primitive
    # Register a primitive operation directly
    MLX::Extension.register_primitive(
      "double", 
      ->(x) { MLX.multiply(x, 2.0) },
      vjp: ->(x, dy, _) { MLX.multiply(dy, 2.0) }
    )
    
    # Test the operation
    x = MLX.array([1.0, 2.0, 3.0])
    result = MLX.double(x)
    
    # Check result
    assert_array_equal(result, [2.0, 4.0, 6.0])
    
    # Test gradients
    grad_fn = MLX.grad(->(x) { MLX.double(x) })
    grad_x = grad_fn.call(x)
    
    # Gradient of 2x is 2
    assert_array_equal(grad_x, [2.0, 2.0, 2.0])
  end
end 