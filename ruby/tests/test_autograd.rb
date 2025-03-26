require_relative 'mlx_test_case'

class TestAutograd < MLXTestCase
  def test_jvp
    # Test jacobian-vector product
    fun = ->(x) { x * 2 }
    out, dout = MLX.jvp(fun, [MLX.array(1.0)], [MLX.array(2.0)])
    assert_equal 2.0, out[0].item
    assert_equal 4.0, dout[0].item
    
    # Test jvp with multiple inputs and outputs
    fun = ->(x, y) { x * y }
    _, out = MLX.jvp(
      fun, 
      [MLX.array(4.0), MLX.array(2.0)], 
      [MLX.array(3.0), MLX.array(2.0)]
    )
    # Expected: 4.0 * 2.0 + 2.0 * 3.0 = 8.0 + 6.0 = 14.0
    assert_equal 14.0, out[0].item
    
    # Test jvp with multiple outputs
    fun = ->(x, y, z) { [x * y, y * z] }
    _, out = MLX.jvp(
      fun,
      [MLX.array(2.0), MLX.array(4.0), MLX.array(6.0)],
      [MLX.array(1.0), MLX.array(3.0), MLX.array(1.0)]
    )
    assert_equal 2, out.length
    # For first output: 4.0 * 1.0 + 2.0 * 3.0 = 4.0 + 6.0 = 10.0
    assert_equal 10.0, out[0].item
    # For second output: 4.0 * 1.0 + 6.0 * 3.0 = 4.0 + 18.0 = 22.0
    assert_equal 22.0, out[1].item
  end
  
  def test_vjp
    # Test vector-jacobian product
    fun = ->(x) { x * 2 }
    out, dout = MLX.vjp(fun, [MLX.array(1.0)], [MLX.array(2.0)])
    assert_equal 2.0, out[0].item
    assert_equal 4.0, dout[0].item
    
    # Test vjp with multiple inputs
    fun = ->(x, y) { x * y }
    _, dout = MLX.vjp(
      fun, 
      [MLX.array(4.0), MLX.array(2.0)], 
      [MLX.array(3.0)]
    )
    assert_equal 6.0, dout[0].item  # 3.0 * 2.0
    assert_equal 12.0, dout[1].item  # 3.0 * 4.0
    
    # Test vjp with multiple outputs
    fun = ->(x, y, z) { [x * y, y * z] }
    _, out = MLX.vjp(
      fun,
      [MLX.array(2.0), MLX.array(4.0), MLX.array(6.0)],
      [MLX.array(1.0), MLX.array(3.0)]
    )
    assert_equal 3, out.length
    assert_equal 4.0, out[0].item  # 1.0 * 4.0
    assert_equal 14.0, out[1].item  # 1.0 * 2.0 + 3.0 * 6.0
    assert_equal 12.0, out[2].item  # 3.0 * 4.0
  end
  
  def test_grad
    # Test basic gradient
    fun = ->(x) { x * x }
    
    value, dfdx = MLX.value_and_grad(fun).call(MLX.array(0.5))
    assert_equal 0.25, value.item
    assert_equal 1.0, dfdx.item
    
    dfdx = MLX.grad(fun).call(MLX.array(0.5))
    assert_equal 1.0, dfdx.item
    
    # Test higher order gradients
    df2dx2 = MLX.grad(MLX.grad(fun)).call(MLX.array(0.5))
    assert_equal 2.0, df2dx2.item
    
    df3dx3 = MLX.grad(MLX.grad(MLX.grad(fun))).call(MLX.array(0.5))
    assert_equal 0.0, df3dx3.item
    
    # Test gradients with respect to specific arguments
    fun = ->(x, y) { x * y }
    x = MLX.array(2.0)
    y = MLX.array(3.0)
    
    dfdx = MLX.grad(fun, argnums: 0).call(x, y)
    assert_equal 3.0, dfdx.item
    
    dfdy = MLX.grad(fun, argnums: 1).call(x, y)
    assert_equal 2.0, dfdy.item
    
    # Test that grad works with non-array arguments
    fun = ->(x, y) { x }
    value, dfdx = MLX.value_and_grad(fun).call(MLX.array(2.0), "hello")
    assert_equal 2.0, value.item
    assert_equal 1.0, dfdx.item
    
    dfdx = MLX.grad(fun).call(MLX.array(2.0), "hello")
    assert_equal 1.0, dfdx.item
    
    # Test errors
    # Function doesn't return array
    fun = ->(x) { "hello" }
    assert_raises(ValueError) do
      MLX.grad(fun).call(MLX.array(2.0))
    end
    
    # Invalid argument number
    fun = ->(x) { x }
    assert_raises(ValueError) do
      MLX.grad(fun, argnums: 2).call(MLX.array(2.0))
    end
    
    assert_raises(ValueError) do
      MLX.grad(fun, argnums: -2).call(MLX.array(2.0))
    end
    
    # Invalid argument type
    assert_raises(ValueError) do
      MLX.grad(fun).call("hello")
    end
    
    # Output not a scalar
    fun = ->(x) { MLX.sum(x, keepdims: true) }
    assert_raises(ValueError) do
      MLX.grad(fun).call(MLX.ones([2, 2]))
    end
  end
  
  def test_grad_trees
    # Test gradients with trees of arguments
    fun = ->(x, y) { x * y }
    
    value, dfdx = MLX.value_and_grad(fun, [0, 1]).call(MLX.array(0.5), MLX.array(2.0))
    assert_equal 1.0, value.item
    assert dfdx.is_a?(Array)
    assert_equal 2.0, dfdx[0].item
    assert_equal 0.5, dfdx[1].item
    
    value, dfdx = MLX.value_and_grad(fun, 1).call(MLX.array(0.5), MLX.array(2.0))
    assert_equal 1.0, value.item
    assert_equal 0.5, dfdx.item
    
    # Test with hash inputs
    fun = ->(p) { p["x"] * p["y"] }
    
    value, dfdx = MLX.value_and_grad(fun).call({"x" => MLX.array(0.5), "y" => MLX.array(2.0)})
    assert_equal 1.0, value.item
    assert_equal 2.0, dfdx["x"].item
    assert_equal 0.5, dfdx["y"].item
    
    # Test errors
    assert_raises(ValueError) do
      MLX.value_and_grad(fun).call({"x" => 0.5, "y" => MLX.array(2.0)})
    end
    
    assert_raises(ValueError) do
      MLX.value_and_grad(fun, [0, 1]).call({"x" => MLX.array(0.5), "y" => MLX.array(2.0)})
    end
    
    # Test with nested structures
    fun = ->(p, b) { MLX.square(p[0]["foo"][2]) * b }
    
    value, dfdx = MLX.value_and_grad(fun).call(
      [{"foo" => [[], [], MLX.array(2.0)]}], 
      MLX.array(0.5)
    )
    assert_equal 2.0, value.item
    assert_equal 2.0, dfdx[0]["foo"][2].item
    
    # More error cases
    fun = ->(x) { x }
    
    assert_raises(TypeError) do
      MLX.value_and_grad(fun, [nil, nil])
    end
    
    assert_raises(ValueError) do
      MLX.value_and_grad(fun, [])
    end
    
    assert_raises(ValueError) do
      MLX.grad(fun, argnums: [0, 0])
    end
  end
  
  def test_auxiliary_values
    # Test functions that return additional values beyond the loss
    fun = ->(x, y) do
      l = MLX.sum(x * y)
      extra = {
        "loss" => l, 
        "foo" => MLX.square(y) + MLX.square(x), 
        "bar" => [1, 2, 3, y, x]
      }
      [l, extra]
    end
    
    fun_value_grad = MLX.value_and_grad(fun)
    fun_grad = MLX.grad(fun)
    
    (loss, a), b = fun_value_grad.call(MLX.ones([2, 2]), MLX.ones([2, 2]))
    assert_equal 4, a["loss"].item
    assert MLX.array_equal(b, MLX.ones([2, 2]))
    assert MLX.array_equal(a["foo"], 2 * MLX.ones([2, 2]))
    assert_equal [1, 2, 3], a["bar"][0..2]
    assert MLX.array_equal(a["bar"][3], MLX.ones([2, 2]))
    assert MLX.array_equal(a["bar"][4], MLX.ones([2, 2]))
    
    # Error case - grad doesn't support auxiliary values
    assert_raises(ValueError) do
      fun_grad.call(MLX.ones([2, 2]), MLX.ones([2, 2]))
    end
  end
  
  def test_captured
    # Test functions with captured variables
    a = MLX.array(5.0)
    
    f = ->(x) { a + x }
    g = ->(x) { a + a }
    h = ->(x) { x + x }
    
    dfdx = MLX.grad(f)
    assert_equal 1.0, dfdx.call(a).item
    
    dgdx = MLX.grad(g)
    assert_equal 0.0, dgdx.call(a).item
    
    dhdx = MLX.grad(h)
    assert_equal 2.0, dhdx.call(a).item
    
    # Test second-order gradients
    d2fdx2 = MLX.grad(dfdx)
    assert_equal 0.0, d2fdx2.call(a).item
    
    d2gdx2 = MLX.grad(dgdx)
    assert_equal 0.0, d2gdx2.call(a).item
    
    d2hdx2 = MLX.grad(dhdx)
    assert_equal 0.0, d2hdx2.call(a).item
  end
  
  def test_stop_gradient
    # Test stop_gradient function
    shape_in = [4, 4]
    w_in = MLX.ones(shape_in)
    x_in = MLX.ones(shape_in)
    
    h = ->(w, x) do
      x1 = 2 * x
      y = MLX.stop_gradient(x1)
      y1 = 3 * y
      w.matmul(y1)
    end
    
    # Compute gradients with respect to w and x
    h_x = MLX.grad(h, argnums: 1)
    h_w = MLX.grad(h, argnums: 0)
    
    # Gradient w.r.t. w should be normal
    dw = h_w.call(w_in, x_in)
    assert MLX.array_equal(dw, 6 * MLX.ones(shape_in))
    
    # Gradient w.r.t. x should be zero due to stop_gradient
    dx = h_x.call(w_in, x_in)
    assert MLX.array_equal(dx, MLX.zeros(shape_in))
  end
  
  def test_custom_function
    # Only run if custom function API is available
    skip unless defined?(MLX::Extension.custom_op)
    
    # Create a custom exponential function
    my_exp = MLX::Extension.custom_op("my_exp", ->(x) { MLX.exp(x) })
    
    # Define VJP (gradient) for the custom function
    my_exp = my_exp.def_vjp(->(x, dx, ex) { 
      # Gradient of exp(x) is exp(x) * dx
      dx * ex  
    })
    
    # Define JVP (forward-mode gradient)
    my_exp = my_exp.def_jvp(->(x, dx) { 
      # Forward-mode derivative is also exp(x) * dx
      [MLX.exp(x), MLX.exp(x) * dx]  
    })
    
    # Register the function
    my_exp = my_exp.register
    
    # Test function
    x = MLX.array(2.0)
    result = my_exp.call(x)
    expected = MLX.exp(x)
    assert_allclose(result, expected)
    
    # Test gradient
    grad_fn = MLX.grad(->(x) { my_exp.call(x) })
    dx = grad_fn.call(x)
    expected_dx = MLX.exp(x)
    assert_allclose(dx, expected_dx)
  end
end 