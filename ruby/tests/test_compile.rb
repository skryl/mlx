require_relative 'mlx_test_case'
require 'stringio'

class TestCompile < MLXTestCase
  def setup
    # Set random seed for reproducibility
    MLX.random.seed(42)
  end
  
  def test_simple_compile
    # Test basic compilation functionality
    fun = lambda do |x, y|
      x + y
    end
    
    compiled_fn = MLX.compile(fun)
    # Compile twice to test caching
    compiled_fn = MLX.compile(fun)
    
    x = MLX.array(1.0)
    y = MLX.array(1.0)
    out = compiled_fn.call(x, y)
    assert_equal 2.0, out.item
    
    # Try again with same inputs
    out = compiled_fn.call(x, y)
    assert_equal 2.0, out.item
    
    # Change sizes - should recompile
    x = MLX.array([1.0, 2.0])
    out = compiled_fn.call(x, y)
    assert MLX.array_equal(out, MLX.array([2.0, 3.0]))
    
    y = MLX.array([1.0, 2.0])
    out = compiled_fn.call(x, y)
    assert MLX.array_equal(out, MLX.array([2.0, 4.0]))
    
    # Change types - should recompile
    x = MLX.array([1, 2], dtype: MLX.int32)
    y = MLX.array([1, 2], dtype: MLX.int32)
    out = compiled_fn.call(x, y)
    assert_equal MLX.int32, out.dtype
    assert MLX.array_equal(out, MLX.array([2, 4]))
  end
  
  def test_compile_grad
    # Test compilation with gradients
    loss_fn = lambda do |x|
      MLX.sum(MLX.exp(x))
    end
    
    grad_fn = MLX.grad(loss_fn)
    
    x = MLX.array([0.5, -0.5, 1.2])
    dfdx = grad_fn.call(x)
    
    compile_grad_fn = MLX.compile(grad_fn)
    c_dfdx = grad_fn.call(x)
    
    assert MLX.allclose(c_dfdx, dfdx)
    
    # Run it again without calling compile
    c_dfdx = compile_grad_fn.call(x)
    assert MLX.allclose(c_dfdx, dfdx)
    
    # Run it again with calling compile
    c_dfdx = MLX.compile(grad_fn).call(x)
    assert MLX.allclose(c_dfdx, dfdx)
    
    # Value and grad
    loss_fn = lambda do |x|
      [MLX.sum(MLX.exp(x)), MLX.sin(x)]
    end
    
    val_and_grad_fn = MLX.value_and_grad(loss_fn)
    (loss, val), dfdx = val_and_grad_fn.call(x)
    (c_loss, c_val), c_dfdx = MLX.compile(val_and_grad_fn).call(x)
    
    assert MLX.allclose(c_dfdx, dfdx)
    assert MLX.allclose(c_loss, loss)
    assert MLX.allclose(c_val, val)
  end
  
  def test_compile_inputs_with_primitives
    # Test compilation with inputs that depend on primitive operations
    x = MLX.array([1, 2, 3])
    y = MLX.array([1, 2, 3])
    
    5.times do
      x = x + y
      y = y + 1
    end
    
    fun = lambda do |x, y|
      x * y
    end
    
    out = fun.call(x, y)
    
    # Reset and recompute
    x = MLX.array([1, 2, 3])
    y = MLX.array([1, 2, 3])
    
    5.times do
      x = x + y
      y = y + 1
    end
    
    c_out = MLX.compile(fun).call(x, y)
    assert MLX.array_equal(out, c_out)
    
    # Try again
    c_out = MLX.compile(fun).call(x, y)
    assert MLX.array_equal(out, c_out)
  end
  
  def test_compile_with_closure
    # Test compilation of functions with closures
    x = MLX.array(1)
    
    closure = lambda do |y|
      x + y
    end
    
    compiled = MLX.compile(closure)
    out = compiled.call(MLX.array(1))
    assert_equal 2, out.item
    
    # Try again
    out = compiled.call(MLX.array(1))
    assert_equal 2, out.item
    
    # Change the shape of the enclosed variable
    x = MLX.array([1, 2])
    out = compiled.call(MLX.array(1))
    
    # We still get the original input (closures are not updated)
    assert_equal 2, out.item
    
    # Try with a tree of enclosed variables
    x = { a: MLX.array(1), b: MLX.array(2) }
    
    closure = lambda do |y|
      x[:a] + y + x[:b]
    end
    
    compiled = MLX.compile(closure)
    out = compiled.call(MLX.array(1))
    assert_equal 4, out.item
    
    # Change the shape of one input
    x[:a] = MLX.array([4, 5])
    out = compiled.call(MLX.array(1))
    assert_equal 4, out.item
    
    x[:b] = MLX.array([-6, -8])
    out = compiled.call(MLX.array(1))
    assert_equal 4, out.item
    
    # Enclosed variable is not evaluated yet
    x = MLX.array(1)
    x = x + x
    
    closure = lambda do |y|
      x + y
    end
    
    compiled = MLX.compile(closure)
    out = compiled.call(MLX.array(2))
    assert_equal 4, out.item
    
    # And again
    out = compiled.call(MLX.array(2))
    assert_equal 4, out.item
  end
  
  def test_function_creates_array
    # Test compilation of functions that create arrays inside
    fun = lambda do |x|
      x + MLX.array(1)
    end
    
    cfun = MLX.compile(fun)
    out = cfun.call(MLX.array(3))
    assert_equal 4, out.item
    
    # And again
    out = cfun.call(MLX.array(3))
    assert_equal 4, out.item
  end
  
  def test_enable_disable
    # Test enabling and disabling of compilation
    fun = lambda do |x|
      y = x + 1
      z = x + 1
      y + z
    end
    
    count_prims = lambda do |outputs|
      buf = StringIO.new
      MLX.export_to_dot(buf, outputs)
      buf.rewind
      buf.read.split.count { |line| line.include?("label") }
    end
    
    x = MLX.array(1.0)
    cfun = MLX.compile(fun)
    n_compiled = count_prims.call(cfun.call(x))
    
    # Check disabled
    MLX.disable_compile
    n_uncompiled = count_prims.call(cfun.call(x))
    assert n_compiled < n_uncompiled, "Compiled function should have fewer primitives"
    
    # Check re-enabled
    MLX.enable_compile
    n_enable_compiled = count_prims.call(cfun.call(x))
    assert_equal n_compiled, n_enable_compiled
  end
  
  def test_compile_two_input_grad
    # Test compilation of gradient functions with two inputs
    loss = lambda do |w, x|
      y = x * w
      MLX.sum(y * MLX.exp(y))
    end
    
    x = MLX.array([1.0, 0.5, 2.0, -0.5])
    w = MLX.array([-1.0, 0.3, 1.0, -0.9])
    
    expected_grad = MLX.grad(loss).call(w, x)
    compiled_grad = MLX.compile(MLX.grad(loss)).call(w, x)
    assert MLX.allclose(expected_grad, compiled_grad)
  end
  
  def test_vmap_compiled
    # Test vmap with compiled functions
    simple_unary = lambda do |x|
      -MLX.exp(x)
    end
    
    x = MLX.array([[1.0, 2.0], [2.0, 3.0]])
    
    expected_out = MLX.vmap(simple_unary).call(x)
    out = MLX.vmap(MLX.compile(simple_unary)).call(x)
    assert MLX.allclose(expected_out, out)
    
    simple_binary = lambda do |x, y|
      MLX.abs(MLX.exp(x + y) + y)
    end
    
    x = MLX.array([[1.0, -3.0], [0.5, -0.5]])
    y = MLX.array([[2.0, -1.0], [0.25, -0.25]])
    
    expected_out = MLX.vmap(simple_binary).call(x, y)
    out = MLX.vmap(MLX.compile(simple_binary)).call(x, y)
    assert MLX.allclose(expected_out, out)
    
    expected_out = MLX.vmap(simple_binary, in_axes: [0, 1]).call(x, y)
    out = MLX.vmap(MLX.compile(simple_binary), in_axes: [0, 1]).call(x, y)
    assert MLX.allclose(expected_out, out)
    
    y = MLX.array([0.25, -0.25])
    expected_out = MLX.vmap(simple_binary, in_axes: [0, nil]).call(x, y)
    out = MLX.vmap(MLX.compile(simple_binary), in_axes: [0, nil]).call(x, y)
    assert MLX.allclose(expected_out, out)
    
    # Test nested compilation within vmap
    simple_unary_outer = lambda do |x|
      x_abs = MLX.abs(x)
      
      simple_unary_inner = MLX.compile(lambda { |z| -MLX.exp(x_abs) })
      
      simple_unary_inner.call(x_abs)
    end
    
    expected_out = -MLX.exp(MLX.abs(x))
    out = MLX.vmap(simple_unary_outer).call(x)
    assert MLX.allclose(expected_out, out)
  end
  
  def test_compile_kwargs
    # Test compilation with keyword arguments
    fun = lambda do |x, y, z|
      x + y + z
    end
    
    cfun = MLX.compile(fun)
    
    x = MLX.array(1)
    y = MLX.array(2)
    z = MLX.array(3)
    
    out = cfun.call(x, y, z)
    assert_equal 6, out.item
    
    # With keyword arguments
    out = cfun.call(x, z: z, y: y)
    assert_equal 6, out.item
  end
  
  def test_shapeless_compile
    # Test shapeless compilation (JIT based on input shapes, not values)
    fun = lambda do |x|
      y = x * 2
      z = x + 1
      y + z
    end
    
    # Compile with shapeless=true parameter
    cfun = MLX.compile(fun, shapeless: true)
    
    x = MLX.array([1.0, 2.0, 3.0])
    out = cfun.call(x)
    expected = x * 2 + x + 1
    assert MLX.allclose(out, expected)
    
    # Try different shape
    x = MLX.array([1.0, 2.0, 3.0, 4.0])
    out = cfun.call(x)
    expected = x * 2 + x + 1
    assert MLX.allclose(out, expected)
    
    # Try different type
    x = MLX.array([1, 2, 3], dtype: MLX.int32)
    out = cfun.call(x)
    expected = x * 2 + x + 1
    assert MLX.allclose(out, expected)
  end
  
  def test_compile_rng
    # Test compilation with random number generation
    # Function that accesses MLX random state
    fun = lambda do
      MLX.random.uniform(shape: [3, 4])
    end
    
    # Compile with random state as input and output
    cfun = MLX.compile(
      fun,
      inputs: MLX.random.state,
      outputs: MLX.random.state
    )
    
    # First call
    out1 = cfun.call
    
    # Second call - should get different values since random state is updated
    out2 = cfun.call
    
    # Should not be equal
    assert !MLX.array_equal(out1, out2)
  end
  
  def test_compile_with_constant
    # Test compilation with constants
    fun = lambda do |x, y|
      x + y
    end
    
    cfun = MLX.compile(fun)
    
    x = MLX.array(1)
    out = cfun.call(x, 2.0)
    assert_equal 3.0, out.item
    
    # Try with integer
    out = cfun.call(x, 2)
    assert_equal 3.0, out.item
    
    # Try with array of different shape
    out = cfun.call(x, MLX.array([2.0, 3.0]))
    assert_equal [3.0, 4.0], out.to_a
    
    # Try with tuples
    fun_tuple = lambda do |x, y|
      case y
      when Array
        x + y[0] + y[1]
      else
        x + y
      end
    end
    
    cfun = MLX.compile(fun_tuple)
    out = cfun.call(x, [2, 3])
    assert_equal 6, out.item
  end
  
  def test_compile_multi_output
    # Test compilation of function with multiple outputs
    fn = lambda do |x|
      y = x * 2
      z = x + 1
      w = x - 1
      [y, z, w]
    end
    
    cfn = MLX.compile(fn)
    
    x = MLX.array([1.0, 2.0, 3.0])
    y, z, w = cfn.call(x)
    
    assert MLX.array_equal(y, x * 2)
    assert MLX.array_equal(z, x + 1)
    assert MLX.array_equal(w, x - 1)
  end
  
  def test_dtypes
    # Test compilation with different dtypes
    dtypes = [MLX.float32, MLX.int32, MLX.uint32, MLX.int64]
    
    dtypes.each do |dtype|
      fn = lambda do |x|
        x + x
      end
      
      cfn = MLX.compile(fn)
      
      x = MLX.array([1, 2, 3], dtype: dtype)
      y = cfn.call(x)
      
      assert_equal dtype, y.dtype
      assert MLX.array_equal(y, x + x)
    end
  end
  
  def test_compile_many_inputs
    # Test compilation with many inputs
    fun = lambda do |*inputs|
      result = inputs[0]
      inputs[1..-1].each do |input|
        result = result + input
      end
      result
    end
    
    cfun = MLX.compile(fun)
    
    inputs = (0...100).map { |i| MLX.array(i) }
    expected = inputs.reduce { |sum, x| sum + x }
    
    result = cfun.call(*inputs)
    assert MLX.array_equal(result, expected)
    
    # Test with array of inputs
    fun_array = lambda do |arrs|
      result = arrs[0]
      arrs[1..-1].each do |arr|
        result = result + arr
      end
      result
    end
    
    cfun = MLX.compile(fun_array)
    
    arrs = (0...100).map { |i| MLX.array(i) }
    expected = arrs.reduce { |sum, x| sum + x }
    
    result = cfun.call(arrs)
    assert MLX.array_equal(result, expected)
  end
  
  def test_compile_many_outputs
    # Test compilation with multiple outputs
    fun = lambda do |arr|
      results = []
      10.times do |i|
        results << arr * i
      end
      results
    end
    
    cfun = MLX.compile(fun)
    
    arr = MLX.array([1.0, 2.0, 3.0])
    results = cfun.call(arr)
    
    assert_equal 10, results.length
    
    expected = (0...10).map { |i| arr * i }
    
    results.each_with_index do |result, i|
      assert MLX.array_equal(result, expected[i])
    end
  end
end 