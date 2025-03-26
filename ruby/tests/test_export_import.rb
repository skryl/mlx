require_relative 'mlx_test_case'
require 'tmpdir'
require 'fileutils'

class TestExportImport < MLXTestCase
  def setup
    # Set random seed for reproducibility
    MLX.random.seed(42)
    
    # Create a temporary directory for tests
    @test_dir = Dir.mktmpdir("mlx_test")
  end
  
  def teardown
    # Clean up temporary directory
    FileUtils.remove_entry(@test_dir)
  end
  
  def test_basic_export_import
    path = File.join(@test_dir, "fn.mlxfn")
    
    # Function with no inputs
    fun = lambda do
      MLX.zeros([3, 3])
    end
    
    MLX.export_function(path, fun)
    imported = MLX.import_function(path)
    
    expected = fun.call
    out = imported.call.first
    assert MLX.array_equal(out, expected)
    
    # Simple function with inputs
    fun = lambda do |x|
      MLX.abs(MLX.sin(x))
    end
    
    inputs = MLX.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    MLX.export_function(path, fun, inputs)
    imported = MLX.import_function(path)
    
    expected = fun.call(inputs)
    out = imported.call(inputs).first
    assert MLX.allclose(out, expected)
    
    # Inputs in a list or array
    fun = lambda do |x|
      x = MLX.abs(MLX.sin(x))
      x
    end
    
    MLX.export_function(path, fun, [inputs])
    imported = MLX.import_function(path)
    
    expected = fun.call(inputs)
    out = imported.call([inputs]).first
    assert MLX.allclose(out, expected)
    
    out = imported.call(inputs).first
    assert MLX.allclose(out, expected)
    
    MLX.export_function(path, fun, [inputs])
    imported = MLX.import_function(path)
    out = imported.call([inputs]).first
    assert MLX.allclose(out, expected)
    
    # Outputs in an array
    fun = lambda do |x|
      [MLX.abs(MLX.sin(x))]
    end
    
    MLX.export_function(path, fun, inputs)
    imported = MLX.import_function(path)
    out = imported.call(inputs).first
    assert MLX.allclose(out, expected)
    
    # Check throws on invalid inputs / outputs
    fun = lambda do |x|
      MLX.abs(x)
    end
    
    assert_raises(ArgumentError) do
      MLX.export_function(path, fun, "hi")
    end
    
    assert_raises(ArgumentError) do
      MLX.export_function(path, fun, MLX.array(1.0), "hi")
    end
    
    fun = lambda do |x|
      MLX.abs(x[0][0])
    end
    
    assert_raises(ArgumentError) do
      MLX.export_function(path, fun, [[MLX.array(1.0)]])
    end
    
    fun = lambda do
      [MLX.zeros([3, 3]), 1]
    end
    
    assert_raises(ArgumentError) do
      MLX.export_function(path, fun)
    end
    
    fun = lambda do
      [MLX.zeros([3, 3]), [MLX.zeros([3, 3])]]
    end
    
    assert_raises(ArgumentError) do
      MLX.export_function(path, fun)
    end
    
    fun = lambda do |x, y|
      x + y
    end
    
    MLX.export_function(path, fun, MLX.array(1.0), MLX.array(1.0))
    imported = MLX.import_function(path)
    
    assert_raises(ArgumentError) do
      imported.call(MLX.array(1.0), 1.0)
    end
    
    assert_raises(ArgumentError) do
      imported.call(MLX.array(1.0), MLX.array(1.0), MLX.array(1.0))
    end
    
    assert_raises(ArgumentError) do
      imported.call(MLX.array(1.0), [MLX.array(1.0)])
    end
  end
  
  def test_export_random_sample
    path = File.join(@test_dir, "fn.mlxfn")
    
    MLX.random.seed(5)
    
    fun = lambda do
      MLX.random.uniform(shape: [3])
    end
    
    MLX.export_function(path, fun)
    imported = MLX.import_function(path)
    
    out = imported.call.first
    
    MLX.random.seed(5)
    expected = fun.call
    
    assert MLX.array_equal(out, expected)
  end
  
  def test_export_with_kwargs
    path = File.join(@test_dir, "fn.mlxfn")
    
    fun = lambda do |x, z: nil|
      out = x
      out += z unless z.nil?
      out
    end
    
    x = MLX.array([1, 2, 3])
    y = MLX.array([1, 1, 0])
    z = MLX.array([2, 2, 2])
    
    MLX.export_function(path, fun, [x], {z: z})
    imported_fun = MLX.import_function(path)
    
    assert_raises(ArgumentError) do
      imported_fun.call(x, z)
    end
    
    assert_raises(ArgumentError) do
      imported_fun.call(x, y: z)
    end
    
    assert_raises(ArgumentError) do
      imported_fun.call([x], {y: z})
    end
    
    out = imported_fun.call(x, z: z).first
    assert MLX.array_equal(out, MLX.array([3, 4, 5]))
    
    out = imported_fun.call([x], {z: z}).first
    assert MLX.array_equal(out, MLX.array([3, 4, 5]))
    
    MLX.export_function(path, fun, x, z: z)
    imported_fun = MLX.import_function(path)
    out = imported_fun.call(x, z: z).first
    assert MLX.array_equal(out, MLX.array([3, 4, 5]))
    
    out = imported_fun.call([x], {z: z}).first
    assert MLX.array_equal(out, MLX.array([3, 4, 5]))
    
    # Only specify kwargs
    MLX.export_function(path, fun, x: x, z: z)
    imported_fun = MLX.import_function(path)
    
    assert_raises(ArgumentError) do
      out = imported_fun.call(x, z: z).first
    end
    
    out = imported_fun.call(x: x, z: z).first
    assert MLX.array_equal(out, MLX.array([3, 4, 5]))
    
    out = imported_fun.call({x: x, z: z}).first
    assert MLX.array_equal(out, MLX.array([3, 4, 5]))
  end
  
  def test_export_variable_inputs
    path = File.join(@test_dir, "fn.mlxfn")
    
    fun = lambda do |x, y, z: nil|
      out = x + y
      out += z unless z.nil?
      out
    end
    
    # Use exporter context manager
    exporter = MLX.exporter(path, fun)
    exporter.call(MLX.array([1, 2, 3]), MLX.array([1, 1, 1]))
    exporter.call(MLX.array([1, 2, 3]), MLX.array([1, 1, 1]), z: MLX.array([2]))
    
    assert_raises(RuntimeError) do
      exporter.call(MLX.array([1, 2, 3, 4]), MLX.array([1, 1, 1, 1]))
    end
    
    exporter.close
    
    imported_fun = MLX.import_function(path)
    out = imported_fun.call(MLX.array([1, 2, 3]), MLX.array([1, 1, 1])).first
    assert MLX.array_equal(out, MLX.array([2, 3, 4]))
    
    out = imported_fun.call(MLX.array([1, 2, 3]), MLX.array([1, 1, 1]), z: MLX.array([2])).first
    assert MLX.array_equal(out, MLX.array([4, 5, 6]))
    
    assert_raises(ArgumentError) do
      imported_fun.call(MLX.array([1, 2, 3, 4]), MLX.array([1, 1, 1, 1]))
    end
    
    # A function with a large constant
    constant = MLX.zeros([16, 2048])
    MLX.eval(constant)
    
    fun = lambda do |*args|
      constant + args.reduce(0) { |sum, arg| sum + arg }
    end
    
    # Use block form of exporter
    MLX.exporter(path, fun) do |exporter|
      5.times do |i|
        exporter.call(*Array.new(i) { MLX.array(1) })
      end
    end
    
    # Check the exported file size < constant size + small amount
    constants_size = constant.nbytes + 8192
    assert File.size(path) < constants_size
  end
  
  def test_leaks
    path = File.join(@test_dir, "fn.mlxfn")
    
    # Skip memory tests if not on Metal
    if MLX.metal.available?
      mem_pre = MLX.get_active_memory
    else
      mem_pre = 0
      skip "Memory tests are only meaningful with Metal"
    end
    
    # Create a function in a local scope
    outer = lambda do
      f = lambda do |x|
        MLX.sin(x) * MLX.sin(x)
      end
      
      MLX.export_function(path, f, MLX.array(1.0))
      imported = MLX.import_function(path)
      imported.call(MLX.array(1.0))
    end
    
    outer.call
    
    # Force garbage collection
    GC.start
    
    # Check memory usage hasn't changed significantly
    if MLX.metal.available?
      mem_post = MLX.get_active_memory
      assert_in_delta mem_pre, mem_post, mem_pre * 0.1  # Allow 10% variance
    end
  end
end 