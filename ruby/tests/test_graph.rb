require_relative 'mlx_test_case'
require 'stringio'

class TestGraph < MLXTestCase
  def setup
    # Set random seed for reproducibility
    MLX.random.seed(42)
  end
  
  def test_to_dot
    # Test that export_to_dot works with different cases
    
    # Simple case with a single array
    a = MLX.array(1.0)
    f = StringIO.new
    MLX.export_to_dot(f, a)
    f.rewind
    content = f.read
    assert content.length > 0, "Graph output should not be empty"
    
    # Test with a binary operation
    b = MLX.array(2.0)
    c = a + b
    f = StringIO.new
    MLX.export_to_dot(f, c)
    f.rewind
    content = f.read
    assert content.length > 0, "Graph output should not be empty"
    
    # Test with a multi-output operation
    c = MLX.divmod(a, b)
    f = StringIO.new
    MLX.export_to_dot(f, *c)
    f.rewind
    content = f.read
    assert content.length > 0, "Graph output should not be empty"
  end
  
  def test_graph_with_complex_operations
    # Test graph export with more complex operations
    
    # Create a small computation graph
    a = MLX.random.normal(shape: [3, 4])
    b = MLX.random.normal(shape: [4, 5])
    c = MLX.matmul(a, b)
    d = MLX.sin(c)
    e = MLX.exp(d)
    f = e + 1.0
    
    # Export the graph
    output = StringIO.new
    MLX.export_to_dot(output, f)
    output.rewind
    content = output.read
    
    # Verify the output
    assert content.length > 0, "Graph output should not be empty"
    assert content.include?("digraph"), "Graph should start with 'digraph'"
    
    # Test with multiple outputs
    output = StringIO.new
    MLX.export_to_dot(output, c, d, e, f)
    output.rewind
    content = output.read
    
    # Verify the output
    assert content.length > 0, "Graph output should not be empty"
    assert content.include?("digraph"), "Graph should start with 'digraph'"
  end
  
  def test_graph_with_array_transformations
    # Test graph export with array transformations
    
    # Create arrays and apply transformations
    a = MLX.arange(16).reshape([4, 4])
    b = MLX.transpose(a)
    c = MLX.reshape(b, [2, 8])
    d = MLX.concatenate([c, c], axis: 1)
    
    # Export the graph
    output = StringIO.new
    MLX.export_to_dot(output, d)
    output.rewind
    content = output.read
    
    # Verify the output
    assert content.length > 0, "Graph output should not be empty"
    assert content.include?("digraph"), "Graph should start with 'digraph'"
  end
  
  def test_graph_with_mathematical_operations
    # Test graph export with various mathematical operations
    
    a = MLX.array([1.0, 2.0, 3.0, 4.0])
    b = MLX.log(a)
    c = MLX.sqrt(a)
    d = MLX.exp(a)
    e = MLX.add(b, c)
    f = MLX.divide(e, d)
    
    # Export the graph
    output = StringIO.new
    MLX.export_to_dot(output, f)
    output.rewind
    content = output.read
    
    # Verify the output
    assert content.length > 0, "Graph output should not be empty"
    assert content.include?("digraph"), "Graph should start with 'digraph'"
  end
  
  def test_graph_with_reduction_operations
    # Test graph export with reduction operations
    
    a = MLX.arange(9).reshape([3, 3])
    b = MLX.sum(a, axis: 0)
    c = MLX.mean(a, axis: 1)
    d = MLX.max(a)
    e = MLX.argmin(a, axis: 0)
    
    # Export the graph
    output = StringIO.new
    MLX.export_to_dot(output, b, c, d, e)
    output.rewind
    content = output.read
    
    # Verify the output
    assert content.length > 0, "Graph output should not be empty"
    assert content.include?("digraph"), "Graph should start with 'digraph'"
  end
end 