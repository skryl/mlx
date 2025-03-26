require_relative 'mlx_test_case'

class TestTree < MLXTestCase
  def setup
    super
    
    # Create sample tree structures for testing
    @tree1 = {
      "a" => MLX.array([1, 2, 3]),
      "b" => {
        "c" => MLX.array([4, 5, 6]),
        "d" => MLX.array([7, 8, 9])
      },
      "e" => [
        MLX.array([10, 11, 12]),
        MLX.array([13, 14, 15])
      ]
    }
    
    @tree2 = {
      "a" => MLX.array([2, 3, 4]),
      "b" => {
        "c" => MLX.array([5, 6, 7]),
        "d" => MLX.array([8, 9, 10])
      },
      "e" => [
        MLX.array([11, 12, 13]),
        MLX.array([14, 15, 16])
      ]
    }
  end
  
  def test_tree_map
    # Apply a function to each array in the tree
    doubled = MLX::Utils.tree_map(->(x) { 
      x.is_a?(MLX::Array) ? x * 2 : x
    }, @tree1)
    
    # Check the results
    doubled_a = doubled["a"]
    doubled_bc = doubled["b"]["c"]
    doubled_bd = doubled["b"]["d"]
    doubled_e0 = doubled["e"][0]
    doubled_e1 = doubled["e"][1]
    
    assert_array_equal(doubled_a, [2, 4, 6])
    assert_array_equal(doubled_bc, [8, 10, 12])
    assert_array_equal(doubled_bd, [14, 16, 18])
    assert_array_equal(doubled_e0, [20, 22, 24])
    assert_array_equal(doubled_e1, [26, 28, 30])
  end
  
  def test_tree_map_with_path
    # Apply a function that uses path information
    path_aware = MLX::Utils.tree_map_with_path(->(path, x) {
      if x.is_a?(MLX::Array)
        # Multiply by 2 only if in the "b" subtree
        path.include?("b") ? x * 2 : x
      else
        x
      end
    }, @tree1)
    
    # Check the results
    path_a = path_aware["a"]
    path_bc = path_aware["b"]["c"]
    path_bd = path_aware["b"]["d"]
    path_e0 = path_aware["e"][0]
    
    assert_array_equal(path_a, [1, 2, 3]) # unchanged
    assert_array_equal(path_bc, [8, 10, 12]) # doubled
    assert_array_equal(path_bd, [14, 16, 18]) # doubled
    assert_array_equal(path_e0, [10, 11, 12]) # unchanged
  end
  
  def test_tree_flatten
    # Flatten the tree into a list
    flat = MLX::Utils.tree_flatten(@tree1)
    
    # Check we have the right number of arrays
    assert_equal 5, flat.length
    
    # Check the paths and values
    paths_and_values = flat.map { |p, v| [p, v] }
    
    assert_includes paths_and_values.map(&:first), "a"
    assert_includes paths_and_values.map(&:first), "b.c"
    assert_includes paths_and_values.map(&:first), "b.d"
    assert_includes paths_and_values.map(&:first), "e.0"
    assert_includes paths_and_values.map(&:first), "e.1"
    
    # Find the value for a specific path
    a_val = paths_and_values.find { |p, _| p == "a" }[1]
    assert_array_equal(a_val, [1, 2, 3])
  end
  
  def test_tree_unflatten
    # Flatten the tree
    flat = MLX::Utils.tree_flatten(@tree1)
    
    # Create a template with the same structure but nil values
    template = MLX::Utils.tree_map(->(x) { 
      x.is_a?(MLX::Array) ? nil : x
    }, @tree1)
    
    # Unflatten back to original structure
    reconstructed = MLX::Utils.tree_unflatten(flat)
    
    # Check the structure and values
    assert reconstructed.key?("a")
    assert reconstructed.key?("b")
    assert reconstructed["b"].key?("c")
    assert reconstructed["b"].key?("d")
    assert reconstructed.key?("e")
    assert_equal 2, reconstructed["e"].length
    
    # Check specific values
    assert_array_equal(reconstructed["a"], [1, 2, 3])
    assert_array_equal(reconstructed["b"]["c"], [4, 5, 6])
    assert_array_equal(reconstructed["b"]["d"], [7, 8, 9])
    assert_array_equal(reconstructed["e"][0], [10, 11, 12])
    assert_array_equal(reconstructed["e"][1], [13, 14, 15])
  end
  
  def test_tree_reduce
    # Compute the sum of all array elements
    sum = MLX::Utils.tree_reduce(
      ->(acc, x) { 
        if x.is_a?(MLX::Array)
          acc + MLX.sum(x).item
        else
          acc
        end
      },
      @tree1,
      0.0
    )
    
    # Expected sum: 1+2+3 + 4+5+6 + 7+8+9 + 10+11+12 + 13+14+15 = 120
    assert_equal 120, sum
  end
  
  def test_tree_merge
    # Merge two trees by adding arrays
    merged = MLX::Utils.tree_merge(
      @tree1,
      @tree2,
      ->(x, y) {
        if x.is_a?(MLX::Array) && y.is_a?(MLX::Array)
          x + y
        else
          y
        end
      }
    )
    
    # Check the results
    assert_array_equal(merged["a"], [3, 5, 7]) # 1+2, 2+3, 3+4
    assert_array_equal(merged["b"]["c"], [9, 11, 13]) # 4+5, 5+6, 6+7
    assert_array_equal(merged["b"]["d"], [15, 17, 19]) # 7+8, 8+9, 9+10
    assert_array_equal(merged["e"][0], [21, 23, 25]) # 10+11, 11+12, 12+13
    assert_array_equal(merged["e"][1], [27, 29, 31]) # 13+14, 14+15, 15+16
  end
  
  def test_gradient_clipping
    # Create a simple model structure with "gradients"
    grads = {
      "layer1" => {
        "weight" => MLX.random.normal([5, 5]) * 10,  # Large gradients
        "bias" => MLX.random.normal([5]) * 10
      },
      "layer2" => {
        "weight" => MLX.random.normal([5, 5]) * 10,
        "bias" => MLX.random.normal([5]) * 10
      }
    }
    
    # Calculate the norm
    grad_norm = MLX::Utils.global_norm(grads)
    
    # Clip gradients
    clipped_grads = MLX::Utils.clip_by_norm(grads, 5.0)
    
    # Check the norm of clipped gradients
    clipped_norm = MLX::Utils.global_norm(clipped_grads)
    
    # Allow for small floating point differences
    assert_in_delta 5.0, clipped_norm, 1e-5
    
    # Clip by value
    max_value = 1.0
    value_clipped = MLX::Utils.clip_by_value(grads, -max_value, max_value)
    
    # Check that all values are within range
    MLX::Utils.tree_map(->(x) {
      if x.is_a?(MLX::Array)
        assert MLX.all(x <= max_value).item
        assert MLX.all(x >= -max_value).item
      end
    }, value_clipped)
  end
end 