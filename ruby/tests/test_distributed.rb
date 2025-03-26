require_relative 'mlx_test_case'

class TestDistributed < MLXTestCase
  def setup
    super
    
    # Store the original distributed state
    @original_world = MLX::Distributed.world if MLX::Distributed.is_initialized?
  end
  
  def teardown
    # Restore the original distributed state
    if defined?(@original_world) && @original_world
      # Reset the distributed environment
      MLX::Distributed.reset
    end
    
    super
  end
  
  def test_distributed_initialization
    # Initialize the distributed environment
    world = MLX::Distributed.init
    
    # Basic properties
    assert MLX::Distributed.is_initialized?
    assert_equal 1, world.size
    assert_equal 0, world.rank
    assert world.is_root?
    
    # Check if we have the right methods
    assert_respond_to MLX::Distributed, :all_sum
    assert_respond_to MLX::Distributed, :all_gather
  end
  
  def test_distributed_operations
    # Initialize the distributed environment
    world = MLX::Distributed.init
    
    # Create arrays to test with
    x = MLX.array([1.0, 2.0, 3.0, 4.0])
    
    # Test all_sum - in single process, should return the same array
    result = MLX::Distributed.all_sum(x)
    assert_array_equal(result, x)
    
    # Test all_gather - in single process, should return an array with one element
    result = MLX::Distributed.all_gather(x)
    assert_equal 1, result.length
    assert_array_equal(result[0], x)
  end
  
  def test_distributed_subgroups
    # Initialize the distributed environment
    world = MLX::Distributed.init
    
    # Since we're in a single-process environment, subgroups behave the same
    if world.respond_to?(:subgroup)
      subgroup = world.subgroup([0], name: "test_subgroup")
      assert_equal 1, subgroup.size
      assert_equal 0, subgroup.rank
      assert_equal "test_subgroup", subgroup.name
    end
  end
  
  def test_distributed_model_layers
    # Skip if distributed layers aren't available
    skip unless defined?(MLX::NN::Layers.shard_inplace)
    
    # Initialize the distributed environment
    world = MLX::Distributed.init
    
    # Create a model
    model = MLX::NN::Sequential.new(
      MLX::NN::Layers::Linear.new(10, 20),
      MLX::NN::Layers::ReLU.new,
      MLX::NN::Layers::Linear.new(20, 5)
    )
    
    # Shard the last layer
    last_layer = model.modules[2]
    distributed_layer = MLX::NN::Layers.shard_linear(
      last_layer,
      "all-to-sharded",
      group: world
    )
    
    # Replace the last layer with the distributed version
    model.modules[2] = distributed_layer
    
    # Run a forward pass
    x = MLX.random.normal([2, 10])
    output = model.call(x)
    
    # Output should have the expected shape
    assert_equal [2, 5], output.shape
  end
  
  def test_distributed_run_module
    # Skip if the distributed_run module isn't available
    skip unless defined?(MLX::DistributedRun)
    
    # Test that we can parse a hostfile
    hostfile = Tempfile.new(["mlx_hosts", ".json"])
    begin
      # Create a sample hostfile
      hostfile.write(<<~JSON)
        [
          {"ssh": "localhost", "ips": ["127.0.0.1"]},
          {"ssh": "localhost", "ips": ["127.0.0.1"]}
        ]
      JSON
      hostfile.close
      
      # Parse the hostfile
      hosts = MLX::DistributedRun.parse_hostfile(hostfile.path)
      
      # Check the parsed hosts
      assert_equal 2, hosts.length
      assert_equal 0, hosts[0].rank
      assert_equal 1, hosts[1].rank
      assert_equal "localhost", hosts[0].ssh_hostname
      assert_equal "localhost", hosts[1].ssh_hostname
      assert_equal ["127.0.0.1"], hosts[0].ips
      assert_equal ["127.0.0.1"], hosts[1].ips
    ensure
      hostfile.unlink
    end
    
    # Test host list parsing
    hosts = MLX::DistributedRun.parse_hostlist("localhost,localhost", 1)
    assert_equal 2, hosts.length
    assert_equal 0, hosts[0].rank
    assert_equal 1, hosts[1].rank
    assert_equal "localhost", hosts[0].ssh_hostname
    assert_equal "localhost", hosts[1].ssh_hostname
  end
  
  def test_distributed_gradient_averaging
    # Skip unless we have distributed gradient utilities
    skip unless defined?(MLX::NN::Utils) && MLX::NN::Utils.respond_to?(:average_gradients)
    
    # Initialize the distributed environment
    world = MLX::Distributed.init
    
    # Create a model
    model = MLX::NN::Sequential.new(
      MLX::NN::Layers::Linear.new(10, 5),
      MLX::NN::Layers::ReLU.new,
      MLX::NN::Layers::Linear.new(5, 1)
    )
    
    # Create random data
    x = MLX.random.normal([8, 10])
    y = MLX.random.normal([8, 1])
    
    # Define a loss function
    loss_fn = lambda do |model, x, y|
      preds = model.call(x)
      MLX.mean(MLX.square(preds - y))
    end
    
    # Compute gradients
    _, grads = MLX.value_and_grad(model, loss_fn).call(model, x, y)
    
    # Average gradients across distributed processes
    # In a single-process environment, this should return the same gradients
    avg_grads = MLX::NN::Utils.average_gradients(grads, group: world)
    
    # Check that gradients are the same (since we only have one process)
    grads.each do |name, grad|
      assert_array_equal(grad, avg_grads[name])
    end
  end
  
  def test_distributed_in_place_sharding
    # Skip if distributed sharding isn't available
    skip unless defined?(MLX::NN::Layers.shard_inplace)
    
    # Initialize the distributed environment
    world = MLX::Distributed.init
    
    # Create a model
    model = MLX::NN::Sequential.new(
      MLX::NN::Layers::Linear.new(10, 20),
      MLX::NN::Layers::ReLU.new,
      MLX::NN::Layers::Linear.new(20, 5)
    )
    
    # Store original parameters
    orig_params = model.parameters.transform_values(&:copy)
    
    # Apply in-place sharding to last layer
    MLX::NN::Layers.shard_inplace(
      model.modules[2],
      "all-to-sharded",
      group: world
    )
    
    # Check that parameters have changed shape
    model.modules[2].parameters.each do |name, param|
      # In a single process setup, the shape should remain the same
      # since we're not actually distributing across multiple processes
      assert_equal orig_params["2.#{name}"].shape, param.shape
    end
  end
  
  def test_tree_map_merge_with_distributed
    # Initialize distributed environment
    world = MLX::Distributed.init
    
    # Create two trees with different process-specific values
    tree1 = {
      "layer1" => { "weight" => MLX.array([1.0, 2.0, 3.0]) * world.rank },
      "layer2" => { "weight" => MLX.array([4.0, 5.0, 6.0]) * world.rank }
    }
    
    tree2 = {
      "layer1" => { "weight" => MLX.array([7.0, 8.0, 9.0]) },
      "layer2" => { "weight" => MLX.array([10.0, 11.0, 12.0]) }
    }
    
    # Merge the trees
    merged = MLX::Utils.tree_merge(tree1, tree2) do |a, b|
      if a.is_a?(MLX::Array) && b.is_a?(MLX::Array)
        a + b
      else
        b
      end
    end
    
    # Check the results - in a single process environment, rank is 0
    # so the values from tree1 should be unchanged
    assert_array_equal(merged["layer1"]["weight"], [7.0, 8.0, 9.0])
    assert_array_equal(merged["layer2"]["weight"], [10.0, 11.0, 12.0])
  end
end 