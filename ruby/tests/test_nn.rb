require_relative 'mlx_test_case'

class TestNN < MLXTestCase
  def test_module_basics
    # Create a simple module
    class SimpleModule < MLX::NN::Module
      def initialize
        super()
        @weight = MLX.array([1.0, 2.0, 3.0])
        @bias = MLX.array([0.5])
        
        # Register parameters
        register_parameter("weight", @weight)
        register_parameter("bias", @bias)
      end
      
      def forward(x)
        MLX.dot(x, @weight) + @bias
      end
    end
    
    # Instantiate the module
    mod = SimpleModule.new
    
    # Check parameters
    params = mod.parameters
    assert_equal 2, params.size
    assert params.key?("weight")
    assert params.key?("bias")
    assert_array_equal(params["weight"], [1.0, 2.0, 3.0])
    assert_array_equal(params["bias"], [0.5])
    
    # Test forward pass
    x = MLX.array([2.0, 3.0, 4.0])
    output = mod.call(x)
    
    # Expected: 2*1 + 3*2 + 4*3 + 0.5 = 2 + 6 + 12 + 0.5 = 20.5
    assert_array_equal(output, [20.5])
  end
  
  def test_sequential
    # Create a sequential model
    seq = MLX::NN::Sequential.new(
      MLX::NN::Layers::Linear.new(3, 4),
      MLX::NN::Layers::ReLU.new,
      MLX::NN::Layers::Linear.new(4, 1)
    )
    
    # Check structure
    assert_equal 3, seq.modules.length
    assert seq.modules[0].is_a?(MLX::NN::Layers::Linear)
    assert seq.modules[1].is_a?(MLX::NN::Layers::ReLU)
    assert seq.modules[2].is_a?(MLX::NN::Layers::Linear)
    
    # Test forward pass
    x = MLX.array([1.0, 2.0, 3.0])
    output = seq.call(x)
    
    # Output should be a scalar
    assert_equal 1, output.size
  end
  
  def test_linear
    # Test linear layer
    linear = MLX::NN::Layers::Linear.new(3, 2)
    
    # Check parameters
    params = linear.parameters
    assert params.key?("weight")
    assert params.key?("bias")
    assert_equal [2, 3], params["weight"].shape
    assert_equal [2], params["bias"].shape
    
    # Test forward pass
    x = MLX.array([1.0, 2.0, 3.0])
    output = linear.call(x)
    
    # Output should have the right shape
    assert_equal [2], output.shape
  end
  
  def test_activation_functions
    # Test ReLU
    relu = MLX::NN::Layers::ReLU.new
    x = MLX.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    relu_out = relu.call(x)
    assert_array_equal(relu_out, [0.0, 0.0, 0.0, 1.0, 2.0])
    
    # Test Sigmoid
    sigmoid = MLX::NN::Layers::Sigmoid.new
    sigmoid_out = sigmoid.call(x)
    expected = MLX.array([
      1.0 / (1.0 + Math.exp(2.0)),
      1.0 / (1.0 + Math.exp(1.0)),
      0.5,
      1.0 / (1.0 + Math.exp(-1.0)),
      1.0 / (1.0 + Math.exp(-2.0))
    ])
    assert_array_equal(sigmoid_out, expected)
    
    # Test Tanh
    tanh = MLX::NN::Layers::Tanh.new
    tanh_out = tanh.call(x)
    expected = MLX.array([Math.tanh(-2.0), Math.tanh(-1.0), Math.tanh(0.0), 
                           Math.tanh(1.0), Math.tanh(2.0)])
    assert_array_equal(tanh_out, expected)
  end
  
  def test_loss_functions
    # Test cross entropy loss
    logits = MLX.array([[0.1, 0.5, 2.0], [0.2, 1.5, 0.3]])
    targets = MLX.array([2, 1])
    
    loss = MLX::NN.losses.cross_entropy(logits, targets)
    
    # Loss should be a scalar
    assert_equal 0, loss.ndim
  end
  
  def test_conv2d
    # Test 2D convolution
    conv = MLX::NN::Layers::Conv2d.new(3, 2, kernel_size: 3, stride: 1, padding: 1)
    
    # Check parameters
    params = conv.parameters
    assert params.key?("weight")
    assert params.key?("bias")
    assert_equal [2, 3, 3, 3], params["weight"].shape
    assert_equal [2], params["bias"].shape
    
    # Test forward pass
    x = MLX.random.normal([1, 3, 28, 28])
    output = conv.call(x)
    
    # Output should have the right shape
    assert_equal [1, 2, 28, 28], output.shape
  end
  
  def test_pooling
    # Test 2D max pooling
    pool = MLX::NN::Layers::MaxPool2d.new(kernel_size: 2, stride: 2)
    
    # Test forward pass
    x = MLX.random.normal([1, 3, 28, 28])
    output = pool.call(x)
    
    # Output should have the right shape
    assert_equal [1, 3, 14, 14], output.shape
  end
  
  def test_dropout
    # Test dropout layer
    drop = MLX::NN::Layers::Dropout.new(p: 0.5)
    
    # Set deterministic mode for testing
    MLX.deterministic!(true)
    
    # Test forward pass in eval mode
    x = MLX.ones([10, 10])
    drop.eval!
    output = drop.call(x)
    
    # In eval mode, dropout doesn't drop anything
    assert_array_equal(output, x)
    
    # Reset deterministic mode
    MLX.deterministic!(false)
  end
  
  def test_embedding
    # Test embedding layer
    embed = MLX::NN::Layers::Embedding.new(10, 5)
    
    # Check parameters
    params = embed.parameters
    assert params.key?("weight")
    assert_equal [10, 5], params["weight"].shape
    
    # Test forward pass
    x = MLX.array([0, 2, 5])
    output = embed.call(x)
    
    # Output should have the right shape
    assert_equal [3, 5], output.shape
  end
  
  def test_batch_norm
    # Test batch normalization
    bn = MLX::NN::Layers::BatchNorm2d.new(3)
    
    # Check parameters
    params = bn.parameters
    assert params.key?("weight")
    assert params.key?("bias")
    assert_equal [3], params["weight"].shape
    assert_equal [3], params["bias"].shape
    
    # Test forward pass
    x = MLX.random.normal([2, 3, 4, 4])
    output = bn.call(x)
    
    # Output should have the same shape as input
    assert_equal x.shape, output.shape
  end
  
  def test_layer_norm
    # Test layer normalization
    ln = MLX::NN::Layers::LayerNorm.new(10)
    
    # Check parameters
    params = ln.parameters
    assert params.key?("weight")
    assert params.key?("bias")
    assert_equal [10], params["weight"].shape
    assert_equal [10], params["bias"].shape
    
    # Test forward pass
    x = MLX.random.normal([5, 10])
    output = ln.call(x)
    
    # Output should have the same shape as input
    assert_equal x.shape, output.shape
  end
  
  def test_multi_head_attention
    # Test multi-head attention
    mha = MLX::NN::Layers::MultiheadAttention.new(
      embed_dim: 12,
      num_heads: 3
    )
    
    # Check parameters
    params = mha.parameters
    assert params.key?("q_proj.weight")
    assert params.key?("k_proj.weight")
    assert params.key?("v_proj.weight")
    assert params.key?("out_proj.weight")
    
    # Test forward pass
    q = MLX.random.normal([2, 4, 12])
    k = MLX.random.normal([2, 6, 12])
    v = MLX.random.normal([2, 6, 12])
    
    output = mha.call(q, k, v)
    
    # Output should match query sequence length
    assert_equal [2, 4, 12], output.shape
  end
  
  def test_positional_encoding
    # Test sinusoidal positional encoding
    pe = MLX::NN::Layers::SinusoidalPositionalEncoding.new(d_model: 8, max_len: 10)
    
    # Test forward pass
    x = MLX.random.normal([2, 4, 8])
    output = pe.call(x)
    
    # Output should have the same shape as input
    assert_equal x.shape, output.shape
  end
  
  def test_initialization
    # Test Xavier uniform initialization
    w = MLX::NN.init.xavier_uniform([10, 5])
    assert_equal [10, 5], w.shape
    
    # Test Xavier normal initialization
    w = MLX::NN.init.xavier_normal([5, 10])
    assert_equal [5, 10], w.shape
    
    # Test constant initialization
    w = MLX::NN.init.constant([3, 3], 1.0)
    assert_array_equal(w, MLX.ones([3, 3]))
  end
  
  def test_parameter_groups
    # Create a simple network
    class SimpleNet < MLX::NN::Module
      attr_reader :fc1, :fc2
      
      def initialize
        super()
        @fc1 = MLX::NN::Layers::Linear.new(10, 20)
        @fc2 = MLX::NN::Layers::Linear.new(20, 1)
      end
      
      def forward(x)
        x = MLX::NN.activations.relu(@fc1.call(x))
        @fc2.call(x)
      end
    end
    
    net = SimpleNet.new
    
    # Get all parameters
    all_params = net.parameters
    
    # Check parameter count
    assert_equal 4, all_params.size  # fc1.weight, fc1.bias, fc2.weight, fc2.bias
    
    # Get parameter groups
    param_groups = net.parameters_by_prefix
    
    # Check grouping
    assert param_groups.key?("fc1")
    assert param_groups.key?("fc2")
    assert_equal 2, param_groups["fc1"].size
    assert_equal 2, param_groups["fc2"].size
  end
  
  def test_quantized_linear
    # Test quantized linear layer, if available
    if defined?(MLX::NN::Layers::QuantizedLinear)
      # Create a regular linear layer first
      linear = MLX::NN::Layers::Linear.new(10, 5)
      
      # Quantize it
      quantized = MLX::NN::Layers::QuantizedLinear.from_linear(
        linear, 
        weight_params: { group_size: 4, bits: 4 }
      )
      
      # Check parameters
      params = quantized.parameters
      assert params.key?("qweight")
      assert params.key?("scales")
      
      # Test forward pass
      x = MLX.random.normal([3, 10])
      output = quantized.call(x)
      
      # Output should have the right shape
      assert_equal [3, 5], output.shape
    end
  end
  
  def test_distributed_linear
    # Test distributed linear layer, if available
    if defined?(MLX::NN::Layers::ShardedToAllLinear)
      # Create a regular linear layer first
      linear = MLX::NN::Layers::Linear.new(10, 5)
      
      # Initialize distributed environment
      world = MLX::Distributed.init
      
      # Create distributed linear layer
      distributed = MLX::NN::Layers.sharded_to_all_linear(
        linear,
        group: world
      )
      
      # Check parameters
      params = distributed.parameters
      assert params.key?("weight")
      
      # Test forward pass
      x = MLX.random.normal([3, 10])
      output = distributed.call(x)
      
      # Output should have the right shape
      assert_equal [3, 5], output.shape
    end
  end
end 