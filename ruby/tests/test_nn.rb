require_relative 'mlx_test_case'

class TestNN < MLXTestCase
  def setup
    # Set random seed for reproducibility
    MLX.random.seed(42)
    
    # Common dimensions
    @batch_size = 4
    @in_features = 8
    @out_features = 6
    @seq_length = 10
    @embedding_dim = 16
    @vocab_size = 1000
    @num_heads = 4
    @channels_in = 3
    @channels_out = 16
    @height = 32
    @width = 32
  end
  
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
    # Test basic linear layer
    layer = MLX::NN::Layers::Linear.new(3, 2)
    
    # Check parameters
    params = layer.parameters
    assert params.key?("weight")
    assert params.key?("bias")
    assert_equal [2, 3], params["weight"].shape
    assert_equal [2], params["bias"].shape
    
    # Test forward pass
    x = MLX.array([1.0, 2.0, 3.0])
    output = layer.call(x)
    
    # Output should have the right shape
    assert_equal [2], output.shape
    
    # Test linear layer without bias
    layer_no_bias = MLX.nn.Linear(@in_features, @out_features, bias: false)
    assert_nil layer_no_bias.bias
    
    x = MLX.random.normal(shape: [@batch_size, @seq_length, @in_features])
    y_no_bias = layer_no_bias.call(x)
    assert_equal [@batch_size, @seq_length, @out_features], y_no_bias.shape
    
    # Test with custom initialization
    initializer = MLX.nn.initializers.Constant(0.5)
    layer_custom_init = MLX.nn.Linear(@in_features, @out_features, 
                                  weight_initializer: initializer,
                                  bias_initializer: initializer)
    
    assert MLX.allclose(layer_custom_init.weight, MLX.full([
      @out_features, @in_features], 0.5))
    assert MLX.allclose(layer_custom_init.bias, MLX.full([@out_features], 0.5))
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
    
    # Test Leaky ReLU
    leaky_relu = MLX::NN::Layers::LeakyReLU.new(negative_slope: 0.1)
    leaky_out = leaky_relu.call(x)
    expected = MLX.array([-0.2, -0.1, 0.0, 1.0, 2.0])
    assert MLX.allclose(leaky_out, expected)
    
    # Test GELU
    gelu = MLX::NN::Layers::GELU.new
    gelu_out = gelu.call(x)
    # Approximate expected values for GELU
    expected = MLX.array([-0.046, -0.159, 0.0, 0.841, 1.954])
    assert MLX.allclose(gelu_out, expected, atol: 0.001)
    
    # Test SiLU/Swish
    silu = MLX::NN::Layers::SiLU.new
    silu_out = silu.call(x)
    expected = x * MLX.sigmoid(x)
    assert MLX.allclose(silu_out, expected)
  end
  
  def test_loss_functions
    # Test cross entropy loss
    logits = MLX.array([[0.1, 0.5, 2.0], [0.2, 1.5, 0.3]])
    targets = MLX.array([2, 1])
    
    loss = MLX::NN.losses.cross_entropy(logits, targets)
    
    # Loss should be a scalar
    assert_equal 0, loss.ndim
  end
  
  def test_conv1d
    # Test 1D convolution layer
    in_channels = 4
    out_channels = 8
    kernel_size = 3
    
    # Create input with shape [batch_size, in_channels, sequence_length]
    x = MLX.random.normal(shape: [@batch_size, in_channels, @seq_length])
    
    # Test with default parameters
    layer = MLX.nn.Conv1d(in_channels, out_channels, kernel_size)
    y = layer.call(x)
    
    # Output shape should be [batch_size, out_channels, sequence_length - kernel_size + 1]
    expected_output_length = @seq_length - kernel_size + 1
    assert_equal [@batch_size, out_channels, expected_output_length], y.shape
    
    # Test with padding
    padding = 1
    layer = MLX.nn.Conv1d(in_channels, out_channels, kernel_size, padding: padding)
    y = layer.call(x)
    
    # Output shape should be [batch_size, out_channels, sequence_length - kernel_size + 1 + 2*padding]
    expected_output_length = @seq_length - kernel_size + 1 + 2 * padding
    assert_equal [@batch_size, out_channels, expected_output_length], y.shape
    
    # Test with stride
    stride = 2
    layer = MLX.nn.Conv1d(in_channels, out_channels, kernel_size, stride: stride)
    y = layer.call(x)
    
    # Output shape should be [batch_size, out_channels, (sequence_length - kernel_size + 1) / stride]
    expected_output_length = (@seq_length - kernel_size + 1) / stride
    assert_equal [@batch_size, out_channels, expected_output_length], y.shape
    
    # Test with dilation
    dilation = 2
    layer = MLX.nn.Conv1d(in_channels, out_channels, kernel_size, dilation: dilation)
    y = layer.call(x)
    
    # Output shape with dilation
    expected_output_length = @seq_length - (kernel_size - 1) * dilation - 1
    assert_equal [@batch_size, out_channels, expected_output_length], y.shape
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
    
    # Test with different parameters
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    
    # Create input with shape [batch_size, in_channels, height, width]
    x = MLX.random.normal(shape: [@batch_size, in_channels, @height, @width])
    
    # Test with stride
    stride = 2
    layer = MLX.nn.Conv2d(in_channels, out_channels, kernel_size, stride: stride)
    y = layer.call(x)
    
    # Output shape with stride
    expected_output_height = (@height - kernel_size + 1) / stride
    expected_output_width = (@width - kernel_size + 1) / stride
    assert_equal [@batch_size, out_channels, expected_output_height, expected_output_width], y.shape
    
    # Test with rectangular kernel
    kernel_size = [3, 5]
    layer = MLX.nn.Conv2d(in_channels, out_channels, kernel_size)
    y = layer.call(x)
    
    # Output shape with rectangular kernel
    expected_output_height = @height - kernel_size[0] + 1
    expected_output_width = @width - kernel_size[1] + 1
    assert_equal [@batch_size, out_channels, expected_output_height, expected_output_width], y.shape
  end
  
  def test_pooling
    # Test 2D max pooling
    pool = MLX::NN::Layers::MaxPool2d.new(kernel_size: 2, stride: 2)
    
    # Test forward pass
    x = MLX.random.normal([1, 3, 28, 28])
    output = pool.call(x)
    
    # Output should have the right shape
    assert_equal [1, 3, 14, 14], output.shape
    
    # Test AvgPool2d
    pool = MLX.nn.AvgPool2d(kernel_size: 2)
    x = MLX.random.normal(shape: [@batch_size, @channels_in, @height, @width])
    y = pool.call(x)
    assert_equal [@batch_size, @channels_in, @height / 2, @width / 2], y.shape
    
    # Test GlobalAvgPool2d
    pool = MLX.nn.GlobalAvgPool2d()
    y = pool.call(x)
    assert_equal [@batch_size, @channels_in, 1, 1], y.shape
    
    # Test GlobalMaxPool2d
    pool = MLX.nn.GlobalMaxPool2d()
    y = pool.call(x)
    assert_equal [@batch_size, @channels_in, 1, 1], y.shape
    
    # Test adaptive pooling
    pool = MLX.nn.AdaptiveAvgPool2d(output_size: [3, 3])
    y = pool.call(x)
    assert_equal [@batch_size, @channels_in, 3, 3], y.shape
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
    
    # Test in training mode
    drop.train!
    output = drop.call(x)
    
    # Check that some elements are dropped
    zeros_count = MLX.sum(output == 0).item
    assert zeros_count > 0
    
    # Reset deterministic mode
    MLX.deterministic!(false)
    
    # Test Dropout2d
    drop = MLX.nn.Dropout2d(p: 0.5)
    x = MLX.random.normal(shape: [@batch_size, @channels_in, @height, @width])
    drop.eval!
    y = drop.call(x)
    assert_equal x.shape, y.shape
    
    # Test with training
    drop.train!
    y = drop.call(x)
    assert_equal x.shape, y.shape
    
    # Some channels should be entirely dropped
    dropout_pattern = MLX.sum(y == 0, dims: [2, 3])
    nonzero_channels = MLX.sum(dropout_pattern > 0).item
    assert nonzero_channels > 0
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
    
    # Test with 2D input (batch)
    x = MLX.array([[1, 2, 3], [4, 5, 6]])
    layer = MLX.nn.Embedding(@vocab_size, @embedding_dim)
    y = layer.call(x)
    
    # Output shape should be [2, 3, embedding_dim]
    assert_equal [2, 3, @embedding_dim], y.shape
    
    # Test embedding lookup (the first row of output should be the first row of the embedding table)
    x = MLX.array([0])
    y = layer.call(x)
    assert MLX.array_equal(y[0], layer.weight[0])
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
    
    # Test BatchNorm1d
    layer = MLX.nn.BatchNorm1d(@in_features)
    x = MLX.random.normal(shape: [@batch_size, @in_features])
    y = layer.call(x)
    assert_equal x.shape, y.shape
    
    # Test BatchNorm3d
    channels = 4
    depth = 8
    layer = MLX.nn.BatchNorm3d(channels)
    x = MLX.random.normal(shape: [@batch_size, channels, depth, @height, @width])
    y = layer.call(x)
    assert_equal x.shape, y.shape
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
    
    # Test without bias
    layer = MLX.nn.LayerNorm([@in_features], bias: false)
    assert_nil layer.bias
    y = layer.call(x)
    assert_equal x.shape, y.shape
    
    # Test with larger normalized shape
    normalized_shape = [@seq_length, @in_features]
    layer = MLX.nn.LayerNorm(normalized_shape)
    
    # Parameter shapes should match the normalized shape
    assert_equal normalized_shape, layer.weight.shape
    assert_equal normalized_shape, layer.bias.shape
    
    x = MLX.random.normal(shape: [@batch_size, @seq_length, @in_features])
    y = layer.call(x)
    assert_equal x.shape, y.shape
  end
  
  def test_group_norm
    # Test group normalization
    num_groups = 2  # Must divide channels
    channels = 4
    layer = MLX.nn.GroupNorm(num_groups, channels)
    
    # Check parameters
    assert_equal [channels], layer.weight.shape
    assert_equal [channels], layer.bias.shape
    
    # Test forward pass
    x = MLX.random.normal(shape: [@batch_size, channels, @height, @width])
    y = layer.call(x)
    
    # Check output shape
    assert_equal x.shape, y.shape
    
    # Test with num_groups = channels (should be similar to instance norm)
    layer = MLX.nn.GroupNorm(channels, channels)
    y = layer.call(x)
    assert_equal x.shape, y.shape
  end
  
  def test_instance_norm
    # Test instance normalization
    layer = MLX.nn.InstanceNorm2d(@channels_in)
    
    # Check parameters
    assert_equal [@channels_in], layer.weight.shape
    assert_equal [@channels_in], layer.bias.shape
    
    # Test forward pass
    x = MLX.random.normal(shape: [@batch_size, @channels_in, @height, @width])
    y = layer.call(x)
    
    # Check output shape
    assert_equal x.shape, y.shape
    
    # Test InstanceNorm1d
    layer = MLX.nn.InstanceNorm1d(@in_features)
    x = MLX.random.normal(shape: [@batch_size, @in_features, @seq_length])
    y = layer.call(x)
    assert_equal x.shape, y.shape
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
    
    # Setup for more detailed attention tests
    @batch_size = 2
    @seq_len = 4
    @hidden_size = 8
    @num_heads = 2
    @head_dim = @hidden_size / @num_heads
    
    # Test with attention mask
    layer = MLX.nn.MultiHeadAttention(@hidden_size, @num_heads)
    x = MLX.random.normal(shape: [@batch_size, @seq_len, @hidden_size])
    mask = MLX.zeros([@seq_len, @seq_len])
    output = layer.call(x, x, x, mask: mask)
    assert_equal [@batch_size, @seq_len, @hidden_size], output.shape
  end
  
  def test_self_attention
    # Configure for attention tests
    @batch_size = 2
    @seq_len = 4
    @hidden_size = 8
    @num_heads = 2
    
    # Test self attention module
    layer = MLX.nn.SelfAttention(@hidden_size, @num_heads)
    
    # Test forward pass
    x = MLX.random.normal(shape: [@batch_size, @seq_len, @hidden_size])
    output = layer.call(x)
    
    # Check output shape
    assert_equal [@batch_size, @seq_len, @hidden_size], output.shape
    
    # Test with dropout
    layer = MLX.nn.SelfAttention(@hidden_size, @num_heads, dropout: 0.1)
    output = layer.call(x, training: true)
    assert_equal [@batch_size, @seq_len, @hidden_size], output.shape
  end
  
  def test_cross_attention
    # Configure for attention tests
    @batch_size = 2
    @seq_len = 4
    @hidden_size = 8
    @num_heads = 2
    
    # Test cross attention module
    layer = MLX.nn.CrossAttention(@hidden_size, @num_heads)
    
    # Test forward pass
    x = MLX.random.normal(shape: [@batch_size, @seq_len, @hidden_size])
    context = MLX.random.normal(shape: [@batch_size, @seq_len + 2, @hidden_size])
    
    output = layer.call(x, context)
    
    # Check output shape - should be same as x
    assert_equal [@batch_size, @seq_len, @hidden_size], output.shape
  end
  
  def test_rnn_cell
    # Configure for RNN tests
    @batch_size = 2
    @input_size = 8
    @hidden_size = 10
    
    # Test the basic RNN cell
    cell = MLX.nn.RNNCell(@input_size, @hidden_size)
    
    # Check parameters
    assert_equal [[@hidden_size, @input_size], [@hidden_size, @hidden_size], [@hidden_size]], [
      cell.weight_ih.shape, cell.weight_hh.shape, cell.bias.shape]
    
    # Test forward pass
    x = MLX.random.normal(shape: [@batch_size, @input_size])
    h = MLX.zeros([@batch_size, @hidden_size])
    
    new_h = cell.call(x, h)
    
    # Check output shape
    assert_equal [@batch_size, @hidden_size], new_h.shape
  end
  
  def test_lstm_cell
    # Configure for RNN tests
    @batch_size = 2
    @input_size = 8
    @hidden_size = 10
    
    # Test the LSTM cell
    cell = MLX.nn.LSTMCell(@input_size, @hidden_size)
    
    # Check parameters - LSTM has 4 gates
    assert_equal [[@hidden_size * 4, @input_size], [@hidden_size * 4, @hidden_size], [@hidden_size * 4]], [
      cell.weight_ih.shape, cell.weight_hh.shape, cell.bias.shape]
    
    # Test forward pass
    x = MLX.random.normal(shape: [@batch_size, @input_size])
    h = MLX.zeros([@batch_size, @hidden_size])
    c = MLX.zeros([@batch_size, @hidden_size])
    
    new_h, new_c = cell.call(x, [h, c])
    
    # Check output shapes
    assert_equal [@batch_size, @hidden_size], new_h.shape
    assert_equal [@batch_size, @hidden_size], new_c.shape
  end
  
  def test_gru_cell
    # Configure for RNN tests
    @batch_size = 2
    @input_size = 8
    @hidden_size = 10
    
    # Test the GRU cell
    cell = MLX.nn.GRUCell(@input_size, @hidden_size)
    
    # Check parameters - GRU has 3 gates
    assert_equal [[@hidden_size * 3, @input_size], [@hidden_size * 3, @hidden_size], [@hidden_size * 3]], [
      cell.weight_ih.shape, cell.weight_hh.shape, cell.bias.shape]
    
    # Test forward pass
    x = MLX.random.normal(shape: [@batch_size, @input_size])
    h = MLX.zeros([@batch_size, @hidden_size])
    
    new_h = cell.call(x, h)
    
    # Check output shape
    assert_equal [@batch_size, @hidden_size], new_h.shape
  end
  
  def test_rnn
    # Configure for RNN tests
    @batch_size = 2
    @seq_len = 4
    @input_size = 8
    @hidden_size = 10
    @num_layers = 2
    
    # Test the RNN module (multi-layer)
    rnn = MLX.nn.RNN(@input_size, @hidden_size, num_layers: @num_layers)
    
    # Check that we have the right number of cells
    assert_equal @num_layers, rnn.cells.length
    
    # Test forward pass
    x = MLX.random.normal(shape: [@batch_size, @seq_len, @input_size])
    h0 = MLX.zeros([@num_layers, @batch_size, @hidden_size])
    
    output, hn = rnn.call(x, h0)
    
    # Check output shape
    # Output should be [batch_size, seq_len, hidden_size]
    assert_equal [@batch_size, @seq_len, @hidden_size], output.shape
    # hn should be [num_layers, batch_size, hidden_size]
    assert_equal [@num_layers, @batch_size, @hidden_size], hn.shape
    
    # Test with bidirectional
    rnn = MLX.nn.RNN(@input_size, @hidden_size, num_layers: @num_layers, bidirectional: true)
    output, hn = rnn.call(x)
    
    # For bidirectional, hidden size is doubled in output
    assert_equal [@batch_size, @seq_len, @hidden_size * 2], output.shape
    # For bidirectional, num_layers is doubled in hn
    assert_equal [@num_layers * 2, @batch_size, @hidden_size], hn.shape
  end
  
  def test_lstm
    # Configure for RNN tests
    @batch_size = 2
    @seq_len = 4
    @input_size = 8
    @hidden_size = 10
    @num_layers = 2
    
    # Test the LSTM module (multi-layer)
    lstm = MLX.nn.LSTM(@input_size, @hidden_size, num_layers: @num_layers)
    
    # Check that we have the right number of cells
    assert_equal @num_layers, lstm.cells.length
    
    # Test forward pass
    x = MLX.random.normal(shape: [@batch_size, @seq_len, @input_size])
    h0 = MLX.zeros([@num_layers, @batch_size, @hidden_size])
    c0 = MLX.zeros([@num_layers, @batch_size, @hidden_size])
    
    output, (hn, cn) = lstm.call(x, [h0, c0])
    
    # Check output shape
    # Output should be [batch_size, seq_len, hidden_size]
    assert_equal [@batch_size, @seq_len, @hidden_size], output.shape
    # hn and cn should be [num_layers, batch_size, hidden_size]
    assert_equal [@num_layers, @batch_size, @hidden_size], hn.shape
    assert_equal [@num_layers, @batch_size, @hidden_size], cn.shape
  end
  
  def test_gru
    # Configure for RNN tests
    @batch_size = 2
    @seq_len = 4
    @input_size = 8
    @hidden_size = 10
    @num_layers = 2
    
    # Test the GRU module (multi-layer)
    gru = MLX.nn.GRU(@input_size, @hidden_size, num_layers: @num_layers)
    
    # Check that we have the right number of cells
    assert_equal @num_layers, gru.cells.length
    
    # Test forward pass
    x = MLX.random.normal(shape: [@batch_size, @seq_len, @input_size])
    h0 = MLX.zeros([@num_layers, @batch_size, @hidden_size])
    
    output, hn = gru.call(x, h0)
    
    # Check output shape
    # Output should be [batch_size, seq_len, hidden_size]
    assert_equal [@batch_size, @seq_len, @hidden_size], output.shape
    # hn should be [num_layers, batch_size, hidden_size]
    assert_equal [@num_layers, @batch_size, @hidden_size], hn.shape
  end
  
  def test_flatten
    # Test flatten layer
    layer = MLX.nn.Flatten()
    
    # Test with 4D input
    x = MLX.random.normal(shape: [@batch_size, @channels_in, @height, @width])
    y = layer.call(x)
    
    # Output shape should be [batch_size, channels_in * height * width]
    assert_equal [@batch_size, @channels_in * @height * @width], y.shape
    
    # Test with start_dim and end_dim
    layer = MLX.nn.Flatten(start_dim: 2)
    x = MLX.random.normal(shape: [@batch_size, @channels_in, @height, @width])
    y = layer.call(x)
    
    # Output shape should be [batch_size, channels_in, height * width]
    assert_equal [@batch_size, @channels_in, @height * @width], y.shape
  end
  
  def test_identity
    # Test identity layer
    layer = MLX.nn.Identity()
    
    # Test with various input shapes
    shapes = [
      [@batch_size],
      [@batch_size, @in_features],
      [@batch_size, @seq_length, @in_features],
      [@batch_size, @channels_in, @height, @width]
    ]
    
    shapes.each do |shape|
      x = MLX.random.normal(shape: shape)
      y = layer.call(x)
      
      # Output should be identical to input
      assert_equal shape, y.shape
      assert MLX.array_equal(x, y)
    end
  end
  
  def test_positional_encoding
    # Test positional encoding in Transformers
    max_seq_len = 100
    d_model = 64
    
    pos_encoder = MLX.nn.SinusoidalPositionalEncoding(d_model, max_seq_len)
    
    # Test shape of encoding
    encoding = pos_encoder.call()
    assert_equal [max_seq_len, d_model], encoding.shape
    
    # Test applying to a sequence
    seq_len = 10
    batch_size = 2
    x = MLX.random.normal(shape: [batch_size, seq_len, d_model])
    
    output = pos_encoder.call(x)
    assert_equal [batch_size, seq_len, d_model], output.shape
  end
  
  def test_module_utilities
    m = MLX.nn.Sequential(
      MLX.nn.Sequential(MLX.nn.Linear(2, 10), MLX::NN::Activations.relu),
      MLX.nn.Sequential(MLX.nn.Linear(10, 10), MLX.nn.ReLU.new),
      MLX.nn.Linear(10, 1),
      MLX::NN::Activations.sigmoid
    )

    children = m.children
    assert children.is_a?(Hash)
    assert_equal 1, children.size
    assert children["layers"].is_a?(Array)
    assert_equal 4, children["layers"].size
    assert_equal({}, children["layers"][3])
    
    # Test leaf modules
    leaves = m.leaf_modules
    flattened_leaves = flatten_module_dict(leaves)
    assert_equal 4, flattened_leaves.size
    assert_equal "layers.0.layers.0", flattened_leaves[0][0]
    assert_equal "layers.1.layers.0", flattened_leaves[1][0]
    assert_equal "layers.1.layers.1", flattened_leaves[2][0]
    assert_equal "layers.2", flattened_leaves[3][0]
    assert flattened_leaves[0][1].is_a?(MLX::NN::Layers::Linear)
    assert flattened_leaves[1][1].is_a?(MLX::NN::Layers::Linear)
    assert flattened_leaves[2][1].is_a?(MLX::NN::Layers::ReLU)
    assert flattened_leaves[3][1].is_a?(MLX::NN::Layers::Linear)

    # Test eval mode
    m.eval!

    def assert_not_training(k, m)
      assert !m.training?
    end

    m.apply_to_modules(method(:assert_not_training))

    # Test train mode
    m.train!

    def assert_training(k, m)
      assert m.training?
    end

    m.apply_to_modules(method(:assert_training))
  end

  def test_module_attributes
    class AttributeModel < MLX::NN::Module
      def initialize
        super()
        @val = nil
        initialize_value
      end

      def initialize_value
        @val = MLX.array(1.0)
      end
    end

    model = AttributeModel.new
    assert MLX.array_equal(model.instance_variable_get(:@val), MLX.array(1.0))

    model.instance_variable_set(:@val, nil)
    assert_nil model.instance_variable_get(:@val)

    model.instance_variable_set(:@val, MLX.array([3]))
    assert_equal 3, model.instance_variable_get(:@val).item
  end

  def test_model_with_dict
    class DictModule < MLX::NN::Module
      def initialize
        super()
        @weights = {
          "w1" => MLX.zeros([2, 2]), 
          "w2" => MLX.ones([2, 2])
        }
        register_parameter("weights.w1", @weights["w1"])
        register_parameter("weights.w2", @weights["w2"])
      end
    end

    model = DictModule.new
    params = model.parameters
    assert_equal 2, params.size
    assert MLX.array_equal(params["weights.w1"], MLX.zeros([2, 2]))
    assert MLX.array_equal(params["weights.w2"], MLX.ones([2, 2]))
  end

  def test_save_weights
    # Helper method to create a model
    def make_model
      MLX.nn.Sequential(
        MLX.nn.Linear(2, 2),
        MLX.nn.ReLU.new,
        MLX.nn.Linear(2, 2)
      )
    end

    m = make_model()
    
    # Create a temporary directory for saving
    temp_dir = Dir.mktmpdir
    begin
      # Test saving/loading npz format
      npz_file = File.join(temp_dir, "model.npz")
      m.save_weights(npz_file)
      
      # Create a new model to load weights into
      m_load = make_model()
      m_load.load_weights(npz_file)
      
      # Evaluate to ensure everything is computed
      MLX.eval(m_load.state)
      
      # Compare parameters
      m_params = m.parameters
      m_load_params = m_load.parameters
      
      m_params.each do |key, val|
        assert MLX.array_equal(val, m_load_params[key])
      end
      
      # Test safetensors format if supported
      if m.respond_to?(:save_weights) && m.method(:save_weights).arity != 1
        safetensors_file = File.join(temp_dir, "model.safetensors")
        m.save_weights(safetensors_file)
        
        m_load = make_model()
        m_load.load_weights(safetensors_file)
        
        # Evaluate to ensure everything is computed
        MLX.eval(m_load.state)
        
        # Compare parameters
        m_params = m.parameters
        m_load_params = m_load.parameters
        
        m_params.each do |key, val|
          assert MLX.array_equal(val, m_load_params[key])
        end
      end
    ensure
      # Clean up temporary directory
      FileUtils.remove_entry(temp_dir)
    end
  end

  def test_load_from_weights
    m = MLX.nn.Linear(2, 2)

    # Test with too few weights
    weights = [["weight", MLX.ones([2, 2])]]
    assert_raises(StandardError) do
      m.load_weights(weights)
    end

    # Test with strict=false
    m.load_weights(weights, strict: false)
    assert MLX.array_equal(m.weight, weights[0][1])

    # Test with wrong name
    assert_raises(StandardError) do
      m.load_weights([["weihgt", MLX.ones([2, 2])]])
    end

    # Test with strict=false and wrong name
    m.load_weights([["weihgt", MLX.ones([2, 2])]], strict: false)

    # Test with too many weights
    assert_raises(StandardError) do
      m.load_weights([
        ["weight", MLX.ones([2, 2])],
        ["bias", MLX.ones([2])],
        ["bias2", MLX.ones([2])]
      ])
    end

    # Test with wrong shape
    assert_raises(StandardError) do
      m.load_weights([
        ["weight", MLX.ones([2, 2])],
        ["bias", MLX.ones([2, 1])]
      ])
    end

    # Test with wrong type
    assert_raises(StandardError) do
      m.load_weights([
        ["weight", MLX.ones([2, 2])],
        ["bias", 3]
      ])
    end

    # Test empty weights with strict=false
    m.load_weights([], strict: false)
  end

  def test_chaining
    m = MLX.nn.Sequential(
      MLX.nn.Linear(2, 2),
      MLX.nn.ReLU.new,
      MLX.nn.Linear(2, 1)
    )
    
    pre_freeze_num_params = m.parameters.size
    m.freeze.unfreeze
    assert_equal pre_freeze_num_params, m.parameters.size
    
    params_dict = m.parameters
    
    # Test chainable evaluation
    assert !m.update(params_dict).eval!.training?
    assert m.train!.training?
  end

  def test_bilinear
    in_features1 = 5
    in_features2 = 4
    out_features = 3
    batch_size = 2
    
    # Test bilinear layer
    layer = MLX.nn.Bilinear(in_features1, in_features2, out_features)
    
    # Check parameters
    assert_equal [out_features, in_features1, in_features2], layer.weight.shape
    assert_equal [out_features], layer.bias.shape
    
    # Test forward pass
    x1 = MLX.random.normal(shape: [batch_size, in_features1])
    x2 = MLX.random.normal(shape: [batch_size, in_features2])
    
    y = layer.call(x1, x2)
    assert_equal [batch_size, out_features], y.shape
    
    # Test without bias
    layer = MLX.nn.Bilinear(in_features1, in_features2, out_features, bias: false)
    assert_nil layer.bias
    
    y = layer.call(x1, x2)
    assert_equal [batch_size, out_features], y.shape
  end
  
  def test_upsample
    b, h, w, c = 1, 2, 2, 1
    scale_factor = 2
    
    # Create upsampling layers with different modes
    upsample_nearest = MLX.nn.Upsample(scale_factor: scale_factor, mode: "nearest", align_corners: true)
    upsample_bilinear = MLX.nn.Upsample(scale_factor: scale_factor, mode: "linear", align_corners: true)
    upsample_nearest_no_align_corners = MLX.nn.Upsample(scale_factor: scale_factor, mode: "nearest", align_corners: false)
    upsample_bilinear_no_align_corners = MLX.nn.Upsample(scale_factor: scale_factor, mode: "linear", align_corners: false)
    
    # Test single feature map with align_corners
    x = MLX.arange(b * h * w * c).reshape([b, c, h, w]).transpose([0, 2, 3, 1])
    
    # Expected values for nearest-neighbor upsampling with align_corners
    expected_nearest = MLX.array(
      [[[[0], [0], [1], [1]], 
        [[0], [0], [1], [1]], 
        [[2], [2], [3], [3]], 
        [[2], [2], [3], [3]]]]
    )
    
    nearest_result = upsample_nearest.call(x)
    assert MLX.allclose(nearest_result, expected_nearest)
    
    # Test single feature map without align_corners
    x = MLX.arange(1, b * h * w * c + 1).reshape([b, c, h, w]).transpose([0, 2, 3, 1])
    
    # Expected values for nearest-neighbor upsampling without align_corners
    expected_nearest_no_align_corners = MLX.array(
      [[[[1], [1], [2], [2]], 
        [[1], [1], [2], [2]], 
        [[3], [3], [4], [4]], 
        [[3], [3], [4], [4]]]]
    )
    
    nearest_no_align_result = upsample_nearest_no_align_corners.call(x)
    assert MLX.allclose(nearest_no_align_result, expected_nearest_no_align_corners)
    
    # Test with complex batch
    b, h, w, c = 2, 3, 3, 2
    x = MLX.arange(b * c * h * w).reshape([b, c, h, w]).transpose([0, 2, 3, 1])
    
    # Test output shapes
    nearest_result = upsample_nearest.call(x)
    bilinear_result = upsample_bilinear.call(x)
    
    expected_h = h * scale_factor
    expected_w = w * scale_factor
    
    assert_equal [b, expected_h, expected_w, c], nearest_result.shape
    assert_equal [b, expected_h, expected_w, c], bilinear_result.shape
  end

  def test_sin_pe
    # Test sinusoidal positional encoding
    hidden_size = 32
    max_len = 20
    batch_size = 2
    seq_len = 10
    
    # Create sinusoidal positional encoding
    sin_pe = MLX.nn.SinusoidalPositionalEncoding(hidden_size, max_len)
    
    # Test encoding shape when called without input
    encoding = sin_pe.call
    assert_equal [max_len, hidden_size], encoding.shape
    
    # Test with input sequence
    x = MLX.random.normal(shape: [batch_size, seq_len, hidden_size])
    output = sin_pe.call(x)
    
    # Output should have same shape as input
    assert_equal x.shape, output.shape
    
    # Output should not be the same as input
    assert !MLX.array_equal(x, output)
  end

  def test_alibi
    # Test ALiBi (Attention with Linear Biases)
    batch_size = 2
    num_heads = 4
    seq_len = 10
    
    # Create random query and key tensors
    q = MLX.random.normal(shape: [batch_size, num_heads, seq_len, 8])
    k = MLX.random.normal(shape: [batch_size, num_heads, seq_len, 8])
    
    # Create ALiBi attention bias
    alibi = MLX.nn.attention.alibi(num_heads, seq_len)
    
    # Check shape of alibi bias
    assert_equal [1, num_heads, 1, seq_len], alibi.shape
    
    # Verify dtype handling
    assert_equal MLX.float32, alibi.dtype
    
    # Test with float16
    q_fp16 = q.astype(MLX.float16)
    alibi_fp16 = MLX.nn.attention.alibi(num_heads, seq_len, dtype: MLX.float16)
    assert_equal MLX.float16, alibi_fp16.dtype
  end

  def test_extended_activations
    # Test additional activation functions
    x = MLX.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # Test softmax
    softmax = MLX.nn.Softmax.new(dim: 0)
    softmax_out = softmax.call(x)
    # Sum of softmax outputs should be approximately 1
    assert_in_delta 1.0, MLX.sum(softmax_out).item, 1e-5
    
    # Test log_softmax
    log_softmax = MLX.nn.LogSoftmax.new(dim: 0)
    log_softmax_out = log_softmax.call(x)
    # Check that log_softmax values are less than or equal to 0
    assert MLX.all(log_softmax_out <= 0).item
    
    # Test softplus
    softplus = MLX.nn.Softplus.new
    softplus_out = softplus.call(x)
    # Check that softplus values are positive
    assert MLX.all(softplus_out > 0).item
    
    # Test mish
    mish = MLX.nn.Mish.new
    mish_out = mish.call(x)
    # Verify expected behavior: negative inputs give smaller values
    assert mish_out[0].item < mish_out[3].item
    
    # Test log_sigmoid
    log_sigmoid = MLX.nn.LogSigmoid.new
    log_sigmoid_out = log_sigmoid.call(x)
    # Check that log_sigmoid values are less than or equal to 0
    assert MLX.all(log_sigmoid_out <= 0).item
    
    # Test CELU
    celu = MLX.nn.CELU.new(alpha: 1.0)
    celu_out = celu.call(x)
    # Verify behavior: positive values unchanged, negative values transformed
    assert_equal x[3].item, celu_out[3].item
    assert celu_out[0].item > -2.0
    
    # Test GELU (already in test_activation_functions)
    
    # Test PReLU
    prelu = MLX.nn.PReLU.new
    prelu_out = prelu.call(x)
    # Verify behavior: positive values unchanged, negative values scaled
    assert_equal x[3].item, prelu_out[3].item
    assert prelu_out[0].item > -2.0
    
    # Test SELU
    selu = MLX.nn.SELU.new
    selu_out = selu.call(x)
    # Verify behavior: normalized outputs
    assert selu_out[0].item > -2.0
    assert_equal x[2].item, selu_out[2].item
    
    # Test Hard Tanh
    hard_tanh = MLX.nn.Hardtanh.new(min_val: -1.0, max_val: 1.0)
    hard_tanh_out = hard_tanh.call(x)
    # Check that values are clipped to [-1, 1]
    assert_equal -1.0, hard_tanh_out[0].item
    assert_equal 1.0, hard_tanh_out[4].item
    assert_equal x[2].item, hard_tanh_out[2].item
  end

  def test_dropout3d
    # Test 3D dropout
    x = MLX.ones([2, 4, 4, 4, 4])  # [batch, channels, depth, height, width]
    drop = MLX.nn.Dropout3d(p: 0.5)
    
    # Test in eval mode
    drop.eval!
    y = drop.call(x)
    assert_equal x.shape, y.shape
    assert MLX.array_equal(x, y)
    
    # Test in training mode with deterministic settings
    MLX.deterministic!(true)
    drop.train!
    y = drop.call(x)
    assert_equal x.shape, y.shape
    
    # Some channels should be entirely dropped (or scaled in deterministic mode)
    # We should test the pattern is consistent across the spatial dimensions
    dropout_pattern = MLX.sum(y == 0, dims: [2, 3, 4])
    nonzero_pattern = dropout_pattern > 0
    channel_dims_product = 4 * 4 * 4  # depth * height * width
    
    # Channels should be either all 0 or all (channel_dims_product)
    all_zeros = MLX.sum(dropout_pattern == channel_dims_product)
    all_nonzeros = MLX.sum(dropout_pattern == 0)
    assert (all_zeros + all_nonzeros).item > 0
    
    # Reset deterministic mode
    MLX.deterministic!(false)
  end

  def test_quantize
    # Test model quantization
    def make_model
      MLX.nn.Sequential(
        MLX.nn.Embedding(5, 256),
        MLX.nn.ReLU.new,
        MLX.nn.Linear(256, 256)
      )
    end
    
    # Create original model
    m = make_model()
    
    # Apply quantization
    MLX.nn.quantize(m)
    
    # Check that layers are properly quantized
    assert m.layers[0].is_a?(MLX::NN::QuantizedEmbedding)
    assert m.layers[1].is_a?(MLX::NN::Layers::ReLU)
    assert m.layers[2].is_a?(MLX::NN::QuantizedLinear)
    
    # Test with class predicate
    m = make_model()
    MLX.nn.quantize(m) do |_, mod|
      mod.is_a?(MLX::NN::Layers::Linear)
    end
    
    # Check that only Linear is quantized
    assert m.layers[0].is_a?(MLX::NN::Layers::Embedding)
    assert m.layers[1].is_a?(MLX::NN::Layers::ReLU)
    assert m.layers[2].is_a?(MLX::NN::QuantizedLinear)
  end

  def test_quantized_embedding
    # Test quantized embedding
    embedding_dim = 256
    vocab_size = 32
    
    # Create regular embedding
    emb = MLX.nn.Embedding(vocab_size, embedding_dim)
    
    # Create quantized embedding from regular embedding
    qemb = MLX.nn.QuantizedEmbedding.from_embedding(emb, bits: 8)
    
    # Test forward pass
    x = MLX.array([2, 6, 9, 3, 0, 3])
    y = emb.call(x)
    yq = qemb.call(x)
    
    # Verify yq is close to y
    assert (y - yq).abs.max.item < qemb.scales.max.item
    
    # Test as_linear
    x = MLX.random.uniform(shape: [2, embedding_dim])
    y = emb.as_linear(x)
    yq = qemb.as_linear(x)
    
    # Compute cosine similarity
    def cosine(a, b)
      ab = (a * b).sum(-1)
      aa = MLX.linalg.norm(a, axis: -1)
      bb = MLX.linalg.norm(b, axis: -1)
      ab / (aa * bb)
    end
    
    # Verify cosine similarity is high (close to 1)
    assert cosine(y, yq).min.item > 0.99
  end

  def test_causal_mask
    # Test creation of causal masks for attention
    seq_len = 4
    
    # Create causal mask with float16
    mask = MLX.nn.MultiHeadAttention.create_additive_causal_mask(seq_len, MLX.float16)
    
    # Check that mask contains no NaN values
    assert !MLX.any(MLX.isnan(mask)).item
    
    # Check that mask is properly causal (upper triangular values are negative)
    assert mask[0, -1].item < 0
    
    # Test with bfloat16
    mask = MLX.nn.MultiHeadAttention.create_additive_causal_mask(seq_len, MLX.bfloat16)
    
    # Check that mask contains no NaN values
    assert !MLX.any(MLX.isnan(mask)).item
    
    # Check that mask is properly causal
    assert mask[0, -1].item < 0
  end

  def test_set_dtype
    # Test setting dtype for model parameters
    def assert_dtype(layer, dtype)
      layer.parameters.each do |k, v|
        assert_equal dtype, v.dtype, "dtype mismatch for #{k}"
      end
    end
    
    # Create layer
    layer = MLX.nn.Linear(input_dims: 4, output_dims: 8, bias: true)
    assert_dtype(layer, MLX.float32)
    
    # Change dtype
    layer.set_dtype(MLX.bfloat16)
    assert_dtype(layer, MLX.bfloat16)
    
    # Test with false predicate
    layer.set_dtype(MLX.float32) { |x| false }
    assert_dtype(layer, MLX.bfloat16)
    
    # Test with true predicate
    layer.set_dtype(MLX.int32) { |x| true }
    assert_dtype(layer, MLX.int32)
    
    # Test with nil predicate
    layer.set_dtype(MLX.int64, predicate: nil)
    assert_dtype(layer, MLX.int64)
    
    # Test with dtype-based predicate
    layer.set_dtype(MLX.int16) { |x| MLX.issubdtype(x, MLX.integer) }
    assert_dtype(layer, MLX.int16)
  end
  
  # Helper method to flatten nested module dictionaries
  def flatten_module_dict(modules_dict, prefix = "", result = [])
    modules_dict.each do |key, value|
      path = prefix.empty? ? key : "#{prefix}.#{key}"
      
      if value.is_a?(Hash)
        flatten_module_dict(value, path, result)
      else
        result << [path, value]
      end
    end
    
    result
  end
  
  # Metrics tests
  def test_accuracy
    # Test binary classification
    y_true = MLX.array([0, 1, 0, 1, 0])
    y_pred = MLX.array([0, 1, 1, 1, 0])
    acc = MLX.metrics.accuracy(y_true, y_pred)
    assert_equal 0.8, acc.item
    
    # Test multiclass classification
    y_true = MLX.array([0, 1, 2, 1, 0])
    y_pred = MLX.array([0, 1, 1, 1, 0])
    acc = MLX.metrics.accuracy(y_true, y_pred)
    assert_equal 0.8, acc.item
    
    # Test with logits
    logits = MLX.array([
      [0.9, 0.1, 0.0],
      [0.2, 0.7, 0.1],
      [0.1, 0.8, 0.1],
      [0.0, 0.9, 0.1],
      [0.8, 0.1, 0.1]
    ])
    y_true = MLX.array([0, 1, 2, 1, 0])
    acc = MLX.metrics.accuracy(y_true, MLX.argmax(logits, axis: 1))
    assert_equal 0.6, acc.item
    
    # Test with threshold for binary classification
    y_prob = MLX.array([0.9, 0.6, 0.4, 0.7, 0.1])
    y_true = MLX.array([1, 1, 0, 1, 0])
    
    # Default threshold 0.5
    y_pred = (y_prob >= 0.5).astype(MLX.int32)
    acc = MLX.metrics.accuracy(y_true, y_pred)
    assert_equal 0.8, acc.item
    
    # Custom threshold 0.6
    y_pred = (y_prob >= 0.6).astype(MLX.int32)
    acc = MLX.metrics.accuracy(y_true, y_pred)
    assert_equal 0.6, acc.item
  end
  
  def test_precision
    # Test binary precision
    y_true = MLX.array([0, 1, 0, 1, 0])
    y_pred = MLX.array([1, 1, 0, 1, 0])
    prec = MLX.metrics.precision(y_true, y_pred)
    assert_equal 0.67, (prec.item * 100).round / 100.0 # Rounded to handle floating point
    
    # Test multiclass precision (macro)
    y_true = MLX.array([0, 1, 2, 1, 0])
    y_pred = MLX.array([0, 1, 1, 1, 0])
    prec = MLX.metrics.precision(y_true, y_pred, average: 'macro')
    # Class 0: 1.0, Class 1: 0.67, Class 2: 0.0 -> avg = 0.56
    assert_in_delta 0.56, prec.item, 0.01
    
    # Test multiclass precision (micro)
    prec = MLX.metrics.precision(y_true, y_pred, average: 'micro')
    assert_equal 0.8, prec.item
    
    # Test with binary labels and class index
    y_true = MLX.array([0, 1, 0, 1, 0])
    y_pred = MLX.array([1, 1, 0, 0, 0])
    prec = MLX.metrics.precision(y_true, y_pred, pos_label: 1)
    assert_equal 0.5, prec.item
    
    # Test handling of empty predictions
    y_true = MLX.array([0, 0, 0, 0, 0])
    y_pred = MLX.array([0, 0, 0, 0, 0])
    
    # In this case, precision for class 1 should be undefined or 0
    # Different libraries handle this differently, so we'll check for both possibilities
    prec = MLX.metrics.precision(y_true, y_pred, pos_label: 1)
    assert prec.item == 0.0 || Float::NAN == prec.item
  end
  
  def test_recall
    # Test binary recall
    y_true = MLX.array([0, 1, 0, 1, 0])
    y_pred = MLX.array([1, 1, 0, 0, 0])
    rec = MLX.metrics.recall(y_true, y_pred)
    assert_equal 0.5, rec.item
    
    # Test multiclass recall (macro)
    y_true = MLX.array([0, 1, 2, 1, 0])
    y_pred = MLX.array([0, 1, 1, 1, 0])
    rec = MLX.metrics.recall(y_true, y_pred, average: 'macro')
    # Class 0: 1.0, Class 1: 1.0, Class 2: 0.0 -> avg = 0.67
    assert_in_delta 0.67, rec.item, 0.01
    
    # Test multiclass recall (micro)
    rec = MLX.metrics.recall(y_true, y_pred, average: 'micro')
    assert_equal 0.8, rec.item
    
    # Test with binary labels and class index
    y_true = MLX.array([0, 1, 0, 1, 0])
    y_pred = MLX.array([0, 0, 0, 1, 1])
    rec = MLX.metrics.recall(y_true, y_pred, pos_label: 1)
    assert_equal 0.5, rec.item
    
    # Test handling of empty true positives
    y_true = MLX.array([0, 0, 0, 0, 0])
    y_pred = MLX.array([1, 1, 0, 0, 0])
    
    # In this case, recall for class 1 should be undefined or 0
    rec = MLX.metrics.recall(y_true, y_pred, pos_label: 1)
    assert rec.item == 0.0 || Float::NAN == rec.item
  end
  
  def test_f1_score
    # Test binary f1
    y_true = MLX.array([0, 1, 0, 1, 0])
    y_pred = MLX.array([1, 1, 0, 0, 0])
    f1 = MLX.metrics.f1_score(y_true, y_pred)
    assert_in_delta 0.5, f1.item, 0.01
    
    # Test multiclass f1 (macro)
    y_true = MLX.array([0, 1, 2, 1, 0])
    y_pred = MLX.array([0, 1, 1, 1, 0])
    f1 = MLX.metrics.f1_score(y_true, y_pred, average: 'macro')
    # Class 0: 1.0, Class 1: 0.8, Class 2: 0.0 -> avg = 0.6
    assert_in_delta 0.6, f1.item, 0.01
    
    # Test multiclass f1 (micro)
    f1 = MLX.metrics.f1_score(y_true, y_pred, average: 'micro')
    assert_equal 0.8, f1.item
    
    # Test weighted average
    f1 = MLX.metrics.f1_score(y_true, y_pred, average: 'weighted')
    # Class weights: [2/5, 2/5, 1/5] -> weighted avg = 0.76
    assert_in_delta 0.76, f1.item, 0.01
    
    # Test with binary labels and class index
    y_true = MLX.array([0, 1, 0, 1, 0])
    y_pred = MLX.array([0, 0, 0, 1, 1])
    f1 = MLX.metrics.f1_score(y_true, y_pred, pos_label: 1)
    assert_equal 0.5, f1.item
  end
  
  def test_precision_recall_fscore_support
    # Test all metrics together
    y_true = MLX.array([0, 1, 2, 0, 1, 2])
    y_pred = MLX.array([0, 2, 1, 0, 0, 1])
    
    precision, recall, f1, support = MLX.metrics.precision_recall_fscore_support(y_true, y_pred)
    
    # Check shapes
    assert_equal [3], precision.shape
    assert_equal [3], recall.shape
    assert_equal [3], f1.shape
    assert_equal [3], support.shape
    
    # Check support values
    assert MLX.array_equal(support, MLX.array([2, 2, 2]))
    
    # Check precision values
    # Class 0: 2/3, Class 1: 0/2, Class 2: 0/1
    assert_in_delta 0.67, precision[0].item, 0.01
    assert_equal 0.0, precision[1].item
    assert_equal 0.0, precision[2].item
    
    # Check recall values
    # Class 0: 2/2, Class 1: 0/2, Class 2: 0/2
    assert_equal 1.0, recall[0].item
    assert_equal 0.0, recall[1].item
    assert_equal 0.0, recall[2].item
    
    # Check F1 values
    assert_in_delta 0.8, f1[0].item, 0.01
    assert_equal 0.0, f1[1].item
    assert_equal 0.0, f1[2].item
    
    # Test with macro average
    precision, recall, f1, _ = MLX.metrics.precision_recall_fscore_support(
      y_true, y_pred, average: 'macro'
    )
    
    # Check shapes for averaged values
    assert_equal [], precision.shape  # scalar
    assert_equal [], recall.shape     # scalar
    assert_equal [], f1.shape         # scalar
    
    # Check averaged values
    assert_in_delta 0.22, precision.item, 0.01  # (0.67 + 0 + 0) / 3
    assert_in_delta 0.33, recall.item, 0.01     # (1.0 + 0 + 0) / 3
    assert_in_delta 0.27, f1.item, 0.01         # (0.8 + 0 + 0) / 3
  end
  
  def test_mean_squared_error
    y_true = MLX.array([3.0, -0.5, 2.0, 7.0])
    y_pred = MLX.array([2.5, 0.0, 2.0, 8.0])
    
    mse = MLX.metrics.mean_squared_error(y_true, y_pred)
    # (0.5^2 + 0.5^2 + 0^2 + 1^2) / 4 = 0.375
    assert_equal 0.375, mse.item
    
    # Test with squared=False for RMSE
    rmse = MLX.metrics.mean_squared_error(y_true, y_pred, squared: false)
    assert_in_delta 0.612, rmse.item, 0.001  # √0.375 ≈ 0.612
    
    # Test with sample weights
    weights = MLX.array([1.0, 1.0, 2.0, 0.5])
    mse = MLX.metrics.mean_squared_error(y_true, y_pred, sample_weight: weights)
    # (0.5^2*1 + 0.5^2*1 + 0^2*2 + 1^2*0.5) / 4.5 = 0.333
    assert_in_delta 0.333, mse.item, 0.001
  end
  
  def test_mean_absolute_error
    y_true = MLX.array([3.0, -0.5, 2.0, 7.0])
    y_pred = MLX.array([2.5, 0.0, 2.0, 8.0])
    
    mae = MLX.metrics.mean_absolute_error(y_true, y_pred)
    # (0.5 + 0.5 + 0 + 1) / 4 = 0.5
    assert_equal 0.5, mae.item
    
    # Test with sample weights
    weights = MLX.array([1.0, 1.0, 2.0, 0.5])
    mae = MLX.metrics.mean_absolute_error(y_true, y_pred, sample_weight: weights)
    # (0.5*1 + 0.5*1 + 0*2 + 1*0.5) / 4.5 = 0.333
    assert_in_delta 0.333, mae.item, 0.001
  end
  
  def test_r2_score
    y_true = MLX.array([3.0, -0.5, 2.0, 7.0])
    y_pred = MLX.array([2.5, 0.0, 2.0, 8.0])
    
    r2 = MLX.metrics.r2_score(y_true, y_pred)
    # 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
    # mean(y_true) = 2.875
    # 1 - 1.5 / 36.875 = 0.9593
    assert_in_delta 0.959, r2.item, 0.001
    
    # Test with sample weights
    weights = MLX.array([1.0, 1.0, 2.0, 0.5])
    r2 = MLX.metrics.r2_score(y_true, y_pred, sample_weight: weights)
    assert r2.item < 1.0 && r2.item > 0.9  # Should be similar but not identical
    
    # Test with perfect prediction
    r2 = MLX.metrics.r2_score(y_true, y_true)
    assert_equal 1.0, r2.item
    
    # Test with worst case (predicting the mean)
    mean = MLX.mean(y_true)
    y_pred_mean = MLX.full_like(y_true, mean)
    r2 = MLX.metrics.r2_score(y_true, y_pred_mean)
    assert_in_delta 0.0, r2.item, 0.001
  end
  
  def test_binary_confusion_matrix
    y_true = MLX.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = MLX.array([0, 1, 0, 0, 1, 1, 1, 1])
    
    cm = MLX.metrics.confusion_matrix(y_true, y_pred)
    
    # Expected confusion matrix:
    # [[2, 2],   # TN=2, FP=2
    #  [1, 3]]   # FN=1, TP=3
    
    assert_equal [2, 2], cm.shape.to_a
    assert_equal 2, cm[0, 0].item  # True negatives
    assert_equal 2, cm[0, 1].item  # False positives
    assert_equal 1, cm[1, 0].item  # False negatives
    assert_equal 3, cm[1, 1].item  # True positives
    
    # Test with normalized=true
    cm_norm = MLX.metrics.confusion_matrix(y_true, y_pred, normalize: true)
    
    assert_in_delta 0.25, cm_norm[0, 0].item, 0.01  # TN rate
    assert_in_delta 0.25, cm_norm[0, 1].item, 0.01  # FP rate
    assert_in_delta 0.125, cm_norm[1, 0].item, 0.01  # FN rate
    assert_in_delta 0.375, cm_norm[1, 1].item, 0.01  # TP rate
  end
  
  def test_multiclass_confusion_matrix
    y_true = MLX.array([0, 1, 2, 0, 1, 2, 0, 2])
    y_pred = MLX.array([0, 1, 1, 0, 2, 2, 1, 2])
    
    cm = MLX.metrics.confusion_matrix(y_true, y_pred)
    
    # Expected confusion matrix:
    # [[2, 1, 0],   # Class 0: 2 correct, 1 misclassified as class 1
    #  [0, 1, 1],   # Class 1: 1 correct, 1 misclassified as class 2
    #  [0, 1, 2]]   # Class 2: 2 correct, 1 misclassified as class 1
    
    assert_equal [3, 3], cm.shape.to_a
    
    # Check specific elements
    assert_equal 2, cm[0, 0].item
    assert_equal 1, cm[0, 1].item
    assert_equal 0, cm[0, 2].item
    
    assert_equal 0, cm[1, 0].item
    assert_equal 1, cm[1, 1].item
    assert_equal 1, cm[1, 2].item
    
    assert_equal 0, cm[2, 0].item
    assert_equal 1, cm[2, 1].item
    assert_equal 2, cm[2, 2].item
  end
  
  def test_precision_at_k
    # Test precision@k for multi-label classification
    y_true = MLX.array([
      [1, 0, 1, 0, 0],
      [0, 1, 0, 1, 0],
      [0, 0, 1, 0, 1]
    ])
    
    y_scores = MLX.array([
      [0.5, 0.1, 0.8, 0.2, 0.3],
      [0.2, 0.7, 0.3, 0.6, 0.1],
      [0.1, 0.3, 0.6, 0.2, 0.9]
    ])
    
    # Precision@1
    p_at_1 = MLX.metrics.precision_at_k(y_true, y_scores, k: 1)
    # For each row, top prediction is:
    # Row 0: index 2 (true positive)
    # Row 1: index 1 (true positive)
    # Row 2: index 4 (true positive)
    # 3/3 = 1.0
    assert_equal 1.0, p_at_1.item
    
    # Precision@2
    p_at_2 = MLX.metrics.precision_at_k(y_true, y_scores, k: 2)
    # For each row, top 2 predictions are:
    # Row 0: indices 2, 0 (both true positives)
    # Row 1: indices 1, 3 (both true positives)
    # Row 2: indices 4, 2 (both true positives)
    # 6/6 = 1.0
    assert_equal 1.0, p_at_2.item
    
    # Precision@3
    p_at_3 = MLX.metrics.precision_at_k(y_true, y_scores, k: 3)
    # For each row, top 3 predictions are:
    # Row 0: indices 2, 0, 4 (2 true positives, 1 false positive)
    # Row 1: indices 1, 3, 2 (2 true positives, 1 false positive)
    # Row 2: indices 4, 2, 1 (2 true positives, 1 false positive)
    # 6/9 = 0.6667
    assert_in_delta 0.67, p_at_3.item, 0.01
    
    # Test with sample weights
    weights = MLX.array([1.0, 2.0, 1.0])
    p_at_3_weighted = MLX.metrics.precision_at_k(y_true, y_scores, k: 3, sample_weight: weights)
    # Row 0: 2/3 * 1.0 = 2/3
    # Row 1: 2/3 * 2.0 = 4/3
    # Row 2: 2/3 * 1.0 = 2/3
    # (2/3 + 4/3 + 2/3) / 4 = 2/3 = 0.6667
    assert_in_delta 0.67, p_at_3_weighted.item, 0.01
  end
  
  def test_roc_auc_score
    y_true = MLX.array([0, 0, 1, 1])
    y_score = MLX.array([0.1, 0.4, 0.35, 0.8])
    
    auc = MLX.metrics.roc_auc_score(y_true, y_score)
    # AUC calculation:
    # Sort by score: [(0.1, 0), (0.35, 1), (0.4, 0), (0.8, 1)]
    # TPR/FPR pairs: (0,0), (0.5,0), (0.5,0.5), (1,0.5), (1,1)
    # AUC = 0.5 * (0 + 0.5) + 0.5 * (0.5 * 0.5) + 0.5 * (1 * 0.5) = 0.625
    assert_in_delta 0.625, auc.item, 0.01
    
    # Test with sample weights
    weights = MLX.array([2.0, 1.0, 3.0, 1.0])
    auc_weighted = MLX.metrics.roc_auc_score(y_true, y_score, sample_weight: weights)
    # Weight-adjusted AUC
    assert auc_weighted.item > 0 && auc_weighted.item < 1.0
    
    # Test multi-class ROC AUC with OvR strategy
    y_true = MLX.array([0, 1, 2, 0, 1, 2])
    y_score = MLX.array([
      [0.9, 0.1, 0.2],  # Predicted as class 0
      [0.2, 0.7, 0.1],  # Predicted as class 1
      [0.2, 0.3, 0.8],  # Predicted as class 2
      [0.8, 0.2, 0.1],  # Predicted as class 0
      [0.3, 0.6, 0.2],  # Predicted as class 1
      [0.1, 0.2, 0.9]   # Predicted as class 2
    ])
    
    # One-vs-Rest strategy (default)
    auc_ovr = MLX.metrics.roc_auc_score(y_true, y_score, multi_class: 'ovr')
    assert auc_ovr.item > 0.8  # Should be high for this toy example
  end
end 