require_relative 'mlx_test_case'

class TestRNN < MLXTestCase
  def setup
    # Set random seed for reproducibility
    MLX.random.seed(42)
    
    # Common dimensions
    @batch_size = 2
    @seq_len = 4
    @input_size = 8
    @hidden_size = 10
    @num_layers = 2
    @output_size = 6
  end
  
  def test_rnn_cell
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
    
    # Test with no bias
    cell = MLX.nn.RNNCell(@input_size, @hidden_size, bias: false)
    assert_nil cell.bias
    
    new_h = cell.call(x, h)
    assert_equal [@batch_size, @hidden_size], new_h.shape
    
    # Test with custom nonlinearity
    cell = MLX.nn.RNNCell(@input_size, @hidden_size, nonlinearity: 'relu')
    new_h = cell.call(x, h)
    assert_equal [@batch_size, @hidden_size], new_h.shape
  end
  
  def test_lstm_cell
    # Test the LSTM cell
    cell = MLX.nn.LSTMCell(@input_size, @hidden_size)
    
    # Check parameters
    # LSTM has 4 gates
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
    
    # Test with no bias
    cell = MLX.nn.LSTMCell(@input_size, @hidden_size, bias: false)
    assert_nil cell.bias
    
    new_h, new_c = cell.call(x, [h, c])
    assert_equal [@batch_size, @hidden_size], new_h.shape
    assert_equal [@batch_size, @hidden_size], new_c.shape
  end
  
  def test_gru_cell
    # Test the GRU cell
    cell = MLX.nn.GRUCell(@input_size, @hidden_size)
    
    # Check parameters
    # GRU has 3 gates
    assert_equal [[@hidden_size * 3, @input_size], [@hidden_size * 3, @hidden_size], [@hidden_size * 3]], [
      cell.weight_ih.shape, cell.weight_hh.shape, cell.bias.shape]
    
    # Test forward pass
    x = MLX.random.normal(shape: [@batch_size, @input_size])
    h = MLX.zeros([@batch_size, @hidden_size])
    
    new_h = cell.call(x, h)
    
    # Check output shape
    assert_equal [@batch_size, @hidden_size], new_h.shape
    
    # Test with no bias
    cell = MLX.nn.GRUCell(@input_size, @hidden_size, bias: false)
    assert_nil cell.bias
    
    new_h = cell.call(x, h)
    assert_equal [@batch_size, @hidden_size], new_h.shape
  end
  
  def test_rnn
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
    
    # Test with default initial state (None)
    output, hn = rnn.call(x)
    assert_equal [@batch_size, @seq_len, @hidden_size], output.shape
    assert_equal [@num_layers, @batch_size, @hidden_size], hn.shape
    
    # Test with bidirectional
    rnn = MLX.nn.RNN(@input_size, @hidden_size, num_layers: @num_layers, bidirectional: true)
    output, hn = rnn.call(x)
    
    # For bidirectional, hidden size is doubled in output
    assert_equal [@batch_size, @seq_len, @hidden_size * 2], output.shape
    # For bidirectional, num_layers is doubled in hn
    assert_equal [@num_layers * 2, @batch_size, @hidden_size], hn.shape
    
    # Test with dropout
    rnn = MLX.nn.RNN(@input_size, @hidden_size, num_layers: @num_layers, dropout: 0.5)
    output, hn = rnn.call(x, training: true)
    assert_equal [@batch_size, @seq_len, @hidden_size], output.shape
  end
  
  def test_lstm
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
    
    # Test with default initial state (None)
    output, (hn, cn) = lstm.call(x)
    assert_equal [@batch_size, @seq_len, @hidden_size], output.shape
    assert_equal [@num_layers, @batch_size, @hidden_size], hn.shape
    assert_equal [@num_layers, @batch_size, @hidden_size], cn.shape
    
    # Test with bidirectional
    lstm = MLX.nn.LSTM(@input_size, @hidden_size, num_layers: @num_layers, bidirectional: true)
    output, (hn, cn) = lstm.call(x)
    
    # For bidirectional, hidden size is doubled in output
    assert_equal [@batch_size, @seq_len, @hidden_size * 2], output.shape
    # For bidirectional, num_layers is doubled in hn, cn
    assert_equal [@num_layers * 2, @batch_size, @hidden_size], hn.shape
    assert_equal [@num_layers * 2, @batch_size, @hidden_size], cn.shape
    
    # Test with dropout
    lstm = MLX.nn.LSTM(@input_size, @hidden_size, num_layers: @num_layers, dropout: 0.5)
    output, (hn, cn) = lstm.call(x, training: true)
    assert_equal [@batch_size, @seq_len, @hidden_size], output.shape
  end
  
  def test_gru
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
    
    # Test with default initial state (None)
    output, hn = gru.call(x)
    assert_equal [@batch_size, @seq_len, @hidden_size], output.shape
    assert_equal [@num_layers, @batch_size, @hidden_size], hn.shape
    
    # Test with bidirectional
    gru = MLX.nn.GRU(@input_size, @hidden_size, num_layers: @num_layers, bidirectional: true)
    output, hn = gru.call(x)
    
    # For bidirectional, hidden size is doubled in output
    assert_equal [@batch_size, @seq_len, @hidden_size * 2], output.shape
    # For bidirectional, num_layers is doubled in hn
    assert_equal [@num_layers * 2, @batch_size, @hidden_size], hn.shape
    
    # Test with dropout
    gru = MLX.nn.GRU(@input_size, @hidden_size, num_layers: @num_layers, dropout: 0.5)
    output, hn = gru.call(x, training: true)
    assert_equal [@batch_size, @seq_len, @hidden_size], output.shape
  end
  
  def test_rnn_gradients
    # Test gradient flow through RNN
    
    # Create a small RNN
    rnn = MLX.nn.RNN(@input_size, @hidden_size, num_layers: 1)
    
    # Create a simple linear layer for classification
    linear = MLX.nn.Linear(@hidden_size, @output_size)
    
    # Define a function that computes the output and loss
    def rnn_fn(x, target, rnn, linear)
      # Forward pass through RNN
      output, _ = rnn.call(x)
      
      # Take the last time step output
      last_output = output[:, -1, :]
      
      # Forward pass through linear layer
      logits = linear.call(last_output)
      
      # Compute cross-entropy loss
      loss = MLX.nn.losses.cross_entropy(logits, target)
      
      # Return loss
      loss
    end
    
    # Create random inputs and targets
    x = MLX.random.normal(shape: [@batch_size, @seq_len, @input_size])
    target = MLX.random.randint(0, @output_size, shape: [@batch_size])
    
    # Compute gradients
    grad_fn = MLX.grad(rnn_fn)
    grads = grad_fn.call(x, target, rnn, linear)
    
    # Check that gradients are not None
    assert_not_nil grads
  end
  
  def test_packed_sequence
    # Test PackedSequence for variable length sequences
    
    # Create a batch of variable length sequences
    # Sequence lengths: [4, 2, 3]
    data = [
      MLX.array([1.0, 2.0, 3.0, 4.0]),
      MLX.array([5.0, 6.0]),
      MLX.array([7.0, 8.0, 9.0])
    ]
    
    # Create a PackedSequence
    packed = MLX.nn.utils.rnn.pack_sequence(data, enforce_sorted: false)
    
    # Check attributes
    assert_equal [9, 1], packed.data.shape  # 9 total elements, 1 feature dimension
    assert_equal [3], packed.batch_sizes.shape  # 3 different sequence lengths
    
    # Pad the packed sequence back to a regular tensor
    padded, lengths = MLX.nn.utils.rnn.pad_packed_sequence(packed, batch_first: true)
    
    # Check shapes
    assert_equal [3, 4, 1], padded.shape  # 3 sequences, max length 4, 1 feature
    assert_equal [3], lengths.shape  # 3 sequence lengths
    
    # Check that lengths are correct
    expected_lengths = MLX.array([4, 2, 3])
    assert MLX.array_equal(expected_lengths, lengths)
    
    # Test with multi-dimensional data
    data_2d = [
      MLX.random.normal(shape: [4, 2]),  # Sequence of length 4
      MLX.random.normal(shape: [2, 2]),  # Sequence of length 2
      MLX.random.normal(shape: [3, 2])   # Sequence of length 3
    ]
    
    packed_2d = MLX.nn.utils.rnn.pack_sequence(data_2d, enforce_sorted: false)
    assert_equal [9, 2], packed_2d.data.shape  # 9 total elements, 2 features
    
    padded_2d, lengths_2d = MLX.nn.utils.rnn.pad_packed_sequence(packed_2d, batch_first: true)
    assert_equal [3, 4, 2], padded_2d.shape  # 3 sequences, max length 4, 2 features
  end
  
  def test_rnn_with_packed_sequence
    # Test running an RNN with a PackedSequence
    
    # Create an RNN
    rnn = MLX.nn.RNN(@input_size, @hidden_size)
    
    # Create a batch of variable length sequences
    data = [
      MLX.random.normal(shape: [5, @input_size]),  # Sequence of length 5
      MLX.random.normal(shape: [3, @input_size]),  # Sequence of length 3
      MLX.random.normal(shape: [4, @input_size])   # Sequence of length 4
    ]
    
    # Pack the sequences
    packed = MLX.nn.utils.rnn.pack_sequence(data, enforce_sorted: false)
    
    # Run the RNN on packed sequence
    output_packed, hidden = rnn.call(packed)
    
    # Unpack the output
    output_padded, output_lengths = MLX.nn.utils.rnn.pad_packed_sequence(output_packed, batch_first: true)
    
    # Check output shape
    assert_equal [3, 5, @hidden_size], output_padded.shape  # 3 sequences, max length 5
    
    # Check that output_lengths match input lengths
    expected_lengths = MLX.array([5, 3, 4])
    assert MLX.array_equal(expected_lengths, output_lengths)
    
    # Hidden state should have the regular shape
    assert_equal [1, 3, @hidden_size], hidden.shape  # 1 layer, 3 batch, hidden size
  end
  
  def test_bidirectional_rnn
    # Test detailed behavior of bidirectional RNN
    
    # Create a bidirectional RNN
    rnn = MLX.nn.RNN(@input_size, @hidden_size, bidirectional: true)
    
    # Create input
    x = MLX.random.normal(shape: [@batch_size, @seq_len, @input_size])
    
    # Run forward
    output, hidden = rnn.call(x)
    
    # Check output shape: bidirectional doubles the hidden dimension
    assert_equal [@batch_size, @seq_len, @hidden_size * 2], output.shape
    
    # Check hidden state shape: bidirectional adds a dimension for direction
    assert_equal [2, @batch_size, @hidden_size], hidden.shape  # 2 directions
    
    # Split the output to see forward and backward directions
    output_forward = output[:, :, 0:@hidden_size]
    output_backward = output[:, :, @hidden_size:]
    
    # Check individual output shapes
    assert_equal [@batch_size, @seq_len, @hidden_size], output_forward.shape
    assert_equal [@batch_size, @seq_len, @hidden_size], output_backward.shape
    
    # Get hidden states for each direction
    hidden_forward = hidden[0]
    hidden_backward = hidden[1]
    
    # Check individual hidden shapes
    assert_equal [@batch_size, @hidden_size], hidden_forward.shape
    assert_equal [@batch_size, @hidden_size], hidden_backward.shape
  end
  
  def test_linear_rnn
    # Test a linear RNN (no nonlinearity)
    
    # Create a linear RNN (tanh activation is replaced with identity)
    def identity(x)
      x
    end
    
    rnn = MLX.nn.RNN(@input_size, @hidden_size, nonlinearity: identity)
    
    # Test forward pass
    x = MLX.random.normal(shape: [@batch_size, @seq_len, @input_size])
    output, hidden = rnn.call(x)
    
    # Check shapes
    assert_equal [@batch_size, @seq_len, @hidden_size], output.shape
    assert_equal [1, @batch_size, @hidden_size], hidden.shape
  end
end 