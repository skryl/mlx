module MLX
  module NN
    module Layers
      # Base class for RNN cells (single step)
      class RNNCell < MLX::NN::Module
        attr_reader :input_size, :hidden_size, :bias
        
        def initialize(input_size, hidden_size, bias: true)
          super()
          @input_size = input_size
          @hidden_size = hidden_size
          @bias = bias
          
          # Input to hidden weights
          weight_ih = MLX::NN::Init.xavier_uniform([input_size, hidden_size])
          register_parameter('weight_ih', weight_ih)
          
          # Hidden to hidden weights
          weight_hh = MLX::NN::Init.xavier_uniform([hidden_size, hidden_size])
          register_parameter('weight_hh', weight_hh)
          
          if bias
            # Input-hidden bias
            bias_ih = MLX.zeros([hidden_size])
            register_parameter('bias_ih', bias_ih)
            
            # Hidden-hidden bias
            bias_hh = MLX.zeros([hidden_size])
            register_parameter('bias_hh', bias_hh)
          end
        end
        
        # Forward a single time step
        # @param x [MLX::Array] input tensor of shape (batch_size, input_size)
        # @param h [MLX::Array] hidden state of shape (batch_size, hidden_size)
        # @return [MLX::Array] next hidden state
        def forward(x, h)
          # Linear transformation of input
          ih = MLX.matmul(x, @_parameters['weight_ih'])
          
          # Linear transformation of hidden state
          hh = MLX.matmul(h, @_parameters['weight_hh'])
          
          if @bias
            # Add biases
            ih = MLX.add(ih, @_parameters['bias_ih'])
            hh = MLX.add(hh, @_parameters['bias_hh'])
          end
          
          # Combine and apply activation
          h_next = MLX.tanh(MLX.add(ih, hh))
          
          h_next
        end
        
        def reset_parameters
          # Initialize input-hidden weights
          @_parameters['weight_ih'] = MLX::NN::Init.xavier_uniform([@input_size, @hidden_size])
          
          # Initialize hidden-hidden weights
          @_parameters['weight_hh'] = MLX::NN::Init.xavier_uniform([@hidden_size, @hidden_size])
          
          if @bias
            # Initialize biases to zero
            @_parameters['bias_ih'] = MLX.zeros([@hidden_size])
            @_parameters['bias_hh'] = MLX.zeros([@hidden_size])
          end
        end
      end
      
      # LSTM Cell (single step)
      class LSTMCell < MLX::NN::Module
        attr_reader :input_size, :hidden_size, :bias
        
        def initialize(input_size, hidden_size, bias: true)
          super()
          @input_size = input_size
          @hidden_size = hidden_size
          @bias = bias
          
          # Input to hidden weights (4x because of input, forget, cell, output gates)
          weight_ih = MLX::NN::Init.xavier_uniform([input_size, 4 * hidden_size])
          register_parameter('weight_ih', weight_ih)
          
          # Hidden to hidden weights
          weight_hh = MLX::NN::Init.xavier_uniform([hidden_size, 4 * hidden_size])
          register_parameter('weight_hh', weight_hh)
          
          if bias
            # Input-hidden bias
            bias_ih = MLX.zeros([4 * hidden_size])
            register_parameter('bias_ih', bias_ih)
            
            # Hidden-hidden bias
            bias_hh = MLX.zeros([4 * hidden_size])
            register_parameter('bias_hh', bias_hh)
            
            # Initialize forget gate bias to 1.0 (helps with learning)
            forget_bias = MLX.zeros([4 * hidden_size])
            forget_slice = MLX.slice(forget_bias, [hidden_size], [hidden_size])
            forget_slice = MLX.add(forget_slice, 1.0)
            bias_ih_with_forget = MLX.update_slice(bias_ih, forget_slice, [hidden_size])
            register_parameter('bias_ih', bias_ih_with_forget)
          end
        end
        
        # Forward a single time step
        # @param x [MLX::Array] input tensor of shape (batch_size, input_size)
        # @param h_c [Array<MLX::Array>] tuple of (h, c) where:
        #   h is the hidden state of shape (batch_size, hidden_size)
        #   c is the cell state of shape (batch_size, hidden_size)
        # @return [Array<MLX::Array>] tuple of next (h, c) states
        def forward(x, h_c)
          h, c = h_c
          
          # Linear transformations
          gates = MLX.matmul(x, @_parameters['weight_ih'])
          gates = MLX.add(gates, MLX.matmul(h, @_parameters['weight_hh']))
          
          if @bias
            gates = MLX.add(gates, @_parameters['bias_ih'])
            gates = MLX.add(gates, @_parameters['bias_hh'])
          end
          
          # Split gates into chunks
          chunks = MLX.split(gates, 4, axis: 1)
          i = chunks[0]  # input gate
          f = chunks[1]  # forget gate
          g = chunks[2]  # cell gate
          o = chunks[3]  # output gate
          
          # Apply activations
          i = MLX.sigmoid(i)
          f = MLX.sigmoid(f)
          g = MLX.tanh(g)
          o = MLX.sigmoid(o)
          
          # Update cell state
          c_next = MLX.add(MLX.multiply(f, c), MLX.multiply(i, g))
          
          # Calculate output
          h_next = MLX.multiply(o, MLX.tanh(c_next))
          
          [h_next, c_next]
        end
        
        def reset_parameters
          # Initialize input-hidden weights
          @_parameters['weight_ih'] = MLX::NN::Init.xavier_uniform([@input_size, 4 * @hidden_size])
          
          # Initialize hidden-hidden weights
          @_parameters['weight_hh'] = MLX::NN::Init.xavier_uniform([@hidden_size, 4 * @hidden_size])
          
          if @bias
            # Initialize biases to zero
            @_parameters['bias_ih'] = MLX.zeros([4 * @hidden_size])
            @_parameters['bias_hh'] = MLX.zeros([4 * @hidden_size])
            
            # Set forget gate bias to 1.0
            forget_bias = MLX.zeros([4 * @hidden_size])
            forget_slice = MLX.slice(forget_bias, [@hidden_size], [@hidden_size])
            forget_slice = MLX.add(forget_slice, 1.0)
            bias_ih_with_forget = MLX.update_slice(@_parameters['bias_ih'], forget_slice, [@hidden_size])
            @_parameters['bias_ih'] = bias_ih_with_forget
          end
        end
      end
      
      # GRU Cell (single step)
      class GRUCell < MLX::NN::Module
        attr_reader :input_size, :hidden_size, :bias
        
        def initialize(input_size, hidden_size, bias: true)
          super()
          @input_size = input_size
          @hidden_size = hidden_size
          @bias = bias
          
          # Input to hidden weights (3x because of reset, update gates and candidate hidden state)
          weight_ih = MLX::NN::Init.xavier_uniform([input_size, 3 * hidden_size])
          register_parameter('weight_ih', weight_ih)
          
          # Hidden to hidden weights
          weight_hh = MLX::NN::Init.xavier_uniform([hidden_size, 3 * hidden_size])
          register_parameter('weight_hh', weight_hh)
          
          if bias
            # Input-hidden bias
            bias_ih = MLX.zeros([3 * hidden_size])
            register_parameter('bias_ih', bias_ih)
            
            # Hidden-hidden bias
            bias_hh = MLX.zeros([3 * hidden_size])
            register_parameter('bias_hh', bias_hh)
          end
        end
        
        # Forward a single time step
        # @param x [MLX::Array] input tensor of shape (batch_size, input_size)
        # @param h [MLX::Array] hidden state of shape (batch_size, hidden_size)
        # @return [MLX::Array] next hidden state
        def forward(x, h)
          # Linear transformations
          gates_x = MLX.matmul(x, @_parameters['weight_ih'])
          gates_h = MLX.matmul(h, @_parameters['weight_hh'])
          
          if @bias
            gates_x = MLX.add(gates_x, @_parameters['bias_ih'])
            gates_h = MLX.add(gates_h, @_parameters['bias_hh'])
          end
          
          # Split gates into chunks
          gates_x_chunks = MLX.split(gates_x, 3, axis: 1)
          gates_h_chunks = MLX.split(gates_h, 3, axis: 1)
          
          # Extract gates - r: reset gate, z: update gate, n: new gate
          x_r, x_z, x_n = gates_x_chunks
          h_r, h_z, h_n = gates_h_chunks
          
          # Compute reset and update gates with sigmoid
          r = MLX.sigmoid(MLX.add(x_r, h_r))
          z = MLX.sigmoid(MLX.add(x_z, h_z))
          
          # Compute candidate hidden state
          n = MLX.tanh(MLX.add(x_n, MLX.multiply(r, h_n)))
          
          # Compute new hidden state using update gate
          h_next = MLX.add(MLX.multiply(z, h), MLX.multiply(MLX.subtract(1.0, z), n))
          
          h_next
        end
        
        def reset_parameters
          # Initialize input-hidden weights
          @_parameters['weight_ih'] = MLX::NN::Init.xavier_uniform([@input_size, 3 * @hidden_size])
          
          # Initialize hidden-hidden weights
          @_parameters['weight_hh'] = MLX::NN::Init.xavier_uniform([@hidden_size, 3 * @hidden_size])
          
          if @bias
            # Initialize biases to zero
            @_parameters['bias_ih'] = MLX.zeros([3 * @hidden_size])
            @_parameters['bias_hh'] = MLX.zeros([3 * @hidden_size])
          end
        end
      end
      
      # Full RNN layer with sequence processing
      class RNN < MLX::NN::Module
        attr_reader :input_size, :hidden_size, :num_layers, :bias, :batch_first, :dropout, :bidirectional
        
        def initialize(input_size, hidden_size, num_layers: 1, bias: true, batch_first: false, dropout: 0, bidirectional: false)
          super()
          @input_size = input_size
          @hidden_size = hidden_size
          @num_layers = num_layers
          @bias = bias
          @batch_first = batch_first
          @dropout = dropout
          @bidirectional = bidirectional
          
          # Number of directions (1 for unidirectional, 2 for bidirectional)
          @num_directions = bidirectional ? 2 : 1
          
          # Create cells for each layer and direction
          @num_layers.times do |layer|
            layer_input_size = layer == 0 ? input_size : hidden_size * @num_directions
            
            # Forward direction
            forward_cell = RNNCell.new(layer_input_size, hidden_size, bias: bias)
            register_module("layer#{layer}_forward", forward_cell)
            
            # Backward direction (if bidirectional)
            if bidirectional
              backward_cell = RNNCell.new(layer_input_size, hidden_size, bias: bias)
              register_module("layer#{layer}_backward", backward_cell)
            end
          end
        end
        
        # Forward pass through the RNN
        # @param x [MLX::Array] input tensor of shape (seq_len, batch_size, input_size) or (batch_size, seq_len, input_size)
        # @param h_0 [MLX::Array, nil] initial hidden state of shape (num_layers * num_directions, batch_size, hidden_size)
        # @return [Array<MLX::Array>] output and final hidden state
        def forward(x, h_0 = nil)
          # Handle batch_first
          if @batch_first
            # (batch_size, seq_len, input_size) -> (seq_len, batch_size, input_size)
            x = MLX.transpose(x, [1, 0, 2])
          end
          
          # Get dimensions
          seq_len, batch_size, _ = x.shape
          
          # Initialize hidden state if not provided
          if h_0.nil?
            h_0 = MLX.zeros([@num_layers * @num_directions, batch_size, @hidden_size])
          end
          
          # Reshape h_0 to separate layers and directions
          h_0_reshaped = MLX.reshape(h_0, [@num_layers, @num_directions, batch_size, @hidden_size])
          
          # Process each layer
          layer_outputs = []
          layer_input = x
          
          @num_layers.times do |layer|
            # Initialize outputs for this layer
            layer_output = MLX.zeros([seq_len, batch_size, @hidden_size * @num_directions])
            
            # Get initial hidden states for this layer
            h_forward = MLX.slice(h_0_reshaped, [layer, 0], [1, 1])
            h_forward = MLX.reshape(h_forward, [batch_size, @hidden_size])
            
            # Forward pass
            forward_cell = @_submodules["layer#{layer}_forward"]
            
            # Process sequence forwards
            seq_len.times do |t|
              x_t = MLX.slice(layer_input, [t], [1])
              x_t = MLX.reshape(x_t, [batch_size, -1])
              h_forward = forward_cell.forward(x_t, h_forward)
              
              # Store output
              output_slice = MLX.slice(layer_output, [t], [1])
              output_forward = MLX.reshape(h_forward, [1, batch_size, @hidden_size])
              if @bidirectional
                # Only update the first half for forward direction
                output_slice = MLX.update_slice(output_slice, output_forward, [0, 0, 0])
              else
                # Update the entire slice
                output_slice = output_forward
              end
              layer_output = MLX.update_slice(layer_output, output_slice, [t, 0, 0])
            end
            
            # Backward pass if bidirectional
            if @bidirectional
              h_backward = MLX.slice(h_0_reshaped, [layer, 1], [1, 1])
              h_backward = MLX.reshape(h_backward, [batch_size, @hidden_size])
              
              backward_cell = @_submodules["layer#{layer}_backward"]
              
              # Process sequence backwards
              (seq_len - 1).downto(0) do |t|
                x_t = MLX.slice(layer_input, [t], [1])
                x_t = MLX.reshape(x_t, [batch_size, -1])
                h_backward = backward_cell.forward(x_t, h_backward)
                
                # Store output in second half of features
                output_slice = MLX.slice(layer_output, [t], [1])
                output_backward = MLX.reshape(h_backward, [1, batch_size, @hidden_size])
                
                # Update the second half for backward direction
                output_slice = MLX.update_slice(output_slice, output_backward, [0, 0, @hidden_size])
                layer_output = MLX.update_slice(layer_output, output_slice, [t, 0, 0])
              end
            end
            
            # Apply dropout (except for the last layer)
            if layer < @num_layers - 1 && @dropout > 0
              layer_output = MLX.dropout(layer_output, p: @dropout, training: @training)
            end
            
            # This layer's output becomes the next layer's input
            layer_input = layer_output
            layer_outputs << layer_output
          end
          
          # Final output comes from the last layer
          output = layer_outputs.last
          
          # Collect final hidden states from all layers
          h_n = MLX.zeros([@num_layers * @num_directions, batch_size, @hidden_size])
          
          @num_layers.times do |layer|
            if @bidirectional
              # For bidirectional, collect both directions
              h_forward = MLX.slice(layer_outputs[layer], [seq_len - 1], [1])
              h_forward = MLX.reshape(h_forward, [batch_size, @hidden_size * @num_directions])
              h_forward_only = MLX.slice(h_forward, [0], [@hidden_size])
              
              # The backward final state is the first element of the sequence
              h_backward = MLX.slice(layer_outputs[layer], [0], [1])
              h_backward = MLX.reshape(h_backward, [batch_size, @hidden_size * @num_directions])
              h_backward_only = MLX.slice(h_backward, [@hidden_size], [@hidden_size])
              
              # Set in h_n
              h_n = MLX.update_slice(h_n, MLX.reshape(h_forward_only, [1, batch_size, @hidden_size]), 
                                   [layer * 2, 0, 0])
              h_n = MLX.update_slice(h_n, MLX.reshape(h_backward_only, [1, batch_size, @hidden_size]), 
                                   [layer * 2 + 1, 0, 0])
            else
              # For unidirectional, just take the last output
              h_last = MLX.slice(layer_outputs[layer], [seq_len - 1], [1])
              h_last = MLX.reshape(h_last, [batch_size, @hidden_size])
              
              # Set in h_n
              h_n = MLX.update_slice(h_n, MLX.reshape(h_last, [1, batch_size, @hidden_size]), 
                                   [layer, 0, 0])
            end
          end
          
          # Convert back to batch_first if needed
          if @batch_first
            output = MLX.transpose(output, [1, 0, 2])
          end
          
          [output, h_n]
        end
      end
      
      # Full LSTM layer with sequence processing
      class LSTM < MLX::NN::Module
        attr_reader :input_size, :hidden_size, :num_layers, :bias, :batch_first, :dropout, :bidirectional
        
        def initialize(input_size, hidden_size, num_layers: 1, bias: true, batch_first: false, dropout: 0, bidirectional: false)
          super()
          @input_size = input_size
          @hidden_size = hidden_size
          @num_layers = num_layers
          @bias = bias
          @batch_first = batch_first
          @dropout = dropout
          @bidirectional = bidirectional
          
          # Number of directions (1 for unidirectional, 2 for bidirectional)
          @num_directions = bidirectional ? 2 : 1
          
          # Create cells for each layer and direction
          @num_layers.times do |layer|
            layer_input_size = layer == 0 ? input_size : hidden_size * @num_directions
            
            # Forward direction
            forward_cell = LSTMCell.new(layer_input_size, hidden_size, bias: bias)
            register_module("layer#{layer}_forward", forward_cell)
            
            # Backward direction (if bidirectional)
            if bidirectional
              backward_cell = LSTMCell.new(layer_input_size, hidden_size, bias: bias)
              register_module("layer#{layer}_backward", backward_cell)
            end
          end
        end
        
        # Forward pass through the LSTM
        # @param x [MLX::Array] input tensor of shape (seq_len, batch_size, input_size) or (batch_size, seq_len, input_size)
        # @param h_0_c_0 [Array<MLX::Array>, nil] tuple of initial (h_0, c_0) states, each of shape (num_layers * num_directions, batch_size, hidden_size)
        # @return [Array<MLX::Array>] output, (h_n, c_n) final states
        def forward(x, h_0_c_0 = nil)
          # Handle batch_first
          if @batch_first
            # (batch_size, seq_len, input_size) -> (seq_len, batch_size, input_size)
            x = MLX.transpose(x, [1, 0, 2])
          end
          
          # Get dimensions
          seq_len, batch_size, _ = x.shape
          
          # Initialize hidden and cell states if not provided
          if h_0_c_0.nil?
            h_0 = MLX.zeros([@num_layers * @num_directions, batch_size, @hidden_size])
            c_0 = MLX.zeros([@num_layers * @num_directions, batch_size, @hidden_size])
          else
            h_0, c_0 = h_0_c_0
          end
          
          # Reshape initial states to separate layers and directions
          h_0_reshaped = MLX.reshape(h_0, [@num_layers, @num_directions, batch_size, @hidden_size])
          c_0_reshaped = MLX.reshape(c_0, [@num_layers, @num_directions, batch_size, @hidden_size])
          
          # Process each layer
          layer_outputs = []
          layer_input = x
          
          # Collect final hidden and cell states for all layers
          h_n = MLX.zeros([@num_layers * @num_directions, batch_size, @hidden_size])
          c_n = MLX.zeros([@num_layers * @num_directions, batch_size, @hidden_size])
          
          @num_layers.times do |layer|
            # Initialize outputs for this layer
            layer_output = MLX.zeros([seq_len, batch_size, @hidden_size * @num_directions])
            
            # Get initial states for this layer
            h_forward = MLX.slice(h_0_reshaped, [layer, 0], [1, 1])
            h_forward = MLX.reshape(h_forward, [batch_size, @hidden_size])
            
            c_forward = MLX.slice(c_0_reshaped, [layer, 0], [1, 1])
            c_forward = MLX.reshape(c_forward, [batch_size, @hidden_size])
            
            # Forward pass
            forward_cell = @_submodules["layer#{layer}_forward"]
            
            # Process sequence forwards
            seq_len.times do |t|
              x_t = MLX.slice(layer_input, [t], [1])
              x_t = MLX.reshape(x_t, [batch_size, -1])
              h_forward, c_forward = forward_cell.forward(x_t, [h_forward, c_forward])
              
              # Store output
              output_slice = MLX.slice(layer_output, [t], [1])
              output_forward = MLX.reshape(h_forward, [1, batch_size, @hidden_size])
              if @bidirectional
                # Only update the first half for forward direction
                output_slice = MLX.update_slice(output_slice, output_forward, [0, 0, 0])
              else
                # Update the entire slice
                output_slice = output_forward
              end
              layer_output = MLX.update_slice(layer_output, output_slice, [t, 0, 0])
            end
            
            # Store final forward states
            h_n = MLX.update_slice(h_n, MLX.reshape(h_forward, [1, batch_size, @hidden_size]), 
                                 [layer * @num_directions, 0, 0])
            c_n = MLX.update_slice(c_n, MLX.reshape(c_forward, [1, batch_size, @hidden_size]), 
                                 [layer * @num_directions, 0, 0])
            
            # Backward pass if bidirectional
            if @bidirectional
              h_backward = MLX.slice(h_0_reshaped, [layer, 1], [1, 1])
              h_backward = MLX.reshape(h_backward, [batch_size, @hidden_size])
              
              c_backward = MLX.slice(c_0_reshaped, [layer, 1], [1, 1])
              c_backward = MLX.reshape(c_backward, [batch_size, @hidden_size])
              
              backward_cell = @_submodules["layer#{layer}_backward"]
              
              # Process sequence backwards
              (seq_len - 1).downto(0) do |t|
                x_t = MLX.slice(layer_input, [t], [1])
                x_t = MLX.reshape(x_t, [batch_size, -1])
                h_backward, c_backward = backward_cell.forward(x_t, [h_backward, c_backward])
                
                # Store output in second half of features
                output_slice = MLX.slice(layer_output, [t], [1])
                output_backward = MLX.reshape(h_backward, [1, batch_size, @hidden_size])
                
                # Update the second half for backward direction
                output_slice = MLX.update_slice(output_slice, output_backward, [0, 0, @hidden_size])
                layer_output = MLX.update_slice(layer_output, output_slice, [t, 0, 0])
              end
              
              # Store final backward states
              h_n = MLX.update_slice(h_n, MLX.reshape(h_backward, [1, batch_size, @hidden_size]), 
                                   [layer * @num_directions + 1, 0, 0])
              c_n = MLX.update_slice(c_n, MLX.reshape(c_backward, [1, batch_size, @hidden_size]), 
                                   [layer * @num_directions + 1, 0, 0])
            end
            
            # Apply dropout (except for the last layer)
            if layer < @num_layers - 1 && @dropout > 0
              layer_output = MLX.dropout(layer_output, p: @dropout, training: @training)
            end
            
            # This layer's output becomes the next layer's input
            layer_input = layer_output
            layer_outputs << layer_output
          end
          
          # Final output comes from the last layer
          output = layer_outputs.last
          
          # Convert back to batch_first if needed
          if @batch_first
            output = MLX.transpose(output, [1, 0, 2])
          end
          
          [output, [h_n, c_n]]
        end
      end
      
      # Full GRU layer with sequence processing
      class GRU < MLX::NN::Module
        attr_reader :input_size, :hidden_size, :num_layers, :bias, :batch_first, :dropout, :bidirectional
        
        def initialize(input_size, hidden_size, num_layers: 1, bias: true, batch_first: false, dropout: 0, bidirectional: false)
          super()
          @input_size = input_size
          @hidden_size = hidden_size
          @num_layers = num_layers
          @bias = bias
          @batch_first = batch_first
          @dropout = dropout
          @bidirectional = bidirectional
          
          # Number of directions (1 for unidirectional, 2 for bidirectional)
          @num_directions = bidirectional ? 2 : 1
          
          # Create cells for each layer and direction
          @num_layers.times do |layer|
            layer_input_size = layer == 0 ? input_size : hidden_size * @num_directions
            
            # Forward direction
            forward_cell = GRUCell.new(layer_input_size, hidden_size, bias: bias)
            register_module("layer#{layer}_forward", forward_cell)
            
            # Backward direction (if bidirectional)
            if bidirectional
              backward_cell = GRUCell.new(layer_input_size, hidden_size, bias: bias)
              register_module("layer#{layer}_backward", backward_cell)
            end
          end
        end
        
        # Forward pass through the GRU
        # @param x [MLX::Array] input tensor of shape (seq_len, batch_size, input_size) or (batch_size, seq_len, input_size)
        # @param h_0 [MLX::Array, nil] initial hidden state of shape (num_layers * num_directions, batch_size, hidden_size)
        # @return [Array<MLX::Array>] output and final hidden state
        def forward(x, h_0 = nil)
          # Handle batch_first
          if @batch_first
            # (batch_size, seq_len, input_size) -> (seq_len, batch_size, input_size)
            x = MLX.transpose(x, [1, 0, 2])
          end
          
          # Get dimensions
          seq_len, batch_size, _ = x.shape
          
          # Initialize hidden state if not provided
          if h_0.nil?
            h_0 = MLX.zeros([@num_layers * @num_directions, batch_size, @hidden_size])
          end
          
          # Reshape h_0 to separate layers and directions
          h_0_reshaped = MLX.reshape(h_0, [@num_layers, @num_directions, batch_size, @hidden_size])
          
          # Process each layer
          layer_outputs = []
          layer_input = x
          
          # Collect final hidden states for all layers
          h_n = MLX.zeros([@num_layers * @num_directions, batch_size, @hidden_size])
          
          @num_layers.times do |layer|
            # Initialize outputs for this layer
            layer_output = MLX.zeros([seq_len, batch_size, @hidden_size * @num_directions])
            
            # Get initial hidden states for this layer
            h_forward = MLX.slice(h_0_reshaped, [layer, 0], [1, 1])
            h_forward = MLX.reshape(h_forward, [batch_size, @hidden_size])
            
            # Forward pass
            forward_cell = @_submodules["layer#{layer}_forward"]
            
            # Process sequence forwards
            seq_len.times do |t|
              x_t = MLX.slice(layer_input, [t], [1])
              x_t = MLX.reshape(x_t, [batch_size, -1])
              h_forward = forward_cell.forward(x_t, h_forward)
              
              # Store output
              output_slice = MLX.slice(layer_output, [t], [1])
              output_forward = MLX.reshape(h_forward, [1, batch_size, @hidden_size])
              if @bidirectional
                # Only update the first half for forward direction
                output_slice = MLX.update_slice(output_slice, output_forward, [0, 0, 0])
              else
                # Update the entire slice
                output_slice = output_forward
              end
              layer_output = MLX.update_slice(layer_output, output_slice, [t, 0, 0])
            end
            
            # Store final forward state
            h_n = MLX.update_slice(h_n, MLX.reshape(h_forward, [1, batch_size, @hidden_size]), 
                                 [layer * @num_directions, 0, 0])
            
            # Backward pass if bidirectional
            if @bidirectional
              h_backward = MLX.slice(h_0_reshaped, [layer, 1], [1, 1])
              h_backward = MLX.reshape(h_backward, [batch_size, @hidden_size])
              
              backward_cell = @_submodules["layer#{layer}_backward"]
              
              # Process sequence backwards
              (seq_len - 1).downto(0) do |t|
                x_t = MLX.slice(layer_input, [t], [1])
                x_t = MLX.reshape(x_t, [batch_size, -1])
                h_backward = backward_cell.forward(x_t, h_backward)
                
                # Store output in second half of features
                output_slice = MLX.slice(layer_output, [t], [1])
                output_backward = MLX.reshape(h_backward, [1, batch_size, @hidden_size])
                
                # Update the second half for backward direction
                output_slice = MLX.update_slice(output_slice, output_backward, [0, 0, @hidden_size])
                layer_output = MLX.update_slice(layer_output, output_slice, [t, 0, 0])
              end
              
              # Store final backward state
              h_n = MLX.update_slice(h_n, MLX.reshape(h_backward, [1, batch_size, @hidden_size]), 
                                   [layer * @num_directions + 1, 0, 0])
            end
            
            # Apply dropout (except for the last layer)
            if layer < @num_layers - 1 && @dropout > 0
              layer_output = MLX.dropout(layer_output, p: @dropout, training: @training)
            end
            
            # This layer's output becomes the next layer's input
            layer_input = layer_output
            layer_outputs << layer_output
          end
          
          # Final output comes from the last layer
          output = layer_outputs.last
          
          # Convert back to batch_first if needed
          if @batch_first
            output = MLX.transpose(output, [1, 0, 2])
          end
          
          [output, h_n]
        end
      end
    end
  end
end 