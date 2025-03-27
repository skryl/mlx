module MLX
  module NN
    module Layers
      # Multi-head Attention layer
      class MultiheadAttention < MLX::NN::Module
        attr_reader :embed_dim, :num_heads, :dropout, :kdim, :vdim, :head_dim, :bias, :add_bias_kv, :add_zero_attn
        
        def initialize(embed_dim, num_heads, dropout: 0.0, bias: true, add_bias_kv: false, add_zero_attn: false, kdim: nil, vdim: nil)
          super()
          @embed_dim = embed_dim
          @kdim = kdim || embed_dim
          @vdim = vdim || embed_dim
          @num_heads = num_heads
          @dropout = dropout
          @bias = bias
          @add_bias_kv = add_bias_kv
          @add_zero_attn = add_zero_attn
          
          if @embed_dim % @num_heads != 0
            raise ArgumentError, "embed_dim must be divisible by num_heads"
          end
          
          @head_dim = @embed_dim / @num_heads
          @scaling = (@head_dim) ** -0.5
          
          # Define projection weight matrices
          @q_proj = MLX::NN::Layers::Linear.new(@embed_dim, @embed_dim, bias: bias)
          @k_proj = MLX::NN::Layers::Linear.new(@kdim, @embed_dim, bias: bias)
          @v_proj = MLX::NN::Layers::Linear.new(@vdim, @embed_dim, bias: bias)
          @out_proj = MLX::NN::Layers::Linear.new(@embed_dim, @embed_dim, bias: bias)
          
          register_module('q_proj', @q_proj)
          register_module('k_proj', @k_proj)
          register_module('v_proj', @v_proj)
          register_module('out_proj', @out_proj)
          
          if @add_bias_kv
            # Additional bias for key and value
            bias_k = MLX::NN::Init.xavier_uniform([1, 1, @embed_dim])
            bias_v = MLX::NN::Init.xavier_uniform([1, 1, @embed_dim])
            
            register_parameter('bias_k', bias_k)
            register_parameter('bias_v', bias_v)
          end
        end
        
        # Forward pass for multi-head attention
        # @param query [MLX::Array] Query tensor
        # @param key [MLX::Array, nil] Key tensor (optional)
        # @param value [MLX::Array, nil] Value tensor (optional)
        # @param attn_mask [MLX::Array, nil] Attention mask (optional)
        # @param key_padding_mask [MLX::Array, nil] Key padding mask (optional)
        # @param need_weights [Boolean] Whether to return attention weights
        # @param static_kv [Boolean] Whether key and value are static
        # @return [Array<MLX::Array>] Output tensor and attention weights (optional)
        def forward(query, key: nil, value: nil, attn_mask: nil, key_padding_mask: nil, need_weights: true, static_kv: false)
          # Default key and value to query if not provided
          key = query if key.nil?
          value = key if value.nil?
          
          # Get dimensions
          tgt_len, bsz, embed_dim = query.shape
          src_len = key.shape[0]
          
          # Project query, key, value
          q = @q_proj.forward(query)
          k = @k_proj.forward(key)
          v = @v_proj.forward(value)
          
          # Scale query
          q = MLX.multiply(q, @scaling)
          
          # Handle bias_k and bias_v if present
          if @add_bias_kv
            # Expand bias to batch size
            bias_k = MLX.repeat(@_parameters['bias_k'], bsz, axis: 1)
            bias_v = MLX.repeat(@_parameters['bias_v'], bsz, axis: 1)
            
            # Concatenate to key and value
            k = MLX.concatenate([k, bias_k], axis: 0)
            v = MLX.concatenate([v, bias_v], axis: 0)
            
            # Update source length
            src_len += 1
          end
          
          # Add zero attention if requested
          if @add_zero_attn
            zero_attn_shape = [1, bsz, @embed_dim]
            k = MLX.concatenate([k, MLX.zeros(zero_attn_shape)], axis: 0)
            v = MLX.concatenate([v, MLX.zeros(zero_attn_shape)], axis: 0)
            
            # Update source length
            src_len += 1
            
            # Update attention mask
            if !attn_mask.nil?
              attn_mask_zeros = MLX.zeros([tgt_len, 1])
              attn_mask = MLX.concatenate([attn_mask, attn_mask_zeros], axis: 1)
            end
            
            # Update key padding mask
            if !key_padding_mask.nil?
              key_padding_mask_zeros = MLX.zeros([bsz, 1])
              key_padding_mask = MLX.concatenate([key_padding_mask, key_padding_mask_zeros], axis: 1)
            end
          end
          
          # Reshape for multi-head attention
          # [seq_len, batch_size, embed_dim] -> [seq_len, batch_size * num_heads, head_dim]
          q = MLX.reshape(q, [tgt_len, bsz * @num_heads, @head_dim])
          k = MLX.reshape(k, [src_len, bsz * @num_heads, @head_dim])
          v = MLX.reshape(v, [src_len, bsz * @num_heads, @head_dim])
          
          # Transpose for batched matrix multiplication
          # [seq_len, batch_size * num_heads, head_dim] -> [batch_size * num_heads, seq_len, head_dim]
          q = MLX.transpose(q, [1, 0, 2])
          k = MLX.transpose(k, [1, 0, 2])
          v = MLX.transpose(v, [1, 0, 2])
          
          # Calculate attention scores
          # [batch_size * num_heads, tgt_len, head_dim] x [batch_size * num_heads, head_dim, src_len]
          # -> [batch_size * num_heads, tgt_len, src_len]
          attn_weights = MLX.matmul(q, MLX.transpose(k, [0, 2, 1]))
          
          # Apply attention mask if provided
          if !attn_mask.nil?
            # Broadcast mask across batch and heads
            # [tgt_len, src_len] -> [batch_size * num_heads, tgt_len, src_len]
            mask = MLX.reshape(attn_mask, [1, tgt_len, src_len])
            mask = MLX.repeat(mask, bsz * @num_heads, axis: 0)
            
            # Add mask to attention weights (0 for valid positions, -inf for masked positions)
            masked_weights = MLX.where(mask, attn_weights, MLX.full_like(attn_weights, -Float::INFINITY))
            attn_weights = masked_weights
          end
          
          # Apply key padding mask if provided
          if !key_padding_mask.nil?
            # Reshape key padding mask to apply across heads
            # [batch_size, src_len] -> [batch_size, 1, src_len] -> [batch_size * num_heads, tgt_len, src_len]
            mask = MLX.reshape(key_padding_mask, [bsz, 1, src_len])
            mask = MLX.repeat(mask, 1, axis: 1)
            mask = MLX.repeat(mask, @num_heads, axis: 0)
            mask = MLX.reshape(mask, [bsz * @num_heads, tgt_len, src_len])
            
            # Apply mask (1 for padding positions, 0 for valid positions)
            masked_weights = MLX.where(mask, MLX.full_like(attn_weights, -Float::INFINITY), attn_weights)
            attn_weights = masked_weights
          end
          
          # Apply softmax to get attention probabilities
          attn_weights = MLX.softmax(attn_weights, axis: -1)
          
          # Apply dropout to attention weights
          if @dropout > 0.0 && @training
            attn_weights = MLX.dropout(attn_weights, p: @dropout, training: @training)
          end
          
          # Apply attention to values
          # [batch_size * num_heads, tgt_len, src_len] x [batch_size * num_heads, src_len, head_dim]
          # -> [batch_size * num_heads, tgt_len, head_dim]
          attn_output = MLX.matmul(attn_weights, v)
          
          # Reshape and permute back to original shape
          # [batch_size * num_heads, tgt_len, head_dim] -> [tgt_len, batch_size, embed_dim]
          attn_output = MLX.transpose(attn_output, [1, 0, 2])
          attn_output = MLX.reshape(attn_output, [tgt_len, bsz, @embed_dim])
          
          # Linear projection to output space
          attn_output = @out_proj.forward(attn_output)
          
          # Return attention weights if requested
          if need_weights
            # Average attention weights over heads
            # [batch_size * num_heads, tgt_len, src_len] -> [batch_size, num_heads, tgt_len, src_len]
            attn_weights = MLX.reshape(attn_weights, [bsz, @num_heads, tgt_len, src_len])
            # -> [batch_size, tgt_len, src_len]
            attn_weights = MLX.mean(attn_weights, axis: 1)
            return [attn_output, attn_weights]
          end
          
          [attn_output, nil]
        end
      end
      
      # Transformer Encoder Layer
      class TransformerEncoderLayer < MLX::NN::Module
        attr_reader :d_model, :nhead, :dim_feedforward, :dropout, :activation, :layer_norm_eps, :batch_first, :norm_first
        
        def initialize(d_model, nhead, dim_feedforward: 2048, dropout: 0.1, 
                      activation: 'relu', layer_norm_eps: 1e-5, batch_first: false,
                      norm_first: false)
          super()
          @d_model = d_model
          @nhead = nhead
          @dim_feedforward = dim_feedforward
          @dropout = dropout
          @activation = activation
          @layer_norm_eps = layer_norm_eps
          @batch_first = batch_first
          @norm_first = norm_first
          
          # Self-attention
          @self_attn = MultiheadAttention.new(d_model, nhead, dropout: dropout)
          
          # Feed-forward network
          @linear1 = MLX::NN::Layers::Linear.new(d_model, dim_feedforward)
          @dropout1 = dropout
          @linear2 = MLX::NN::Layers::Linear.new(dim_feedforward, d_model)
          
          # Layer normalization
          @norm1 = MLX::NN::Layers::LayerNorm.new([d_model], eps: layer_norm_eps)
          @norm2 = MLX::NN::Layers::LayerNorm.new([d_model], eps: layer_norm_eps)
          
          # Output dropout
          @dropout2 = dropout
          @dropout3 = dropout
          
          # Register modules
          register_module('self_attn', @self_attn)
          register_module('linear1', @linear1)
          register_module('linear2', @linear2)
          register_module('norm1', @norm1)
          register_module('norm2', @norm2)
        end
        
        def _get_activation_fn(activation)
          case activation.to_s.downcase
          when 'relu'
            ->(x) { MLX::NN::Layers::ActivationFunctions.relu(x) }
          when 'gelu'
            ->(x) { MLX::NN::Layers::ActivationFunctions.gelu(x) }
          when 'gelu_fast', 'fast_gelu'
            ->(x) { MLX::NN::Layers::ActivationFunctions.gelu_fast_approx(x) }
          when 'silu', 'swish'
            ->(x) { MLX::NN::Layers::ActivationFunctions.silu(x) }
          else
            raise ArgumentError, "Unknown activation function: #{activation}"
          end
        end
        
        def _sa_block(x, attn_mask, key_padding_mask)
          x, _ = @self_attn.forward(
            x, 
            attn_mask: attn_mask,
            key_padding_mask: key_padding_mask,
            need_weights: false
          )
          
          x = MLX.dropout(x, p: @dropout1, training: @training) if @dropout1 > 0
          x
        end
        
        def _ff_block(x)
          x = @linear1.forward(x)
          x = _get_activation_fn(@activation).call(x)
          x = MLX.dropout(x, p: @dropout2, training: @training) if @dropout2 > 0
          x = @linear2.forward(x)
          x = MLX.dropout(x, p: @dropout3, training: @training) if @dropout3 > 0
          x
        end
        
        def forward(src, src_mask: nil, src_key_padding_mask: nil)
          # Handle batch_first format
          if @batch_first
            # [batch_size, seq_len, d_model] -> [seq_len, batch_size, d_model]
            src = MLX.transpose(src, [1, 0, 2])
          end
          
          if @norm_first
            # Layer Norm -> Self Attention -> Add & Norm -> Feed Forward -> Add
            sa_output = MLX.add(src, _sa_block(@norm1.forward(src), src_mask, src_key_padding_mask))
            output = MLX.add(sa_output, _ff_block(@norm2.forward(sa_output)))
          else
            # Self Attention -> Add & Norm -> Feed Forward -> Add & Norm
            sa_output = _sa_block(src, src_mask, src_key_padding_mask)
            sa_output = @norm1.forward(MLX.add(src, sa_output))
            ff_output = _ff_block(sa_output)
            output = @norm2.forward(MLX.add(sa_output, ff_output))
          end
          
          # Restore batch_first format if needed
          if @batch_first
            # [seq_len, batch_size, d_model] -> [batch_size, seq_len, d_model]
            output = MLX.transpose(output, [1, 0, 2])
          end
          
          output
        end
      end
      
      # Transformer Encoder (stack of encoder layers)
      class TransformerEncoder < MLX::NN::Module
        attr_reader :d_model, :nhead, :num_layers, :dim_feedforward, :dropout, 
                   :activation, :layer_norm_eps, :batch_first, :norm_first
        
        def initialize(d_model, nhead, num_layers, dim_feedforward: 2048, dropout: 0.1, 
                      activation: 'relu', layer_norm_eps: 1e-5, batch_first: false,
                      norm_first: false)
          super()
          @d_model = d_model
          @nhead = nhead
          @num_layers = num_layers
          @dim_feedforward = dim_feedforward
          @dropout = dropout
          @activation = activation
          @layer_norm_eps = layer_norm_eps
          @batch_first = batch_first
          @norm_first = norm_first
          
          # Create the encoder layers
          @layers = []
          num_layers.times do |i|
            layer = TransformerEncoderLayer.new(
              d_model, 
              nhead, 
              dim_feedforward: dim_feedforward,
              dropout: dropout,
              activation: activation,
              layer_norm_eps: layer_norm_eps,
              batch_first: batch_first,
              norm_first: norm_first
            )
            @layers << layer
            register_module("layers.#{i}", layer)
          end
          
          # Add final normalization layer
          @norm = MLX::NN::Layers::LayerNorm.new([d_model], eps: layer_norm_eps)
          register_module('norm', @norm)
        end
        
        def forward(src, mask: nil, src_key_padding_mask: nil)
          output = src
          
          # Process through each encoder layer
          @layers.each do |layer|
            output = layer.forward(
              output,
              src_mask: mask,
              src_key_padding_mask: src_key_padding_mask
            )
          end
          
          # Apply final normalization
          output = @norm.forward(output)
          
          output
        end
      end
      
      # Transformer Decoder Layer
      class TransformerDecoderLayer < MLX::NN::Module
        attr_reader :d_model, :nhead, :dim_feedforward, :dropout, :activation, 
                   :layer_norm_eps, :batch_first, :norm_first
        
        def initialize(d_model, nhead, dim_feedforward: 2048, dropout: 0.1, 
                      activation: 'relu', layer_norm_eps: 1e-5, batch_first: false,
                      norm_first: false)
          super()
          @d_model = d_model
          @nhead = nhead
          @dim_feedforward = dim_feedforward
          @dropout = dropout
          @activation = activation
          @layer_norm_eps = layer_norm_eps
          @batch_first = batch_first
          @norm_first = norm_first
          
          # Self-attention
          @self_attn = MultiheadAttention.new(d_model, nhead, dropout: dropout)
          
          # Cross-attention (multihead attention over encoder output)
          @multihead_attn = MultiheadAttention.new(d_model, nhead, dropout: dropout)
          
          # Feed-forward network
          @linear1 = MLX::NN::Layers::Linear.new(d_model, dim_feedforward)
          @dropout1 = dropout
          @linear2 = MLX::NN::Layers::Linear.new(dim_feedforward, d_model)
          
          # Layer normalization
          @norm1 = MLX::NN::Layers::LayerNorm.new([d_model], eps: layer_norm_eps)
          @norm2 = MLX::NN::Layers::LayerNorm.new([d_model], eps: layer_norm_eps)
          @norm3 = MLX::NN::Layers::LayerNorm.new([d_model], eps: layer_norm_eps)
          
          # Dropout
          @dropout2 = dropout
          @dropout3 = dropout
          @dropout4 = dropout
          
          # Register modules
          register_module('self_attn', @self_attn)
          register_module('multihead_attn', @multihead_attn)
          register_module('linear1', @linear1)
          register_module('linear2', @linear2)
          register_module('norm1', @norm1)
          register_module('norm2', @norm2)
          register_module('norm3', @norm3)
        end
        
        def _get_activation_fn(activation)
          case activation.to_s.downcase
          when 'relu'
            ->(x) { MLX::NN::Layers::ActivationFunctions.relu(x) }
          when 'gelu'
            ->(x) { MLX::NN::Layers::ActivationFunctions.gelu(x) }
          when 'gelu_fast', 'fast_gelu'
            ->(x) { MLX::NN::Layers::ActivationFunctions.gelu_fast_approx(x) }
          when 'silu', 'swish'
            ->(x) { MLX::NN::Layers::ActivationFunctions.silu(x) }
          else
            raise ArgumentError, "Unknown activation function: #{activation}"
          end
        end
        
        def _sa_block(x, tgt_mask, tgt_key_padding_mask)
          x, _ = @self_attn.forward(
            x,
            attn_mask: tgt_mask,
            key_padding_mask: tgt_key_padding_mask,
            need_weights: false
          )
          
          x = MLX.dropout(x, p: @dropout1, training: @training) if @dropout1 > 0
          x
        end
        
        def _mha_block(x, memory, memory_mask, memory_key_padding_mask)
          x, _ = @multihead_attn.forward(
            x,
            key: memory,
            value: memory,
            attn_mask: memory_mask,
            key_padding_mask: memory_key_padding_mask,
            need_weights: false
          )
          
          x = MLX.dropout(x, p: @dropout2, training: @training) if @dropout2 > 0
          x
        end
        
        def _ff_block(x)
          x = @linear1.forward(x)
          x = _get_activation_fn(@activation).call(x)
          x = MLX.dropout(x, p: @dropout3, training: @training) if @dropout3 > 0
          x = @linear2.forward(x)
          x = MLX.dropout(x, p: @dropout4, training: @training) if @dropout4 > 0
          x
        end
        
        def forward(tgt, memory, tgt_mask: nil, memory_mask: nil, 
                   tgt_key_padding_mask: nil, memory_key_padding_mask: nil)
          # Handle batch_first format
          if @batch_first
            # [batch_size, seq_len, d_model] -> [seq_len, batch_size, d_model]
            tgt = MLX.transpose(tgt, [1, 0, 2])
            memory = MLX.transpose(memory, [1, 0, 2])
          end
          
          if @norm_first
            # Layer Norm -> Self Attention -> Add -> Layer Norm -> Cross Attention -> Add -> Layer Norm -> FF -> Add
            tgt2 = @norm1.forward(tgt)
            tgt2 = _sa_block(tgt2, tgt_mask, tgt_key_padding_mask)
            tgt = MLX.add(tgt, tgt2)
            
            tgt2 = @norm2.forward(tgt)
            tgt2 = _mha_block(tgt2, memory, memory_mask, memory_key_padding_mask)
            tgt = MLX.add(tgt, tgt2)
            
            tgt2 = @norm3.forward(tgt)
            tgt2 = _ff_block(tgt2)
            tgt = MLX.add(tgt, tgt2)
          else
            # Self Attention -> Add & Norm -> Cross Attention -> Add & Norm -> FF -> Add & Norm
            tgt2 = _sa_block(tgt, tgt_mask, tgt_key_padding_mask)
            tgt = @norm1.forward(MLX.add(tgt, tgt2))
            
            tgt2 = _mha_block(tgt, memory, memory_mask, memory_key_padding_mask)
            tgt = @norm2.forward(MLX.add(tgt, tgt2))
            
            tgt2 = _ff_block(tgt)
            tgt = @norm3.forward(MLX.add(tgt, tgt2))
          end
          
          # Restore batch_first format if needed
          if @batch_first
            # [seq_len, batch_size, d_model] -> [batch_size, seq_len, d_model]
            tgt = MLX.transpose(tgt, [1, 0, 2])
          end
          
          tgt
        end
      end
      
      # Transformer Decoder (stack of decoder layers)
      class TransformerDecoder < MLX::NN::Module
        attr_reader :d_model, :nhead, :num_layers, :dim_feedforward, :dropout, 
                   :activation, :layer_norm_eps, :batch_first, :norm_first
        
        def initialize(d_model, nhead, num_layers, dim_feedforward: 2048, dropout: 0.1, 
                      activation: 'relu', layer_norm_eps: 1e-5, batch_first: false,
                      norm_first: false)
          super()
          @d_model = d_model
          @nhead = nhead
          @num_layers = num_layers
          @dim_feedforward = dim_feedforward
          @dropout = dropout
          @activation = activation
          @layer_norm_eps = layer_norm_eps
          @batch_first = batch_first
          @norm_first = norm_first
          
          # Create the decoder layers
          @layers = []
          num_layers.times do |i|
            layer = TransformerDecoderLayer.new(
              d_model, 
              nhead, 
              dim_feedforward: dim_feedforward,
              dropout: dropout,
              activation: activation,
              layer_norm_eps: layer_norm_eps,
              batch_first: batch_first,
              norm_first: norm_first
            )
            @layers << layer
            register_module("layers.#{i}", layer)
          end
          
          # Add final normalization layer
          @norm = MLX::NN::Layers::LayerNorm.new([d_model], eps: layer_norm_eps)
          register_module('norm', @norm)
        end
        
        def forward(tgt, memory, tgt_mask: nil, memory_mask: nil, 
                   tgt_key_padding_mask: nil, memory_key_padding_mask: nil)
          output = tgt
          
          # Process through each decoder layer
          @layers.each do |layer|
            output = layer.forward(
              output,
              memory,
              tgt_mask: tgt_mask,
              memory_mask: memory_mask,
              tgt_key_padding_mask: tgt_key_padding_mask,
              memory_key_padding_mask: memory_key_padding_mask
            )
          end
          
          # Apply final normalization
          output = @norm.forward(output)
          
          output
        end
      end
      
      # Complete Transformer model (encoder + decoder)
      class Transformer < MLX::NN::Module
        attr_reader :d_model, :nhead, :num_encoder_layers, :num_decoder_layers, 
                   :dim_feedforward, :dropout, :activation, :layer_norm_eps, 
                   :batch_first, :norm_first
        
        def initialize(d_model, nhead, num_encoder_layers, num_decoder_layers, 
                      dim_feedforward: 2048, dropout: 0.1, activation: 'relu', 
                      layer_norm_eps: 1e-5, batch_first: false, norm_first: false)
          super()
          @d_model = d_model
          @nhead = nhead
          @num_encoder_layers = num_encoder_layers
          @num_decoder_layers = num_decoder_layers
          @dim_feedforward = dim_feedforward
          @dropout = dropout
          @activation = activation
          @layer_norm_eps = layer_norm_eps
          @batch_first = batch_first
          @norm_first = norm_first
          
          # Create encoder
          @encoder = TransformerEncoder.new(
            d_model, 
            nhead, 
            num_encoder_layers,
            dim_feedforward: dim_feedforward,
            dropout: dropout,
            activation: activation,
            layer_norm_eps: layer_norm_eps,
            batch_first: batch_first,
            norm_first: norm_first
          )
          
          # Create decoder if needed
          if num_decoder_layers > 0
            @decoder = TransformerDecoder.new(
              d_model, 
              nhead, 
              num_decoder_layers,
              dim_feedforward: dim_feedforward,
              dropout: dropout,
              activation: activation,
              layer_norm_eps: layer_norm_eps,
              batch_first: batch_first,
              norm_first: norm_first
            )
          else
            @decoder = nil
          end
          
          # Register modules
          register_module('encoder', @encoder)
          register_module('decoder', @decoder) if @decoder
          
          # Initialize parameters
          reset_parameters
        end
        
        def reset_parameters
          # Initialize parameters with a normal distribution
          parameters.each do |_, param|
            if param.ndim >= 2
              MLX::NN::Init.xavier_uniform(param.shape)
            end
          end
        end
        
        def forward(src, tgt, src_mask: nil, tgt_mask: nil, memory_mask: nil,
                   src_key_padding_mask: nil, tgt_key_padding_mask: nil, memory_key_padding_mask: nil)
          # Pass through encoder
          memory = @encoder.forward(
            src,
            mask: src_mask,
            src_key_padding_mask: src_key_padding_mask
          )
          
          # Return early if no decoder
          return memory if @decoder.nil?
          
          # Pass through decoder
          output = @decoder.forward(
            tgt,
            memory,
            tgt_mask: tgt_mask,
            memory_mask: memory_mask,
            tgt_key_padding_mask: tgt_key_padding_mask,
            memory_key_padding_mask: memory_key_padding_mask
          )
          
          output
        end
        
        # Generate square attention mask to prevent attention to future positions
        def generate_square_subsequent_mask(sz)
          mask = MLX.ones([sz, sz], dtype: MLX::FLOAT32)
          mask = MLX.triu(mask, 1)
          mask = MLX.where(mask == 1, 
                           MLX.full([sz, sz], -Float::INFINITY, dtype: MLX::FLOAT32), 
                           MLX.zeros([sz, sz], dtype: MLX::FLOAT32))
          mask
        end
      end
    end
  end
end 