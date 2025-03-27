module MLX
  module NN
    module Layers
      # Sinusoidal Positional Encoding
      class SinusoidalPositionalEncoding < MLX::NN::Module
        attr_reader :embedding_dim, :max_len, :dropout
        
        def initialize(embedding_dim, max_len: 5000, dropout: 0.0)
          super()
          @embedding_dim = embedding_dim
          @max_len = max_len
          @dropout_val = dropout
          
          if dropout > 0
            @dropout = MLX::NN::Layers::Dropout.new(dropout)
          end
          
          # Create position encodings
          position = MLX::Array.arange(0, max_len).reshape(max_len, 1)
          div_term = MLX::Math.exp(
            MLX::Array.arange(0, embedding_dim, 2) * (-Math.log(10000.0) / embedding_dim)
          )
          
          pe = MLX::Array.zeros([max_len, embedding_dim])
          pe[:":", :"::2"] = MLX::Math.sin(position * div_term)
          pe[:":", :"1::2"] = MLX::Math.cos(position * div_term)
          pe = pe.reshape(1, max_len, embedding_dim)
          
          # Register as buffer (not a trainable parameter)
          register_buffer("pe", pe)
        end
        
        def forward(x)
          # x has shape [batch_size, seq_len, embedding_dim]
          seq_len = x.shape[1]
          
          if seq_len > @max_len
            raise ArgumentError, "Input sequence length #{seq_len} exceeds maximum length #{@max_len}"
          end
          
          # Use proper Ruby indexing notation
          pos_encoding = @pe[0...1, 0...seq_len, 0...@embedding_dim]
          result = x + pos_encoding
          
          @dropout ? @dropout.forward(result) : result
        end
      end
      
      # Rotary Position Embedding (RoPE)
      class RoPE < MLX::NN::Module
        attr_reader :dims, :traditional, :base, :scale
        
        def initialize(dims, traditional: false, base: 10000.0, scale: 1.0)
          super()
          @dims = dims
          @traditional = traditional
          @base = base
          @scale = scale
        end
        
        def forward(x)
          # Apply rotary positional embeddings to the input
          # Uses MLX Fast module to efficiently compute RoPE
          MLX::Fast.rope(x, @dims, @traditional, @base, @scale)
        end
        
        # In-place version of RoPE (more memory efficient)
        def forward_inplace(x)
          MLX::Fast.rope_inplace(x, @dims, @traditional, @base, @scale)
        end
      end
      
      # Attention with Linear Biases (ALiBi)
      class ALiBi < MLX::NN::Module
        attr_reader :num_heads, :mask_value
        
        def initialize(num_heads, mask_value: -1e9)
          super()
          @num_heads = num_heads
          @mask_value = mask_value
          
          # Create alibi slopes
          m = MLX::Array.arange(1, num_heads + 1)
          m = 2.0 ** -(2.0 ** (m - 1) / num_heads)
          register_buffer("alibi_slopes", m)
        end
        
        def get_mask(seq_len)
          # Create position bias
          # Returns a mask of shape [1, num_heads, seq_len, seq_len]
          
          # Create positions: [[0, 1, 2, ...], [0, 1, 2, ...], ...]
          pos = MLX::Array.arange(seq_len)
          # Create position differences
          pos_diff = pos.reshape(1, 1, seq_len, 1) - pos.reshape(1, 1, 1, seq_len)
          # Make position differences negative to penalize later positions
          neg_pos_diff = -pos_diff.abs
          
          # Scale by alibi slopes for each head
          slopes = @alibi_slopes.reshape(1, @num_heads, 1, 1)
          mask = neg_pos_diff * slopes
          
          # Add causal mask (upper triangular part)
          causal_mask = MLX::Array.triu(
            MLX::Array.full([seq_len, seq_len], @mask_value),
            k: 1
          )
          mask + causal_mask.reshape(1, 1, seq_len, seq_len)
        end
        
        def forward(q, k, v, attn_mask = nil)
          # Used in the context of multi-head attention
          # q, k, v: Query, key, value tensors [batch, num_heads, seq_len, head_dim]
          
          seq_len = q.shape[2]
          
          # Get ALiBi mask
          alibi_mask = get_mask(seq_len)
          
          # Combine with attention mask if provided
          if attn_mask
            alibi_mask = alibi_mask + attn_mask
          end
          
          # Standard scaled dot-product attention with ALiBi mask
          scores = MLX.matmul(q, k.transpose(0, 1, 3, 2))
          scores = scores / Math.sqrt(q.shape[-1])
          scores = scores + alibi_mask
          
          # Apply softmax and do attention
          attn_weights = MLX.softmax(scores, axis: -1)
          MLX.matmul(attn_weights, v)
        end
      end
    end
  end
end 