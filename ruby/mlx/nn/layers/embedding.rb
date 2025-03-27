module MLX
  module NN
    module Layers
      # Embedding layer
      class Embedding < MLX::NN::Module
        attr_reader :num_embeddings, :embedding_dim, :padding_idx, :scale_grad_by_freq, :sparse
        
        def initialize(num_embeddings, embedding_dim, padding_idx: nil, scale_grad_by_freq: false, sparse: false)
          super()
          @num_embeddings = num_embeddings
          @embedding_dim = embedding_dim
          @padding_idx = padding_idx
          @scale_grad_by_freq = scale_grad_by_freq
          @sparse = sparse
          
          # Initialize weights with small random values
          scale = 1.0 / Math.sqrt(embedding_dim)
          weight = MLX::Random.uniform(-scale, scale, [num_embeddings, embedding_dim])
          
          # If padding_idx is provided, initialize to zeros
          if padding_idx
            zeros = MLX::Array.zeros([embedding_dim])
            weight[padding_idx] = zeros
          end
          
          register_parameter("weight", weight)
        end
        
        def forward(x)
          # Take embeddings according to the indices in x
          MLX.take(weight, x, axis: 0)
        end
        
        def reset_parameters
          scale = 1.0 / Math.sqrt(@embedding_dim)
          weight = MLX::Random.uniform(-scale, scale, [@num_embeddings, @embedding_dim])
          
          if @padding_idx
            zeros = MLX::Array.zeros([@embedding_dim])
            weight[@padding_idx] = zeros
          end
          
          register_parameter("weight", weight)
        end
      end
      
      # Weight-shared embedding and linear layers for language models
      class EmbeddingLinear < MLX::NN::Module
        attr_reader :num_embeddings, :embedding_dim
        
        def initialize(num_embeddings, embedding_dim)
          super()
          @num_embeddings = num_embeddings
          @embedding_dim = embedding_dim
          
          # Initialize weights with small random values
          scale = 1.0 / Math.sqrt(embedding_dim)
          weight = MLX::Random.uniform(-scale, scale, [num_embeddings, embedding_dim])
          register_parameter("weight", weight)
          
          # Optional bias for the linear part
          bias = MLX::Random.uniform(-scale, scale, [num_embeddings])
          register_parameter("bias", bias)
        end
        
        # Embedding forward pass
        def embedding_forward(x)
          MLX.take(weight, x, axis: 0)
        end
        
        # Linear forward pass (e.g., for language model head)
        def linear_forward(x)
          # x has shape [..., embedding_dim]
          # weight has shape [num_embeddings, embedding_dim]
          # result will have shape [..., num_embeddings]
          result = MLX.matmul(x, weight.transpose)
          result + bias
        end
        
        def forward(x, mode = :embedding)
          if mode == :embedding
            embedding_forward(x)
          elsif mode == :linear
            linear_forward(x)
          else
            raise ArgumentError, "Mode must be :embedding or :linear"
          end
        end
      end
    end
  end
end 