require_relative 'mlx_test_case'

class TestAttention < MLXTestCase
  def setup
    # Set random seed for reproducibility
    MLX.random.seed(42)
    
    # Common dimensions
    @batch_size = 2
    @seq_len = 4
    @hidden_size = 8
    @num_heads = 2
    @head_dim = @hidden_size / @num_heads
  end
  
  def test_scaled_dot_product_attention
    # Test the basic attention mechanism
    # Create query, key, value
    q = MLX.random.normal(shape: [@batch_size, @seq_len, @hidden_size])
    k = MLX.random.normal(shape: [@batch_size, @seq_len, @hidden_size])
    v = MLX.random.normal(shape: [@batch_size, @seq_len, @hidden_size])
    
    # Compute attention
    attn_output = MLX.nn.attention.scaled_dot_product_attention(q, k, v)
    
    # Check output shape
    assert_equal [@batch_size, @seq_len, @hidden_size], attn_output.shape
    
    # Test with attention mask
    # Create a causal mask (lower triangular)
    mask = MLX.nn.attention.create_causal_mask(@seq_len)
    attn_output = MLX.nn.attention.scaled_dot_product_attention(q, k, v, mask)
    
    # Check output shape
    assert_equal [@batch_size, @seq_len, @hidden_size], attn_output.shape
    
    # Test with custom scale factor
    scale = 1.0 / Math.sqrt(@hidden_size / 2)
    attn_output = MLX.nn.attention.scaled_dot_product_attention(q, k, v, scale: scale)
    assert_equal [@batch_size, @seq_len, @hidden_size], attn_output.shape
    
    # Test with dropout
    dropout_p = 0.1
    attn_output = MLX.nn.attention.scaled_dot_product_attention(q, k, v, dropout_p: dropout_p, training: true)
    assert_equal [@batch_size, @seq_len, @hidden_size], attn_output.shape
    
    # Test with different sequence lengths for key and value
    kv_seq_len = 6
    k = MLX.random.normal(shape: [@batch_size, kv_seq_len, @hidden_size])
    v = MLX.random.normal(shape: [@batch_size, kv_seq_len, @hidden_size])
    
    attn_output = MLX.nn.attention.scaled_dot_product_attention(q, k, v)
    assert_equal [@batch_size, @seq_len, @hidden_size], attn_output.shape
  end
  
  def test_multi_head_attention
    # Test the MultiHeadAttention module
    layer = MLX.nn.MultiHeadAttention(@hidden_size, @num_heads)
    
    # Check parameters
    assert_equal [@hidden_size, @hidden_size], layer.wq.shape
    assert_equal [@hidden_size, @hidden_size], layer.wk.shape
    assert_equal [@hidden_size, @hidden_size], layer.wv.shape
    assert_equal [@hidden_size, @hidden_size], layer.wo.shape
    
    # Test forward pass
    x = MLX.random.normal(shape: [@batch_size, @seq_len, @hidden_size])
    output = layer.call(x, x, x)
    
    # Check output shape
    assert_equal [@batch_size, @seq_len, @hidden_size], output.shape
    
    # Test with attention mask
    mask = MLX.nn.attention.create_causal_mask(@seq_len)
    output = layer.call(x, x, x, mask: mask)
    assert_equal [@batch_size, @seq_len, @hidden_size], output.shape
    
    # Test with custom attention implementation
    def custom_attention(q, k, v, mask)
      # A simplified version just to test custom implementation
      attn = MLX.matmul(q, MLX.transpose(k, axes: [0, 1, 3, 2]))
      attn = attn / Math.sqrt(@head_dim)
      
      if mask
        attn = MLX.where(mask == 0, MLX.full_like(attn, -1e9), attn)
      end
      
      attn = MLX.softmax(attn, axis: -1)
      return MLX.matmul(attn, v)
    end
    
    layer = MLX.nn.MultiHeadAttention(@hidden_size, @num_heads, attention_fn: custom_attention)
    output = layer.call(x, x, x)
    assert_equal [@batch_size, @seq_len, @hidden_size], output.shape
    
    # Test with bias
    layer = MLX.nn.MultiHeadAttention(@hidden_size, @num_heads, bias: true)
    
    # Should have bias parameters
    assert_equal [@hidden_size], layer.bq.shape
    assert_equal [@hidden_size], layer.bk.shape
    assert_equal [@hidden_size], layer.bv.shape
    assert_equal [@hidden_size], layer.bo.shape
    
    output = layer.call(x, x, x)
    assert_equal [@batch_size, @seq_len, @hidden_size], output.shape
    
    # Test with different key/value inputs
    key = MLX.random.normal(shape: [@batch_size, @seq_len + 2, @hidden_size])
    value = MLX.random.normal(shape: [@batch_size, @seq_len + 2, @hidden_size])
    
    output = layer.call(x, key, value)
    assert_equal [@batch_size, @seq_len, @hidden_size], output.shape
  end
  
  def test_self_attention
    # Test the SelfAttention module
    layer = MLX.nn.SelfAttention(@hidden_size, @num_heads)
    
    # Test forward pass
    x = MLX.random.normal(shape: [@batch_size, @seq_len, @hidden_size])
    output = layer.call(x)
    
    # Check output shape
    assert_equal [@batch_size, @seq_len, @hidden_size], output.shape
    
    # Test with attention mask
    mask = MLX.nn.attention.create_causal_mask(@seq_len)
    output = layer.call(x, mask: mask)
    assert_equal [@batch_size, @seq_len, @hidden_size], output.shape
    
    # Test with dropout
    layer = MLX.nn.SelfAttention(@hidden_size, @num_heads, dropout: 0.1)
    output = layer.call(x, training: true)
    assert_equal [@batch_size, @seq_len, @hidden_size], output.shape
  end
  
  def test_cross_attention
    # Test the CrossAttention module
    layer = MLX.nn.CrossAttention(@hidden_size, @num_heads)
    
    # Test forward pass
    x = MLX.random.normal(shape: [@batch_size, @seq_len, @hidden_size])
    context = MLX.random.normal(shape: [@batch_size, @seq_len + 2, @hidden_size])
    
    output = layer.call(x, context)
    
    # Check output shape - should be same as x
    assert_equal [@batch_size, @seq_len, @hidden_size], output.shape
    
    # Test with attention mask (cross attention mask)
    mask = MLX.zeros([@batch_size, @seq_len, @seq_len + 2])
    output = layer.call(x, context, mask: mask)
    assert_equal [@batch_size, @seq_len, @hidden_size], output.shape
    
    # Test with dropout
    layer = MLX.nn.CrossAttention(@hidden_size, @num_heads, dropout: 0.1)
    output = layer.call(x, context, training: true)
    assert_equal [@batch_size, @seq_len, @hidden_size], output.shape
  end
  
  def test_attention_mask_creation
    # Test causal mask creation
    mask = MLX.nn.attention.create_causal_mask(@seq_len)
    
    # Causal mask should be lower triangular with 1s on and below diagonal
    expected_shape = [@seq_len, @seq_len]
    assert_equal expected_shape, mask.shape
    
    # Check that mask is correct (1 for valid attention positions, 0 elsewhere)
    for i in 0...@seq_len
      for j in 0...@seq_len
        expected = j <= i ? 1 : 0
        assert_equal expected, mask[i, j].item
      end
    end
    
    # Test padding mask creation
    # Create a sequence with padding tokens
    seq_ids = MLX.array([[1, 2, 0, 0], [3, 4, 5, 0]])  # 0 is padding token
    padding_mask = MLX.nn.attention.create_padding_mask(seq_ids, padding_value: 0)
    
    # Shape should be [batch_size, 1, 1, seq_len]
    assert_equal [@batch_size, 1, 1, @seq_len], padding_mask.shape
    
    # Check mask values
    expected_mask = MLX.array([
      [[[ 1, 1, 0, 0 ]]], # First sequence has valid tokens at positions 0, 1
      [[[ 1, 1, 1, 0 ]]]  # Second sequence has valid tokens at positions 0, 1, 2
    ])
    
    assert MLX.array_equal(expected_mask, padding_mask)
    
    # Test combined mask
    combined_mask = MLX.nn.attention.combine_masks(mask, padding_mask)
    assert MLX.array_equal(combined_mask, mask * padding_mask)
  end
  
  def test_attention_gradients
    # Test that gradients flow correctly through attention
    @hidden_size = 4  # Use a smaller size for gradient testing
    
    q = MLX.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])  # [1, 2, 4]
    k = MLX.array([[[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0]]])  # [1, 2, 4]
    v = MLX.array([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]])  # [1, 2, 4]
    
    # Define a function that computes attention and sum of outputs
    def attention_fn(q, k, v)
      output = MLX.nn.attention.scaled_dot_product_attention(q, k, v)
      MLX.sum(output)
    end
    
    # Compute gradients with respect to all inputs
    grad_fn = MLX.grad(attention_fn, argnums: [0, 1, 2])
    grads = grad_fn.call(q, k, v)
    
    # Check gradient shapes
    grad_q, grad_k, grad_v = grads
    assert_equal q.shape, grad_q.shape
    assert_equal k.shape, grad_k.shape
    assert_equal v.shape, grad_v.shape
    
    # Test gradients with MultiHeadAttention
    def mha_fn(x, wq, wk, wv, wo)
      layer = MLX.nn.MultiHeadAttention(4, 2)
      # Override weights with our inputs
      layer.wq = wq
      layer.wk = wk  
      layer.wv = wv
      layer.wo = wo
      output = layer.call(x, x, x)
      MLX.sum(output)
    end
    
    x = MLX.random.normal(shape: [1, 2, 4])
    wq = MLX.random.normal(shape: [4, 4])
    wk = MLX.random.normal(shape: [4, 4])
    wv = MLX.random.normal(shape: [4, 4])
    wo = MLX.random.normal(shape: [4, 4])
    
    grad_fn = MLX.grad(mha_fn, argnums: [0, 1, 2, 3, 4])
    grads = grad_fn.call(x, wq, wk, wv, wo)
    
    # Check gradient shapes
    grad_x, grad_wq, grad_wk, grad_wv, grad_wo = grads
    assert_equal x.shape, grad_x.shape
    assert_equal wq.shape, grad_wq.shape
    assert_equal wk.shape, grad_wk.shape
    assert_equal wv.shape, grad_wv.shape
    assert_equal wo.shape, grad_wo.shape
  end
  
  def test_rotary_positional_embedding
    # Test Rotary Positional Embedding
    
    # Create input tensor
    x = MLX.random.normal(shape: [@batch_size, @seq_len, @hidden_size])
    
    # Apply rotary positional embedding
    rotary_emb = MLX.nn.attention.RotaryPositionalEmbedding(@hidden_size)
    output = rotary_emb.call(x)
    
    # Check output shape - should be same as input
    assert_equal x.shape, output.shape
    
    # Test with custom base frequency
    rotary_emb = MLX.nn.attention.RotaryPositionalEmbedding(@hidden_size, base: 1000000)
    output = rotary_emb.call(x)
    assert_equal x.shape, output.shape
    
    # Test with different seq_len positions
    positions = MLX.arange(@seq_len) + 10  # Start at position 10
    output = rotary_emb.call(x, positions: positions)
    assert_equal x.shape, output.shape
    
    # Test rotating only a portion of the hidden dimensions
    rotary_fraction = 0.5
    rotary_emb = MLX.nn.attention.RotaryPositionalEmbedding(@hidden_size, rotary_fraction: rotary_fraction)
    output = rotary_emb.call(x)
    assert_equal x.shape, output.shape
  end

  def test_relative_positional_embedding
    # Test Relative Positional Embedding
    
    # Create a multi-head attention with relative position embeddings
    layer = MLX.nn.MultiHeadAttention(@hidden_size, @num_heads, use_relative_position: true, max_distance: @seq_len)
    
    # Check parameters - should have relative position embeddings
    assert layer.has_relative_pe
    assert_equal [@seq_len * 2 - 1, @num_heads, @head_dim], layer.relative_pe.shape
    
    # Test forward pass
    x = MLX.random.normal(shape: [@batch_size, @seq_len, @hidden_size])
    output = layer.call(x, x, x)
    
    # Check output shape
    assert_equal [@batch_size, @seq_len, @hidden_size], output.shape
    
    # Test with longer sequence than max_distance
    long_seq_len = @seq_len * 2
    x_long = MLX.random.normal(shape: [@batch_size, long_seq_len, @hidden_size])
    output = layer.call(x_long, x_long, x_long)
    assert_equal [@batch_size, long_seq_len, @hidden_size], output.shape
  end
  
  def test_attention_with_qkv_projection
    # Test MultiHeadAttention with QKV projection
    layer = MLX.nn.MultiHeadAttention(@hidden_size, @num_heads)
    
    # Get the QKV projections
    x = MLX.random.normal(shape: [@batch_size, @seq_len, @hidden_size])
    q, k, v = layer.compute_qkv(x, x, x)
    
    # Check shapes
    # Each head has its own projection
    expected_shape = [@batch_size, @seq_len, @num_heads, @head_dim]
    assert_equal expected_shape, q.shape
    assert_equal expected_shape, k.shape
    assert_equal expected_shape, v.shape
    
    # Test with separate inputs for k, v
    kv_seq_len = @seq_len + 2
    key = MLX.random.normal(shape: [@batch_size, kv_seq_len, @hidden_size])
    value = MLX.random.normal(shape: [@batch_size, kv_seq_len, @hidden_size])
    
    q, k, v = layer.compute_qkv(x, key, value)
    
    # Check shapes
    assert_equal [@batch_size, @seq_len, @num_heads, @head_dim], q.shape
    assert_equal [@batch_size, kv_seq_len, @num_heads, @head_dim], k.shape
    assert_equal [@batch_size, kv_seq_len, @num_heads, @head_dim], v.shape
  end
  
  def test_flash_attention
    # Test Flash Attention implementation (if available)
    if MLX.nn.attention.respond_to?(:flash_attention)
      # Create inputs
      q = MLX.random.normal(shape: [@batch_size, @seq_len, @num_heads, @head_dim])
      k = MLX.random.normal(shape: [@batch_size, @seq_len, @num_heads, @head_dim])
      v = MLX.random.normal(shape: [@batch_size, @seq_len, @num_heads, @head_dim])
      
      # Run flash attention
      output = MLX.nn.attention.flash_attention(q, k, v)
      
      # Check output shape
      assert_equal [@batch_size, @seq_len, @num_heads, @head_dim], output.shape
      
      # Test with causal mask
      output = MLX.nn.attention.flash_attention(q, k, v, is_causal: true)
      assert_equal [@batch_size, @seq_len, @num_heads, @head_dim], output.shape
      
      # Test with different KV shape
      kv_seq_len = @seq_len * 2
      k = MLX.random.normal(shape: [@batch_size, kv_seq_len, @num_heads, @head_dim])
      v = MLX.random.normal(shape: [@batch_size, kv_seq_len, @num_heads, @head_dim])
      
      output = MLX.nn.attention.flash_attention(q, k, v)
      assert_equal [@batch_size, @seq_len, @num_heads, @head_dim], output.shape
    else
      # Skip test if flash attention is not available
      skip "Flash Attention not available in this MLX version"
    end
  end
end 