require_relative 'mlx_test_case'

class TestFastSDPA < MLXTestCase
  def setup
    # Set random seed for reproducibility
    MLX.random.seed(42)
  end
  
  # Helper functions to mimic the Python implementation
  def mlx_ref_attn(q, k, v, scale=1.0, mask=nil)
    q_dtype = q.dtype
    q = q * MLX.array(scale, dtype: q_dtype)
    n_q_heads = q.shape[-3]
    n_kv_heads = k.shape[-3]
    n_repeats = n_q_heads / n_kv_heads
    
    b = q.shape[0]
    l = q.shape[2]
    kl = k.shape[2]
    
    if n_repeats > 1
      q = MLX.reshape(q, [b, n_kv_heads, n_repeats, l, -1])
      k = MLX.expand_dims(k, 2)
      v = MLX.expand_dims(v, 2)
    end
    
    scores = q.matmul(MLX.swapaxes(k, -1, -2))
    
    unless mask.nil?
      if mask == "causal"
        q_offset = [0, kl - l].max
        q_indices = MLX.arange(q_offset, q_offset + l)
        k_indices = MLX.arange(kl)
        mask = q_indices.reshape([-1, 1]) >= k_indices.reshape([1, -1])
      end
      
      if n_repeats > 1 && mask.ndim >= 3
        if mask.shape[-3] == 1
          mask = MLX.expand_dims(mask, -3)
        else
          mask = MLX.unflatten(mask, -3, [n_kv_heads, n_repeats])
        end
      end
      
      if mask.dtype == MLX.bool
        scores = MLX.where(mask, scores, -Float::INFINITY)
      else
        scores += mask
      end
    end
    
    scores = MLX.softmax(scores, axis: -1, precise: true)
    
    out = scores.matmul(v)
    if n_repeats > 1
      out = MLX.reshape(out, [b, n_q_heads, l, -1])
    end
    
    out
  end
  
  def do_attention(f, q, k, v, scale, mask=nil, transpose=false)
    if transpose
      q_t = MLX.transpose(q, [0, 2, 1, 3])
      k_t = MLX.transpose(k, [0, 2, 1, 3])
      v_t = MLX.transpose(v, [0, 2, 1, 3])
      o_t = f.call(q_t, k_t, v_t, scale: scale, mask: mask)
      MLX.transpose(o_t, [0, 2, 1, 3])
    else
      f.call(q, k, v, scale: scale, mask: mask)
    end
  end
  
  def prepare_inputs(b, ql, kl, d, qh, kh, mask, transpose, dtype)
    MLX.random.seed(0)
    
    shape_q = transpose ? [b, ql, qh, d] : [b, qh, ql, d]
    shape_kv = transpose ? [b, kl, kh, d] : [b, kh, kl, d]
    
    scale = 1.0 / Ops.sqrt(d)
    
    q_mx = (MLX.random.normal(shape: shape_q) * 0.5).astype(dtype)
    k_mx = (MLX.random.normal(shape: shape_kv) * 0.5).astype(dtype)
    v_mx = (MLX.random.normal(shape: shape_kv) * scale).astype(dtype)
    
    if mask
      if mask == "additive"
        mask_mx = (MLX.random.normal(shape: [b, qh, ql, kl]) * 0.5).astype(dtype)
        mask = mask_mx
      elsif mask == "bool"
        mask_mx = MLX.random.uniform(shape: [b, qh, ql, kl]) < 0.5
        mask = mask_mx
      end
    end
    
    [q_mx, k_mx, v_mx, scale, mask]
  end
  
  # SDPA for MHA (n_heads == n_kv_heads)
  def mlx_primitives_sdpa(q, k, v, scale, mask=nil)
    p = (q * scale).matmul(k.transpose(0, 1, 3, 2))
    
    unless mask.nil?
      if mask.dtype == MLX.bool
        p = MLX.where(mask, p, MLX.finfo(MLX.float32).min)
      else
        p += mask
      end
    end
    
    scores = MLX.softmax(p.astype(MLX.float32), axis: -1).astype(p.dtype)
    scores.matmul(v)
  end
  
  # SDPA for GQA (n_heads > n_kv_heads, n_kv_heads > 1, n_heads % n_kv_heads == 0)
  def mlx_primitives_sdpa_with_gqa(q, k, v, scale, mask=nil)
    n_repeats = q.shape[1] / k.shape[1]
    
    # Borrowing KV cache tiling logic
    n_heads = q.shape[1]
    b = q.shape[0]
    l = k.shape[2]
    
    # Helper to repeat keys and values for each query head
    repeat = lambda do |a|
      a = MLX.concatenate(Array.new(n_repeats) { MLX.expand_dims(a, 2) }, axis: 2)
      a.reshape([b, n_heads, l, -1])
    end
    
    k, v = [k, v].map { |x| repeat.call(x) }
    
    mlx_primitives_sdpa(q, k, v, scale, mask: mask)
  end
  
  def test_fast_sdpa
    # Skip if Metal/fast module not available
    skip "Fast SDPA requires Metal support" unless MLX.metal.available?
    
    # Basic test with float32
    r = 20
    l = r
    dk = 64
    h = 3
    scale = 1.0 / Ops.sqrt(dk)
    
    q_mx = (MLX.random.normal(shape: [1, h, r, dk]) * 1.0).astype(MLX.float32)
    k_mx = (MLX.random.normal(shape: [1, h, r, dk]) * 1.0).astype(MLX.float32)
    v_mx = (MLX.random.normal(shape: [1, h, r, dk]) * 1.0).astype(MLX.float32)
    
    reference = mlx_primitives_sdpa(q_mx, k_mx, v_mx, scale)
    
    o_mlx = MLX.fast.scaled_dot_product_attention(
      q_mx, k_mx, v_mx, 
      scale: scale, 
      mask: nil
    )
    
    assert_equal reference.shape.to_a, o_mlx.shape.to_a
    assert MLX.allclose(o_mlx, reference, atol: 1e-4)
    
    # Test with different sequence lengths
    dtypes = [MLX.float32]
    dtypes << MLX.float16 if MLX.metal.available?
    
    dk = 64
    
    [63, 129, 400].each do |sequence_length|
      dtypes.each do |dtype|
        b = 2
        h = 24
        n_kv_heads = h
        
        q_mx = (MLX.random.normal(shape: [b, h, sequence_length, dk]) * 1.0).astype(dtype)
        k_mx = (MLX.random.normal(shape: [b, n_kv_heads, sequence_length, dk]) * 1.0).astype(dtype)
        v_mx = (MLX.random.normal(shape: [b, n_kv_heads, sequence_length, dk]) * 1.0).astype(dtype)
        
        reference = mlx_primitives_sdpa_with_gqa(q_mx, k_mx, v_mx, scale)
        
        o_mlx = MLX.fast.scaled_dot_product_attention(
          q_mx, k_mx, v_mx,
          scale: scale,
          memory_efficient_threshold: 2
        )
        
        assert_equal reference.shape.to_a, o_mlx.shape.to_a
        
        rtol = 1e-3
        atol = 1e-2
        
        rtol = 1e-2 if sequence_length > 500
        rtol = 1e-2 if dtype == MLX.float16
        
        assert MLX.allclose(o_mlx, reference, rtol: rtol, atol: atol)
      end
    end
  end
  
  def test_fast_sdpa_vector
    # Skip if Metal/fast module not available
    skip "Fast SDPA requires Metal support" unless MLX.metal.available?
    
    # Test with single vector query
    l = 43
    r = 1
    dk = 128
    scale = 1.0 / Ops.sqrt(128.0)
    
    q_mx = (MLX.random.normal(shape: [1, 32, r, dk]) * 1.0).astype(MLX.float32)
    k_mx = (MLX.random.normal(shape: [1, 32, l, dk]) * 1.0).astype(MLX.float32)
    v_mx = (MLX.random.normal(shape: [1, 32, l, dk]) * 1.0).astype(MLX.float32)
    
    reference = mlx_primitives_sdpa(q_mx, k_mx, v_mx, scale)
    
    o_mlx = MLX.fast.scaled_dot_product_attention(
      q_mx, k_mx, v_mx,
      scale: scale,
      mask: nil
    )
    
    assert_equal reference.shape.to_a, o_mlx.shape.to_a
    assert MLX.allclose(o_mlx, reference, atol: 1e-4)
    
    # Test with various sequence lengths and GQA configurations
    b = 1
    h = 32
    dtypes = [MLX.float32]
    dtypes << MLX.float16 if MLX.metal.available?
    
    sequence_lengths = [1, 7, 9, 32, 63, 67, 129, 400]
    gqa_options = [false, true]
    
    # Only test a subset for faster execution
    sequence_lengths.sample(3).each do |sequence_length|
      gqa_options.each do |do_gqa|
        dtypes.each do |dtype|
          n_kv_heads = do_gqa ? 8 : 32
          
          q_mx = (MLX.random.normal(shape: [b, h, r, dk]) * 1.0).astype(dtype)
          k_mx = (MLX.random.normal(shape: [b, n_kv_heads, sequence_length, dk]) * 1.0).astype(dtype)
          v_mx = (MLX.random.normal(shape: [b, n_kv_heads, sequence_length, dk]) * 1.0).astype(dtype)
          
          reference = mlx_primitives_sdpa_with_gqa(q_mx, k_mx, v_mx, scale)
          
          o_mlx = MLX.fast.scaled_dot_product_attention(
            q_mx, k_mx, v_mx,
            scale: scale
          )
          
          assert_equal reference.shape.to_a, o_mlx.shape.to_a
          
          # These tolerances match the Python tests
          rtol = 1e-5
          atol = 1e-1
          
          assert MLX.allclose(o_mlx, reference, rtol: rtol, atol: atol)
        end
      end
    end
  end
  
  def test_sdpa_with_mask
    # Skip if not on Metal
    skip "This test requires fast SDPA support" unless MLX.metal.available?
    
    # Test with both causal mask and boolean mask
    mask_types = ["causal", "bool"]
    dtypes = [MLX.float32]
    dtypes << MLX.float16 if MLX.metal.available?
    
    dtypes.each do |dtype|
      mask_types.each do |mask_type|
        b = 1
        h = 4
        q_length = 4
        kv_length = q_length
        dk = 32
        
        q, k, v, scale, mask = prepare_inputs(
          b, q_length, kv_length, dk, h, h, 
          mask_type, false, dtype
        )
        
        # Reference implementation
        ref_fn = lambda do |q, k, v, scale:, mask:|
          mlx_ref_attn(q, k, v, scale, mask)
        end
        
        # Fast implementation
        fast_fn = lambda do |q, k, v, scale:, mask:|
          MLX.fast.scaled_dot_product_attention(q, k, v, scale: scale, mask: mask)
        end
        
        # Execute both
        reference = do_attention(ref_fn, q, k, v, scale, mask)
        fast = do_attention(fast_fn, q, k, v, scale, mask)
        
        # Compare results
        rtol = dtype == MLX.float16 ? 1e-2 : 1e-4
        atol = dtype == MLX.float16 ? 1e-1 : 1e-3
        
        assert MLX.allclose(fast, reference, rtol: rtol, atol: atol)
      end
    end
  end
end 