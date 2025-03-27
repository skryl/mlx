require_relative 'mlx_test_case'
require 'math'

class TestFast < MLXTestCase
  def setup
    # Set random seed for reproducibility
    MLX.random.seed(42)
  end
  
  # Helper functions to match the Python implementation
  def rope_orig(x, dims, traditional, base, scale, offset, freqs=nil)
    offset = offset.item if offset.is_a?(MLX::Array)
    n = x.shape[-2] + offset
    dtype = x.dtype
    half_d = dims / 2
    positions = MLX.arange(offset, n, dtype: dtype) * scale
    
    if freqs.nil?
      inv_freqs = MLX.exp(
        -MLX.arange(0.0, half_d, dtype: dtype) * (Ops.log(base) / half_d)
      )
    else
      inv_freqs = (1.0 / freqs).astype(x.dtype)
    end
    
    theta = MLX.reshape(positions, [-1, 1]) * MLX.reshape(inv_freqs, [1, -1])
    costheta, sintheta = MLX.cos(theta), MLX.sin(theta)
    
    if traditional
      x1 = x[..., 0...dims, 2]
      x2 = x[..., 1...dims, 2]
      rx1 = x1 * costheta - x2 * sintheta
      rx2 = x1 * sintheta + x2 * costheta
      rx = MLX.concatenate([rx1[..., nil], rx2[..., nil]], axis: -1)
      
      if dims < x.shape[-1]
        rx = MLX.reshape(rx, [*x.shape[0...-1], dims])
        rx = MLX.concatenate([rx, x[..., dims...]], axis: -1)
      end
      
      return MLX.reshape(rx, x.shape)
    else
      x1 = x[..., 0...dims/2]
      x2 = x[..., dims/2...dims]
      rx1 = x1 * costheta - x2 * sintheta
      rx2 = x1 * sintheta + x2 * costheta
      
      if dims < x.shape[-1]
        rx = MLX.concatenate([rx1, rx2, x[..., dims...]], axis: -1)
      else
        rx = MLX.concatenate([rx1, rx2], axis: -1)
      end
      
      return rx
    end
  end
  
  def rms_norm(x, weight, eps)
    x = x.astype(MLX.float32)
    x = x * MLX.rsqrt(x.square.mean(-1, keepdims: true) + eps)
    weight * x.astype(weight.dtype)
  end
  
  def layer_norm(x, weight, bias, eps)
    ot = x.dtype
    x = x.astype(MLX.float32)
    mean = x.mean(axis: -1, keepdims: true)
    var = x.var(axis: -1, keepdims: true)
    x = (x - mean) * MLX.rsqrt(var + eps)
    x = x.astype(ot)
    
    x = x * weight if weight
    x = x + bias if bias
    
    x
  end
  
  def test_rope
    # Test RoPE (Rotary Position Embedding) implementation
    t = 4
    
    # Defaults: dims, dtype, base, scale, offset, traditional
    dims, dtype, base, scale, offset, traditional = 8, MLX.float32, 10000.0, 1.0, 0, false
    
    # Per dtype absolute tolerance
    tolerances = {
      MLX.float32 => 1e-6, 
      MLX.float16 => 1e-3, 
      MLX.bfloat16 => 1e-2
    }
    
    # Test parameters
    dtypes = [MLX.float32, MLX.float16, MLX.bfloat16]
    bases = [10000.0, 1000000.0]
    scales = [1.0, 2.0]
    offsets = [0, 3, MLX.array(3)]
    traditionals = [true, false]
    
    traditionals.each do |traditional|
      # Test different bases
      bases.each do |base|
        x = MLX.random.uniform(shape: [2, t, dims]).astype(dtype)
        rx = rope_orig(x, dims, traditional, base, scale, offset)
        rx_fast = MLX.fast.rope(
          x,
          dims,
          traditional: traditional,
          base: base,
          scale: scale,
          offset: offset
        )
        assert MLX.abs(rx - rx_fast).max < tolerances[dtype]
      end
      
      # Test different dtypes
      dtypes.each do |dtype|
        x = MLX.random.uniform(shape: [2, t, dims]).astype(dtype)
        ry = rope_orig(x.astype(MLX.float32), dims, traditional, base, scale, offset)
        rx = rope_orig(x, dims, traditional, base, scale, offset)
        rx_fast = MLX.fast.rope(
          x,
          dims,
          traditional: traditional,
          base: base,
          scale: scale,
          offset: offset
        )
        
        if dtype != MLX.float32
          assert MLX.abs(ry - rx_fast).max <= MLX.abs(ry - rx).max
        end
        
        assert MLX.abs(rx - rx_fast).max < tolerances[dtype]
      end
      
      # Test different offsets
      offsets.each do |offset|
        x = MLX.random.uniform(shape: [2, t, dims]).astype(dtype)
        rx = rope_orig(x, dims, traditional, base, scale, offset)
        rx_fast = MLX.fast.rope(
          x,
          dims,
          traditional: traditional,
          base: base,
          scale: scale,
          offset: offset
        )
        assert MLX.abs(rx - rx_fast).max < tolerances[dtype]
      end
      
      # Test different scales
      scales.each do |scale|
        x = MLX.random.uniform(shape: [2, t, dims]).astype(dtype)
        rx = rope_orig(x, dims, traditional, base, scale, offset)
        rx_fast = MLX.fast.rope(
          x,
          dims,
          traditional: traditional,
          base: base,
          scale: scale,
          offset: offset
        )
        assert MLX.abs(rx - rx_fast).max < tolerances[dtype]
      end
    end
    
    # Test transpose into rope
    x = MLX.random.uniform(shape: [1, 1, 4, dims]).swapaxes(1, 2)
    rx = rope_orig(x, dims, traditional, base, scale, offset)
    rx_fast = MLX.fast.rope(
      1.0 * x,  # multiply to allow donation
      dims,
      traditional: traditional,
      base: base,
      scale: scale,
      offset: offset
    )
    assert MLX.abs(rx - rx_fast).max < tolerances[MLX.float32]
    
    # Test raises with integer inputs
    x = (MLX.random.uniform(shape: [2, t, dims]) * 10).astype(MLX.int32)
    assert_raises(ArgumentError) do
      MLX.fast.rope(
        x,
        dims,
        traditional: traditional,
        base: base,
        scale: scale,
        offset: offset
      )
    end
  end
  
  def test_rope_with_freqs
    MLX.random.seed(0)
    
    # Check error cases
    t = 4
    dims = 8
    x = MLX.random.uniform(shape: [2, t, dims])
    
    # Incorrect frequency size
    assert_raises(ArgumentError) do
      freqs = MLX.random.uniform(shape: [dims - 1])
      MLX.fast.rope(
        x,
        dims,
        traditional: false,
        base: nil,
        scale: 1.0,
        offset: 0,
        freqs: freqs
      )
    end
    
    # Incorrect frequency shape
    assert_raises(ArgumentError) do
      freqs = MLX.random.uniform(shape: [1, dims])
      MLX.fast.rope(
        x,
        dims,
        traditional: false,
        base: nil,
        scale: 1.0,
        offset: 0,
        freqs: freqs
      )
    end
    
    # Correct frequency
    freqs = MLX.random.uniform(shape: [dims / 2])
    
    # Test with different dtypes
    tolerances = {MLX.float32 => 1e-5, MLX.float16 => 1e-2}
    [MLX.float32, MLX.float16].each do |dtype|
      x_typed = x.astype(dtype)
      rx = rope_orig(x_typed, dims, false, nil, 1.0, 0, freqs)
      rx_fast = MLX.fast.rope(
        x_typed,
        dims,
        traditional: false,
        base: nil,
        scale: 1.0,
        offset: 0,
        freqs: freqs
      )
      assert_equal dtype, rx.dtype
      assert MLX.abs(rx - rx_fast).max < tolerances[dtype]
    end
    
    # Test single vector
    x = MLX.random.uniform(shape: [1, 1, dims])
    rx = rope_orig(x, dims, false, nil, 1.0, 0, freqs)
    rx_fast = MLX.fast.rope(
      x,
      dims,
      traditional: false,
      base: nil,
      scale: 1.0,
      offset: 0,
      freqs: freqs
    )
    assert MLX.abs(rx - rx_fast).max < 1e-5
    
    # Test gradient with freqs
    f1 = lambda do |x, y|
      (rope_orig(x, dims, false, nil, 1.0, 0, freqs) * y).sum
    end
    
    f2 = lambda do |x, y|
      (MLX.fast.rope(
        x,
        dims,
        traditional: false,
        base: nil,
        scale: 1.0,
        offset: 0,
        freqs: freqs
      ) * y).sum
    end
    
    x = MLX.random.uniform(shape: [2, 4, dims])
    y = MLX.random.uniform(shape: [2, 4, dims])
    g1 = MLX.grad(f1).call(x, y)
    g2 = MLX.grad(f2).call(x, y)
    assert MLX.allclose(g1, g2, atol: 1e-5)
  end
  
  def test_rms_norm
    # Test RMS normalization
    
    # Per dtype absolute tolerance
    tolerances = {
      MLX.float32 => 1e-6,
      MLX.float16 => 1e-2,
      MLX.bfloat16 => 1e-2
    }
    
    # Test with different dimensions and types
    [64, 65].each do |dim|
      dtypes = [MLX.float32]
      dtypes += [MLX.float16, MLX.bfloat16] if MLX.metal.available?
      
      dtypes.each do |dtype|
        [1e-5, 1e-6].each do |eps|
          x = MLX.random.uniform(shape: [4, 8, dim]).astype(dtype)
          w = MLX.random.uniform(shape: [dim]).astype(dtype)
          
          rx = rms_norm(x, w, eps)
          rx_fast = MLX.fast.rms_norm(x, w, eps)
          
          assert MLX.abs(rx - rx_fast).max < tolerances[dtype]
          
          # Test with single vector
          x = MLX.random.uniform(shape: [dim]).astype(dtype)
          rx = rms_norm(x, w, eps)
          rx_fast = MLX.fast.rms_norm(x, w, eps)
          
          assert MLX.abs(rx - rx_fast).max < tolerances[dtype]
        end
      end
    end
    
    # Test with transposed input
    x = MLX.random.uniform(shape: [1, 32, 64])
    w = MLX.random.uniform(shape: [64])
    eps = 1e-5
    
    x_t = MLX.swapaxes(x, 0, 1)
    rx = rms_norm(x_t, w, eps)
    rx_fast = MLX.fast.rms_norm(x_t, w, eps)
    
    assert MLX.abs(rx - rx_fast).max < tolerances[MLX.float32]
    
    # Test with integer array (should raise)
    x = (MLX.ones([10]) * 1.5).astype(MLX.int32)
    w = MLX.ones([10])
    
    assert_raises(ArgumentError) do
      MLX.fast.rms_norm(x, w, eps)
    end
  end
  
  def test_layer_norm
    # Test layer normalization
    
    # Per dtype absolute tolerance
    tolerances = {
      MLX.float32 => 1e-6,
      MLX.float16 => 1e-2,
      MLX.bfloat16 => 1e-2
    }
    
    # Test with different dimensions and types
    [64, 65].each do |dim|
      dtypes = [MLX.float32]
      dtypes += [MLX.float16, MLX.bfloat16] if MLX.metal.available?
      
      dtypes.each do |dtype|
        [1e-5, 1e-6].each do |eps|
          x = MLX.random.uniform(shape: [4, 8, dim]).astype(dtype)
          w = MLX.random.uniform(shape: [dim]).astype(dtype)
          b = MLX.random.uniform(shape: [dim]).astype(dtype)
          
          rx = layer_norm(x, w, b, eps)
          rx_fast = MLX.fast.layer_norm(x, w, b, eps)
          
          assert MLX.abs(rx - rx_fast).max < tolerances[dtype]
          
          # Test with single vector
          x = MLX.random.uniform(shape: [dim]).astype(dtype)
          rx = layer_norm(x, w, b, eps)
          rx_fast = MLX.fast.layer_norm(x, w, b, eps)
          
          assert MLX.abs(rx - rx_fast).max < tolerances[dtype]
        end
      end
    end
    
    # Test with transposed input
    x = MLX.random.uniform(shape: [1, 32, 64])
    w = MLX.random.uniform(shape: [64])
    b = MLX.random.uniform(shape: [64])
    eps = 1e-5
    
    x_t = MLX.swapaxes(x, 0, 1)
    rx = layer_norm(x_t, w, b, eps)
    rx_fast = MLX.fast.layer_norm(x_t, w, b, eps)
    
    assert MLX.abs(rx - rx_fast).max < tolerances[MLX.float32]
    
    # Test with integer array (should raise)
    x = (MLX.ones([10]) * 1.5).astype(MLX.int32)
    w = MLX.ones([10])
    b = MLX.zeros([10])
    
    assert_raises(ArgumentError) do
      MLX.fast.layer_norm(x, w, b, eps)
    end
    
    # Test without weight and bias
    x = MLX.random.uniform(shape: [4, 8, 64])
    rx = layer_norm(x, nil, nil, eps)
    rx_fast = MLX.fast.layer_norm(x, nil, nil, eps)
    
    assert MLX.abs(rx - rx_fast).max < tolerances[MLX.float32]
  end
  
  def test_fast_transforms
    # Skip on CPU
    skip "Fast transforms not available on CPU" unless MLX.metal.available?
    
    # Test fast transform helper
    # These should be valid and not raise
    [
      MLX.fast.rope,
      MLX.fast.layer_norm,
      MLX.fast.rms_norm,
    ].each do |fun|
      assert fun.respond_to?(:call)
    end
    
    # Test transform function documentation
    assert MLX.fast.rope.to_s.include?("rope")
    assert MLX.fast.layer_norm.to_s.include?("layer_norm")
    assert MLX.fast.rms_norm.to_s.include?("rms_norm")
    
    # Try to get help on fast.rope
    help_text = MLX.fast.rope.to_s
    assert help_text.include?("rope")
    assert help_text.include?("traditional")
    assert help_text.include?("base")
  end
  
  def test_custom_kernel_basic
    # Test basic custom kernel functionality
    skip "Metal is not available" unless MLX.metal.available?
    
    kernel_code = <<-EOT
      #include <metal_stdlib>
      using namespace metal;
    
      kernel void add_one(device float* out,
                          device const float* in,
                          uint index [[thread_position_in_grid]]) {
        out[index] = in[index] + 1.0f;
      }
    EOT
    
    x = MLX.arange(10).astype(MLX.float32)
    y = MLX.zeros_like(x)
    
    metal_fn = MLX.metal.register(kernel_code, "add_one", 1)
    metal_fn.call(y, x, threads: 10)
    
    expected = x + 1
    assert MLX.array_equal(y, expected)
  end
  
  def test_custom_kernel_args
    # Test custom kernel with different argument types
    skip "Metal is not available" unless MLX.metal.available?
    
    kernel_code = <<-EOT
      #include <metal_stdlib>
      using namespace metal;
    
      kernel void scaled_add(device float* out,
                             device const float* in1,
                             device const float* in2,
                             device const float& scale,
                             uint index [[thread_position_in_grid]]) {
        out[index] = scale * (in1[index] + in2[index]);
      }
    EOT
    
    x = MLX.random.uniform(shape: [10]).astype(MLX.float32)
    y = MLX.random.uniform(shape: [10]).astype(MLX.float32)
    z = MLX.zeros_like(x)
    scale = MLX.array(2.0).astype(MLX.float32)
    
    metal_fn = MLX.metal.register(kernel_code, "scaled_add", 1)
    metal_fn.call(z, x, y, scale, threads: 10)
    
    expected = scale * (x + y)
    assert MLX.allclose(z, expected)
  end
  
  def test_custom_kernel_strides
    # Test custom kernel with arrays having different strides
    skip "Metal is not available" unless MLX.metal.available?
    
    kernel_code = <<-EOT
      #include <metal_stdlib>
      using namespace metal;
    
      kernel void add_arrays(device float* out,
                             device const float* in1,
                             device const float* in2,
                             uint index [[thread_position_in_grid]]) {
        out[index] = in1[index] + in2[index];
      }
    EOT
    
    # Test with transposed arrays
    x = MLX.random.uniform(shape: [5, 10]).astype(MLX.float32)
    y = MLX.random.uniform(shape: [10, 5]).transpose.astype(MLX.float32)
    z = MLX.zeros_like(x)
    
    metal_fn = MLX.metal.register(kernel_code, "add_arrays", 1)
    
    # This should throw an error as arrays have different strides
    assert_raises(ArgumentError) do
      metal_fn.call(z, x, y, threads: 50)
    end
    
    # But this should work (with copied arrays)
    x_copy = MLX.array(x.to_a)
    y_copy = MLX.array(y.to_a)
    z_copy = MLX.zeros_like(x_copy)
    
    metal_fn.call(z_copy, x_copy, y_copy, threads: 50)
    
    expected = x_copy + y_copy
    assert MLX.allclose(z_copy, expected)
  end
end 