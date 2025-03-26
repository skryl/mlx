require_relative 'mlx_test_case'

class TestUpsample < MLXTestCase
  def setup
    # Set random seed for reproducibility
    MLX.random.seed(42)
  end
  
  def test_upsample
    # Test upsampling with different parameters
    def run_upsample(n, c, idim, scale_factor, mode, align_corners)
      # Create input tensor (N, H, W, C)
      ih, iw = idim
      input = MLX.random.normal(shape: [n, ih, iw, c]).astype(MLX.float32)
      
      # Apply upsampling
      upsample = MLX.nn.Upsample(
        scale_factor: scale_factor,
        mode: mode,
        align_corners: align_corners
      )
      output = upsample.call(input)
      
      # Calculate expected output shape
      sf_h, sf_w = scale_factor
      expected_h = (ih * sf_h).to_i
      expected_w = (iw * sf_w).to_i
      
      # Check output shape
      assert_equal [n, expected_h, expected_w, c], output.shape
    end
    
    # Test different dimensions and scaling factors
    test_cases = [
      # N, C, [H, W], [scale_H, scale_W], mode, align_corners
      [1, 1, [2, 2], [1.0, 1.0], "linear", false],
      [1, 1, [2, 2], [1.5, 1.5], "linear", false],
      [1, 1, [2, 2], [2.0, 2.0], "linear", false],
      [2, 3, [4, 4], [0.5, 0.5], "linear", false],
      [2, 3, [7, 7], [2.0, 2.0], "linear", true],
      [2, 3, [10, 10], [0.2, 0.2], "linear", true],
      [2, 3, [11, 21], [3.0, 3.0], "cubic", false],
      [2, 3, [11, 21], [3.0, 2.0], "cubic", true]
    ]
    
    test_cases.each do |n, c, idim, scale_factor, mode, align_corners|
      run_upsample(n, c, idim, scale_factor, mode, align_corners)
    end
  end
  
  def test_upsample_nearest
    # Test nearest-neighbor upsampling
    n, c = 2, 3
    input = MLX.random.normal(shape: [n, 5, 5, c]).astype(MLX.float32)
    
    # Apply nearest upsampling
    upsample = MLX.nn.Upsample(scale_factor: [2.0, 2.0], mode: "nearest")
    output = upsample.call(input)
    
    # Check output shape
    assert_equal [n, 10, 10, c], output.shape
    
    # Test with integer scale factors
    scale_factors = [[1, 1], [2, 2], [3, 2], [1, 4]]
    
    scale_factors.each do |sf|
      upsample = MLX.nn.Upsample(scale_factor: sf, mode: "nearest")
      output = upsample.call(input)
      
      expected_h = (5 * sf[0]).to_i
      expected_w = (5 * sf[1]).to_i
      assert_equal [n, expected_h, expected_w, c], output.shape
    end
  end
  
  def test_upsample_1d_and_3d
    # Test 1D upsampling
    n, c = 2, 3
    input_1d = MLX.random.normal(shape: [n, 10, c]).astype(MLX.float32)
    
    upsample_1d = MLX.nn.Upsample(scale_factor: 2.0, mode: "linear")
    output_1d = upsample_1d.call(input_1d)
    
    # Check output shape
    assert_equal [n, 20, c], output_1d.shape
    
    # Test 3D upsampling
    input_3d = MLX.random.normal(shape: [n, 5, 5, 5, c]).astype(MLX.float32)
    
    upsample_3d = MLX.nn.Upsample(scale_factor: [2.0, 2.0, 2.0], mode: "trilinear")
    output_3d = upsample_3d.call(input_3d)
    
    # Check output shape
    assert_equal [n, 10, 10, 10, c], output_3d.shape
  end
  
  def test_upsample_size
    # Test upsampling with fixed output size instead of scale factor
    n, c = 2, 3
    input = MLX.random.normal(shape: [n, 7, 8, c]).astype(MLX.float32)
    
    # Test with explicit size
    upsample = MLX.nn.Upsample(size: [14, 16], mode: "linear")
    output = upsample.call(input)
    
    # Check output shape
    assert_equal [n, 14, 16, c], output.shape
    
    # Test with mixed dimensions (one doubled, one unchanged)
    upsample = MLX.nn.Upsample(size: [14, 8], mode: "linear")
    output = upsample.call(input)
    
    # Check output shape
    assert_equal [n, 14, 8, c], output.shape
  end
  
  def test_interpolation_modes
    # Test different interpolation modes
    n, c = 2, 3
    input = MLX.random.normal(shape: [n, 5, 5, c]).astype(MLX.float32)
    
    # Test all available modes
    modes = ["nearest", "linear", "bilinear", "cubic", "bicubic"]
    
    modes.each do |mode|
      # Skip modes not supported in the Ruby bindings
      next unless ["nearest", "linear", "cubic"].include?(mode)
      
      upsample = MLX.nn.Upsample(scale_factor: [2.0, 2.0], mode: mode)
      output = upsample.call(input)
      
      # Check output shape
      assert_equal [n, 10, 10, c], output.shape
    end
  end
  
  def test_functional_interface
    # Test the functional interface for upsampling
    n, c = 2, 3
    input = MLX.random.normal(shape: [n, 5, 5, c]).astype(MLX.float32)
    
    # Use functional interface
    output = MLX.nn.functional.interpolate(
      input,
      scale_factor: [2.0, 2.0],
      mode: "linear"
    )
    
    # Check output shape
    assert_equal [n, 10, 10, c], output.shape
    
    # Test with explicit size
    output = MLX.nn.functional.interpolate(
      input,
      size: [15, 15],
      mode: "linear"
    )
    
    # Check output shape
    assert_equal [n, 15, 15, c], output.shape
  end
end 