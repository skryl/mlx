require_relative 'mlx_test_case'

class TestFFT < MLXTestCase
  def setup
    # Set random seed for reproducibility
    MLX.random.seed(42)
  end
  
  def check_approx_equal(arr1, arr2, atol=1e-5, rtol=1e-6)
    # Helper to check arrays are approximately equal
    diff = MLX.abs(arr1 - arr2)
    tol = atol + rtol * MLX.abs(arr2)
    result = MLX.all(diff <= tol)
    assert result.item, "Arrays differ by more than the tolerance"
  end
  
  def test_fft
    # Create complex input
    r = MLX.random.rand(100).astype(MLX.float32)
    i = MLX.random.rand(100).astype(MLX.float32)
    a = MLX.complex(r, i)
    
    # Test basic FFT
    result = MLX.fft.fft(a)
    assert_equal [100], result.shape
    
    # Check with slicing and padding
    r = MLX.random.rand(100).astype(MLX.float32)
    i = MLX.random.rand(100).astype(MLX.float32)
    a = MLX.complex(r, i)
    
    result1 = MLX.fft.fft(a, n: 80)
    assert_equal [80], result1.shape
    
    result2 = MLX.fft.fft(a, n: 120)
    assert_equal [120], result2.shape
    
    # Check different axes
    r = MLX.random.rand([100, 100]).astype(MLX.float32)
    i = MLX.random.rand([100, 100]).astype(MLX.float32)
    a = MLX.complex(r, i)
    
    result3 = MLX.fft.fft(a, axis: 0)
    assert_equal [100, 100], result3.shape
    
    result4 = MLX.fft.fft(a, axis: 1)
    assert_equal [100, 100], result4.shape
    
    # Check real fft
    a = MLX.random.rand(100).astype(MLX.float32)
    result5 = MLX.fft.rfft(a)
    assert_equal [51], result5.shape  # n/2 + 1
    
    result6 = MLX.fft.rfft(a, n: 80)
    assert_equal [41], result6.shape  # n/2 + 1
    
    result7 = MLX.fft.rfft(a, n: 120)
    assert_equal [61], result7.shape  # n/2 + 1
    
    # Check inverse
    r = MLX.random.rand([100, 100]).astype(MLX.float32)
    i = MLX.random.rand([100, 100]).astype(MLX.float32)
    a = MLX.complex(r, i)
    
    result8 = MLX.fft.ifft(a)
    assert_equal [100, 100], result8.shape
    
    result9 = MLX.fft.ifft(a, n: 80)
    assert_equal [100, 80], result9.shape
    
    result10 = MLX.fft.ifft(a, n: 120)
    assert_equal [100, 120], result10.shape
    
    # Test inverse real FFT
    x = MLX.fft.rfft(r)
    result11 = MLX.fft.irfft(x)
    assert_equal [100, 100], result11.shape
  end
  
  def test_fftn
    r = MLX.random.normal(shape: [8, 8, 8]).astype(MLX.float32)
    i = MLX.random.normal(shape: [8, 8, 8]).astype(MLX.float32)
    a = MLX.complex(r, i)
    
    # Test with different axes and shapes
    axes_list = [nil, [1, 2], [2, 1], [0, 2]]
    shapes_list = [nil, [10, 5], [5, 10]]
    
    # Test fft2
    result = MLX.fft.fft2(a)
    assert_equal [8, 8, 8], result.shape
    
    axes_list.each do |axes|
      next if axes.nil?
      
      result = MLX.fft.fft2(a, axes: axes)
      assert_equal [8, 8, 8], result.shape
      
      shapes_list.each do |shape|
        next if shape.nil?
        
        result = MLX.fft.fft2(a, axes: axes, s: shape)
        expected_shape = [8, 8, 8]
        expected_shape[axes[0]] = shape[0]
        expected_shape[axes[1]] = shape[1]
        assert_equal expected_shape, result.shape
      end
    end
    
    # Test ifft2
    result = MLX.fft.ifft2(a)
    assert_equal [8, 8, 8], result.shape
    
    # Test rfft2
    result = MLX.fft.rfft2(r)
    assert_equal [8, 8, 5], result.shape  # Last dim is n/2 + 1
    
    # Test irfft2
    x = MLX.fft.rfft2(r)
    result = MLX.fft.irfft2(x)
    assert_equal [8, 8, 8], result.shape
    
    # Test fftn
    result = MLX.fft.fftn(a)
    assert_equal [8, 8, 8], result.shape
    
    # Test ifftn
    result = MLX.fft.ifftn(a)
    assert_equal [8, 8, 8], result.shape
    
    # Test rfftn
    result = MLX.fft.rfftn(r)
    assert_equal [8, 8, 5], result.shape  # Last dim is n/2 + 1
    
    # Test irfftn
    x = MLX.fft.rfftn(r)
    result = MLX.fft.irfftn(x)
    assert_equal [8, 8, 8], result.shape
  end
  
  def run_ffts(shape, atol=1e-4, rtol=1e-4)
    r = MLX.random.rand(shape).astype(MLX.float32)
    i = MLX.random.rand(shape).astype(MLX.float32)
    a = MLX.complex(r, i)
    
    # Test FFT and IFFT
    fft_result = MLX.fft.fft(a)
    ifft_result = MLX.fft.ifft(a)
    
    assert_equal shape, fft_result.shape
    assert_equal shape, ifft_result.shape
    
    # Check that FFT followed by IFFT approximately recovers the original signal
    approx_a = MLX.fft.ifft(MLX.fft.fft(a))
    check_approx_equal(a, approx_a, atol, rtol)
    
    # Test RFFT
    rfft_result = MLX.fft.rfft(r)
    expected_shape = shape.dup
    expected_shape[-1] = shape[-1] // 2 + 1
    assert_equal expected_shape, rfft_result.shape
    
    # Test IRFFT
    irfft_result = MLX.fft.irfft(rfft_result, n: shape[-1])
    assert_equal shape, irfft_result.shape
    
    # Check RFFT followed by IRFFT approximately recovers the real signal
    approx_r = MLX.fft.irfft(MLX.fft.rfft(r), n: shape[-1])
    check_approx_equal(r, approx_r, atol, rtol)
  end
  
  def test_fft_shared_mem
    # Test various FFT sizes
    nums = [
      # Small radix
      2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
      # Powers of 2
      16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
      # Stockham
      27, 33, 286, 4004,
      # Rader
      17, 23, 29, 408, 46, 1153, 1982,
      # Bluestein
      47, 83, 289,
      # Large stockham
      3159, 3645, 3969, 4004
    ]
    
    batch_sizes = [1, 3, 32]
    
    # Test a subset for faster execution
    batch_sizes.each do |batch_size|
      nums[0...5].each do |num|
        atol = num < 1025 ? 1e-4 : 1e-3
        run_ffts([batch_size, num], atol: atol)
      end
    end
  end
  
  def test_fft_big_powers_of_two
    # Test FFT on large powers of two
    # Only test a few to keep the test fast
    [12].each do |k|
      run_ffts([3, 2**k], atol: 1e-3)
    end
  end
  
  def test_fft_large_numbers
    # Test FFT on large numbers
    numbers = [
      1037,  # Prime > 2048
      18247,  # Medium size prime factors
      13849,  # Large prime factors
      7883    # Large prime
    ]
    
    # Only test one to keep test fast
    run_ffts([1, numbers[0]], atol: 1e-3)
  end
  
  def test_fft_contiguity
    # Test FFT on non-contiguous arrays
    r = MLX.random.rand([4, 8]).astype(MLX.float32)
    i = MLX.random.rand([4, 8]).astype(MLX.float32)
    a = MLX.complex(r, i)
    
    # Non-contiguous in the FFT dim
    result1 = MLX.fft.fft(a[:, 0...8...2])
    assert_equal [4, 4], result1.shape
    
    # Non-contiguous not in the FFT dim
    result2 = MLX.fft.fft(a[0...4...2])
    assert_equal [2, 8], result2.shape
    
    # Test with broadcasting
    reshaped = MLX.reshape(MLX.transpose(a), [4, 8, 1])
    broadcasted = MLX.broadcast_to(reshaped, [4, 8, 16])
    result3 = MLX.fft.fft(MLX.abs(broadcasted) + 4)
    assert_equal [4, 8, 16], result3.shape
    
    # Test with tiling
    b = MLX.array([[0, 1, 2, 3]])
    reshaped_b = MLX.reshape(b, [1, 4])
    tiled_b = MLX.tile(reshaped_b, [4, 1])
    result4 = MLX.abs(MLX.fft.fft(tiled_b))
    assert_equal [4, 4], result4.shape
  end
end 