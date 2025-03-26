require_relative 'mlx_test_case'

class TestConstants < MLXTestCase
  def test_constants_values
    # Check if MLX constants match expected values
    assert_in_delta(2.71828182845904523536028747135266249775724709369995, MLX.e, 1e-10)
    assert_in_delta(0.5772156649015328606065120900824024310421, MLX.euler_gamma, 1e-10)
    assert_equal(Float::INFINITY, MLX.inf)
    assert MLX.nan.nan?, "MLX.nan should be NaN"
    assert_nil MLX.newaxis
    assert_in_delta(3.1415926535897932384626433, MLX.pi, 1e-10)
  end
  
  def test_constants_availability
    # Check if MLX constants are available
    assert MLX.respond_to?(:e), "MLX should have 'e' constant"
    assert MLX.respond_to?(:euler_gamma), "MLX should have 'euler_gamma' constant"
    assert MLX.respond_to?(:inf), "MLX should have 'inf' constant"
    assert MLX.respond_to?(:nan), "MLX should have 'nan' constant"
    assert MLX.respond_to?(:newaxis), "MLX should have 'newaxis' constant"
    assert MLX.respond_to?(:pi), "MLX should have 'pi' constant"
  end
  
  def test_newaxis_for_reshaping_arrays
    # Test newaxis for reshaping arrays
    arr_1d = MLX.array([1, 2, 3, 4, 5])
    arr_2d_column = arr_1d[:, MLX.newaxis]
    expected_result = MLX.array([[1], [2], [3], [4], [5]])
    assert MLX.array_equal(arr_2d_column, expected_result)
  end
  
  def test_constants_in_computation
    # Test using constants in computations
    a = MLX.array([1.0, 2.0, 3.0])
    
    # Test with pi
    result = a * MLX.pi
    expected = MLX.array([
      1.0 * 3.1415926535897932384626433,
      2.0 * 3.1415926535897932384626433,
      3.0 * 3.1415926535897932384626433
    ])
    assert MLX.allclose(result, expected, atol: 1e-6)
    
    # Test with e
    result = a * MLX.e
    expected = MLX.array([
      1.0 * 2.71828182845904523536028747135266249775724709369995,
      2.0 * 2.71828182845904523536028747135266249775724709369995,
      3.0 * 2.71828182845904523536028747135266249775724709369995
    ])
    assert MLX.allclose(result, expected, atol: 1e-6)
  end
  
  def test_infinity_and_nan_operations
    # Test operations with infinity and NaN
    a = MLX.array([1.0, 2.0, 3.0])
    
    # Test infinity
    result = a * MLX.inf
    assert result[0].infinite?, "Should be infinite"
    assert result[1].infinite?, "Should be infinite"
    assert result[2].infinite?, "Should be infinite"
    
    # Test NaN
    result = a * MLX.nan
    assert result[0].nan?, "Should be NaN"
    assert result[1].nan?, "Should be NaN"
    assert result[2].nan?, "Should be NaN"
    
    # Test detection of inf and NaN
    arr_with_special = MLX.array([1.0, MLX.inf, MLX.nan, 4.0])
    assert MLX.isnan(arr_with_special[2]), "isnan should detect NaN"
    assert MLX.isinf(arr_with_special[1]), "isinf should detect Infinity"
    assert MLX.isfinite(arr_with_special[0]), "isfinite should detect finite number"
    assert !MLX.isfinite(arr_with_special[1]), "isfinite should detect Infinity as not finite"
    assert !MLX.isfinite(arr_with_special[2]), "isfinite should detect NaN as not finite"
  end
end 