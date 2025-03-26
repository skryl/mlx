require_relative 'mlx_test_case'
require 'tmpdir'
require 'fileutils'

class TestLoad < MLXTestCase
  def setup
    # Set random seed for reproducibility
    MLX.random.seed(42)
    
    # Create a temporary directory for tests
    @test_dir = Dir.mktmpdir("mlx_test")
  end
  
  def teardown
    # Clean up temporary directory
    FileUtils.remove_entry(@test_dir)
  end
  
  def dtypes
    [
      "uint8",
      "uint16",
      "uint32",
      "uint64",
      "int8",
      "int16",
      "int32", 
      "int64",
      "float32",
      "float16",
      "complex64"
    ]
  end
  
  def test_save_and_load
    dtypes.each do |dt|
      dtype = MLX.send(dt)
      
      [[1], [23], [1024, 1024], [4, 6, 3, 1, 2]].each_with_index do |shape, i|
        save_file_mlx = File.join(@test_dir, "mlx_#{dt}_#{i}.npy")
        
        # Create random array
        save_arr = MLX.random.uniform(low: 0.0, high: 32.0, shape: shape).astype(dtype)
        
        # Save the array
        MLX.save(save_file_mlx, save_arr)
        
        # Load array saved by mlx as mlx array
        load_arr_mlx = MLX.load(save_file_mlx)
        assert MLX.array_equal(load_arr_mlx, save_arr)
      end
    end
  end
  
  def test_save_and_load_safetensors
    test_file = File.join(@test_dir, "test.safetensors")
    
    # Test with invalid metadata
    assert_raises(Exception) do
      MLX.save_safetensors(test_file, {"a" => MLX.ones([4, 4])}, {"testing" => 0})
    end
    
    # Test with valid metadata
    MLX.save_safetensors(
      test_file, 
      {"test" => MLX.ones([2, 2])}, 
      {"testing" => "test", "format" => "mlx"}
    )
    
    res = MLX.load(test_file, return_metadata: true)
    assert_equal 2, res.length
    assert_equal({"testing" => "test", "format" => "mlx"}, res[1])
    
    # Test with different data types and shapes
    (dtypes + ["bfloat16"]).each do |dt|
      dtype = MLX.send(dt)
      
      [[1], [23], [1024, 1024], [4, 6, 3, 1, 2]].each_with_index do |shape, i|
        save_file_mlx = File.join(@test_dir, "mlx_#{dt}_#{i}_fs.safetensors")
        
        # Create dictionary with test tensor
        if ["float32", "float16", "bfloat16"].include?(dt)
          save_dict = {"test" => MLX.random.normal(shape: shape, dtype: dtype)}
        else
          save_dict = {"test" => MLX.ones(shape, dtype: dtype)}
        end
        
        # Save the tensor dictionary
        File.open(save_file_mlx, "wb") do |f|
          MLX.save_safetensors(f, save_dict)
        end
        
        # Load the tensor dictionary
        load_dict = nil
        File.open(save_file_mlx, "rb") do |f|
          load_dict = MLX.load(f)
        end
        
        assert load_dict.key?("test")
        assert MLX.array_equal(load_dict["test"], save_dict["test"])
      end
    end
  end
  
  def test_save_and_load_gguf
    # Currently only a subset of dtypes are supported for GGUF
    supported_dtypes = ["float16", "float32", "int8", "int16", "int32"]
    
    supported_dtypes.each do |dt|
      dtype = MLX.send(dt)
      
      [[1], [23], [1024, 1024], [4, 6, 3, 1, 2]].each_with_index do |shape, i|
        save_file_mlx = File.join(@test_dir, "mlx_#{dt}_#{i}_fs.gguf")
        
        # Create dictionary with test tensor
        if ["float32", "float16", "bfloat16"].include?(dt)
          save_dict = {"test" => MLX.random.normal(shape: shape, dtype: dtype)}
        else
          save_dict = {"test" => MLX.ones(shape, dtype: dtype)}
        end
        
        # Save the tensor dictionary
        MLX.save_gguf(save_file_mlx, save_dict)
        
        # Load the tensor dictionary
        load_dict = MLX.load(save_file_mlx)
        
        assert load_dict.key?("test")
        assert MLX.array_equal(load_dict["test"], save_dict["test"])
      end
    end
  end
  
  def test_save_and_load_gguf_metadata_basic
    save_file_mlx = File.join(@test_dir, "mlx_gguf_with_metadata.gguf")
    save_dict = {"test" => MLX.ones([4, 4], dtype: MLX.int32)}
    metadata = {}
    
    # Empty works
    MLX.save_gguf(save_file_mlx, save_dict, metadata)
    
    # Loads without the metadata
    load_dict = MLX.load(save_file_mlx)
    assert load_dict.key?("test")
    assert MLX.array_equal(load_dict["test"], save_dict["test"])
    
    # Loads empty metadata
    load_dict, meta_load_dict = MLX.load(save_file_mlx, return_metadata: true)
    assert load_dict.key?("test")
    assert MLX.array_equal(load_dict["test"], save_dict["test"])
    assert_equal 0, meta_load_dict.length
    
    # Loads string metadata
    metadata = {"meta" => "data"}
    MLX.save_gguf(save_file_mlx, save_dict, metadata)
    load_dict, meta_load_dict = MLX.load(save_file_mlx, return_metadata: true)
    assert load_dict.key?("test")
    assert MLX.array_equal(load_dict["test"], save_dict["test"])
    assert_equal 1, meta_load_dict.length
    assert meta_load_dict.key?("meta")
    assert_equal "data", meta_load_dict["meta"]
  end
  
  def test_save_and_load_gguf_metadata_arrays
    save_file_mlx = File.join(@test_dir, "mlx_gguf_with_metadata.gguf")
    save_dict = {"test" => MLX.ones([4, 4], dtype: MLX.int32)}
    
    # Test scalars and one dimensional arrays
    [
      MLX.uint8, MLX.int8, MLX.uint16, MLX.int16, 
      MLX.uint32, MLX.int32, MLX.uint64, MLX.int64, 
      MLX.float32
    ].each do |dtype|
      [[], [2]].each do |shape|
        arr = MLX.random.uniform(shape: shape).astype(dtype)
        metadata = {"meta" => arr}
        MLX.save_gguf(save_file_mlx, save_dict, metadata)
        _, meta_load_dict = MLX.load(save_file_mlx, return_metadata: true)
        assert_equal 1, meta_load_dict.length
        assert meta_load_dict.key?("meta")
        assert MLX.array_equal(meta_load_dict["meta"], arr)
        assert_equal arr.dtype, meta_load_dict["meta"].dtype
      end
    end
    
    # Test unsupported metadata types
    [MLX.float16, MLX.bfloat16, MLX.complex64].each do |dtype|
      assert_raises(ArgumentError) do
        arr = MLX.array(1, dtype: dtype)
        metadata = {"meta" => arr}
        MLX.save_gguf(save_file_mlx, save_dict, metadata)
      end
    end
  end
  
  def test_save_and_load_gguf_metadata_mixed
    save_file_mlx = File.join(@test_dir, "mlx_gguf_with_metadata.gguf")
    save_dict = {"test" => MLX.ones([4, 4], dtype: MLX.int32)}
    
    # Test string and array
    arr = MLX.array(1.5)
    metadata = {"meta1" => arr, "meta2" => "data"}
    MLX.save_gguf(save_file_mlx, save_dict, metadata)
    _, meta_load_dict = MLX.load(save_file_mlx, return_metadata: true)
    assert_equal 2, meta_load_dict.length
    assert meta_load_dict.key?("meta1")
    assert MLX.array_equal(meta_load_dict["meta1"], arr)
    assert_equal arr.dtype, meta_load_dict["meta1"].dtype
    assert meta_load_dict.key?("meta2")
    assert_equal "data", meta_load_dict["meta2"]
    
    # Test list of strings
    metadata = {"meta" => ["data1", "data2", "data345"]}
    MLX.save_gguf(save_file_mlx, save_dict, metadata)
    _, meta_load_dict = MLX.load(save_file_mlx, return_metadata: true)
    assert_equal 1, meta_load_dict.length
    assert_equal metadata["meta"], meta_load_dict["meta"]
  end
  
  def test_savez_and_loadz
    save_file_mlx = File.join(@test_dir, "test.npz")
    
    # Create test arrays
    a = MLX.ones([4, 4])
    b = MLX.zeros([2, 8])
    c = MLX.random.normal(shape: [3, 3, 3])
    
    # Test with unnamed arrays
    MLX.savez(save_file_mlx, a, b, c)
    load_dict = MLX.loadz(save_file_mlx)
    
    assert_equal 3, load_dict.length
    assert load_dict.key?("arr_0")
    assert load_dict.key?("arr_1")
    assert load_dict.key?("arr_2")
    assert MLX.array_equal(load_dict["arr_0"], a)
    assert MLX.array_equal(load_dict["arr_1"], b)
    assert MLX.array_equal(load_dict["arr_2"], c)
    
    # Test with named arrays
    MLX.savez(save_file_mlx, a: a, b: b, c: c)
    load_dict = MLX.loadz(save_file_mlx)
    
    assert_equal 3, load_dict.length
    assert load_dict.key?("a")
    assert load_dict.key?("b")
    assert load_dict.key?("c")
    assert MLX.array_equal(load_dict["a"], a)
    assert MLX.array_equal(load_dict["b"], b)
    assert MLX.array_equal(load_dict["c"], c)
    
    # Test with mixed named and unnamed arrays
    MLX.savez(save_file_mlx, a, b: b, c: c)
    load_dict = MLX.loadz(save_file_mlx)
    
    assert_equal 3, load_dict.length
    assert load_dict.key?("arr_0")
    assert load_dict.key?("b")
    assert load_dict.key?("c")
    assert MLX.array_equal(load_dict["arr_0"], a)
    assert MLX.array_equal(load_dict["b"], b)
    assert MLX.array_equal(load_dict["c"], c)
  end
  
  def test_non_contiguous
    save_file_mlx = File.join(@test_dir, "test_non_contig.npy")
    
    # Create non-contiguous array
    a = MLX.ones([4, 4])
    non_contig = a.transpose
    
    # Ensure it's not contiguous
    assert !non_contig.flags.includes?(:contiguous)
    
    # Save and load the array
    MLX.save(save_file_mlx, non_contig)
    loaded = MLX.load(save_file_mlx)
    
    # Check that it equals the original
    assert MLX.array_equal(loaded, non_contig)
    
    # The loaded array should be contiguous though
    assert loaded.flags.includes?(:contiguous)
  end
  
  def test_load_donation
    save_file_mlx = File.join(@test_dir, "test_donation.npy")
    
    # Create a large array to test memory usage
    a = MLX.ones([1024, 1024])
    MLX.save(save_file_mlx, a)
    
    # Get current peak memory
    peak_mem = MLX.get_peak_memory
    
    # Load the array with donation
    loaded = MLX.load(save_file_mlx, donate: true)
    
    # Force evaluation
    MLX.eval(loaded)
    
    # Peak memory should be close to the previous level
    assert_in_delta peak_mem, MLX.get_peak_memory, peak_mem * 0.1
  end
end 