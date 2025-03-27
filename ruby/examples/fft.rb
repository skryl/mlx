require_relative '../mlx/mlx'

puts "MLX Ruby Bindings - FFT Example"
puts "------------------------------"

# Create a simple signal
puts "\nCreating a simple cosine signal:"
x = (0...64).to_a.map { |i| Ops.cos(2 * MLX.pi * i / 32.0) }
signal = MLX.array(x)
puts "Signal shape: #{signal.shape.inspect}"

# Compute the FFT
puts "\nComputing FFT:"
signal_fft = MLX::FFT.fft(signal)
puts "FFT shape: #{signal_fft.shape.inspect}"

# Find the dominant frequency
abs_fft = MLX.abs(signal_fft)
puts "FFT magnitude spectrum:\n#{abs_fft}"

# Compute the IFFT to get back the original signal
puts "\nComputing IFFT to recover the original signal:"
recovered_signal = MLX::FFT.ifft(signal_fft)
puts "Recovered signal shape: #{recovered_signal.shape.inspect}"

# Create a 2D array for 2D FFT
puts "\nCreating a 2D signal:"
arr_2d = MLX.reshape(MLX.array((0...16).to_a), [4, 4])
puts "2D signal:\n#{arr_2d}"

# Compute 2D FFT
puts "\nComputing 2D FFT:"
fft_2d = MLX::FFT.fft2(arr_2d)
puts "2D FFT:\n#{fft_2d}"

# Compute 2D IFFT
puts "\nComputing 2D IFFT to recover the original signal:"
recovered_2d = MLX::FFT.ifft2(fft_2d)
puts "Recovered 2D signal:\n#{recovered_2d}"

puts "\nExample completed!" 