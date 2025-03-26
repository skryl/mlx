require_relative '../mlx/mlx'

puts "MLX Ruby Bindings - Linear Algebra Example"
puts "-------------------------------------------"

# Create arrays for linear algebra operations
puts "\nCreating sample matrices:"
matrix_a = MLX.reshape(MLX.array([1, 2, 3, 4]), [2, 2])
puts "Matrix A (2x2):\n#{matrix_a}"

matrix_b = MLX.reshape(MLX.array([5, 6, 7, 8]), [2, 2])
puts "Matrix B (2x2):\n#{matrix_b}"

# Matrix multiplication
puts "\nMatrix Multiplication:"
matmul_result = MLX.matmul(matrix_a, matrix_b)
puts "A * B:\n#{matmul_result}"

# Matrix inverse
puts "\nMatrix Inverse:"
inv_a = MLX::Linalg.inv(matrix_a)
puts "Inverse of A:\n#{inv_a}"

# Verify inverse: A * A^-1 should be identity
identity_check = MLX.matmul(matrix_a, inv_a)
puts "A * A^-1 (should be identity):\n#{identity_check}"

# Determinant
puts "\nDeterminant:"
det_a = MLX::Linalg.det(matrix_a)
puts "Determinant of A: #{det_a}"

# SVD decomposition
puts "\nSVD Decomposition:"
u, s, v = MLX::Linalg.svd(matrix_a)
puts "U:\n#{u}"
puts "Singular values: #{s}"
puts "V:\n#{v}"

# QR decomposition
puts "\nQR Decomposition:"
q, r = MLX::Linalg.qr(matrix_a)
puts "Q:\n#{q}"
puts "R:\n#{r}"

# Verify QR: Q * R should be original matrix
qr_check = MLX.matmul(q, r)
puts "Q * R (should be original matrix):\n#{qr_check}"

# Matrix norms
puts "\nMatrix Norms:"
frobenius_norm = MLX::Linalg.norm(matrix_a)
puts "Frobenius norm of A: #{frobenius_norm}"

# Using mathematical constants
puts "\nMathematical Constants:"
pi_arr = MLX.pi()
puts "Pi: #{pi_arr}"
e_arr = MLX.e()
puts "e: #{e_arr}"

puts "\nExample completed!" 