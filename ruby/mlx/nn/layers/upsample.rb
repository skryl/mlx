module MLX
  module NN
    module Layers
      # Upsample layer for increasing spatial dimensions
      class Upsample < MLX::NN::Module
        attr_reader :size, :scale_factor, :mode, :align_corners
        
        def initialize(size: nil, scale_factor: nil, mode: 'nearest', align_corners: nil)
          super()
          
          if size.nil? && scale_factor.nil?
            raise ArgumentError, "Either size or scale_factor should be defined"
          end
          
          if size && scale_factor
            raise ArgumentError, "Only one of size or scale_factor should be defined"
          end
          
          unless ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'].include?(mode)
            raise ArgumentError, "Upsampling mode #{mode} is not supported. Choose from 'nearest', 'linear', 'bilinear', 'bicubic', or 'trilinear'"
          end
          
          if mode != 'nearest' && align_corners.nil?
            align_corners = false
          end
          
          @size = size
          @scale_factor = scale_factor
          @mode = mode
          @align_corners = align_corners
        end
        
        def forward(x)
          # Compute the output size based on input shape and size/scale_factor
          input_shape = x.shape
          input_dim = input_shape.size - 2  # Subtract batch and channel dimensions
          
          # Determine output_size from either size or scale_factor
          output_size = nil
          
          if @size
            if @size.is_a?(Integer)
              output_size = [@size] * input_dim
            else
              output_size = @size
            end
          elsif @scale_factor
            if @scale_factor.is_a?(Integer) || @scale_factor.is_a?(Float)
              output_size = input_shape[2..].map { |s| (s * @scale_factor).to_i }
            else
              output_size = input_shape[2..].zip(@scale_factor).map { |s, f| (s * f).to_i }
            end
          end
          
          # Call the appropriate interpolation function based on the number of dimensions
          case input_dim
          when 1
            interpolate_1d(x, output_size[0])
          when 2
            interpolate_2d(x, output_size[0], output_size[1])
          when 3
            interpolate_3d(x, output_size[0], output_size[1], output_size[2])
          else
            raise ArgumentError, "Upsampling for #{input_dim}D tensors is not supported"
          end
        end
        
        private
        
        def interpolate_1d(x, size)
          batch_size, channels, length = x.shape
          
          if @mode == 'nearest'
            # Use nearest neighbor interpolation
            scale = size.to_f / length
            indices = MLX::Array.arange(size).floor_divide(scale).to_i
            return x[:, :, indices]
          else
            # Linear interpolation
            scale = size > 1 ? (length - 1) / (size - 1.0) : 0
            
            # Compute the indices and weights for linear interpolation
            idx_f = MLX::Array.arange(size) * scale
            idx_l = idx_f.floor.to_i
            idx_r = idx_l + 1
            idx_r = MLX::Core.minimum(idx_r, MLX::Array.full_like(idx_r, length - 1))
            
            weight_r = idx_f - idx_l
            weight_l = 1 - weight_r
            
            # Interpolate
            left = x[:, :, idx_l]
            right = x[:, :, idx_r]
            
            result = left * weight_l.reshape(1, 1, -1) + right * weight_r.reshape(1, 1, -1)
            result
          end
        end
        
        def interpolate_2d(x, height, width)
          batch_size, channels, in_h, in_w = x.shape
          
          if @mode == 'nearest'
            # Use nearest neighbor interpolation
            h_scale = height.to_f / in_h
            w_scale = width.to_f / in_w
            
            h_indices = MLX::Array.arange(height).floor_divide(h_scale).to_i
            w_indices = MLX::Array.arange(width).floor_divide(w_scale).to_i
            
            return x[:, :, h_indices, :][:, :, :, w_indices]
          elsif @mode == 'bilinear'
            # Bilinear interpolation
            if @align_corners && height > 1 && width > 1
              h_scale = (in_h - 1) / (height - 1.0)
              w_scale = (in_w - 1) / (width - 1.0)
            else
              h_scale = in_h / height.to_f
              w_scale = in_w / width.to_f
            end
            
            # Compute interpolation weights for height
            y_f = MLX::Array.arange(height) * h_scale
            y_l = y_f.floor.to_i
            y_h = y_l + 1
            y_h = MLX::Core.minimum(y_h, MLX::Array.full_like(y_h, in_h - 1))
            y_weight_h = y_f - y_l
            y_weight_l = 1 - y_weight_h
            
            # Compute interpolation weights for width
            x_f = MLX::Array.arange(width) * w_scale
            x_l = x_f.floor.to_i
            x_h = x_l + 1
            x_h = MLX::Core.minimum(x_h, MLX::Array.full_like(x_h, in_w - 1))
            x_weight_h = x_f - x_l
            x_weight_l = 1 - x_weight_h
            
            # Perform bilinear interpolation
            # Get the four points for each output pixel
            x_ll = x[:, :, y_l, :][:, :, :, x_l]  # lower left
            x_lh = x[:, :, y_l, :][:, :, :, x_h]  # lower right
            x_hl = x[:, :, y_h, :][:, :, :, x_l]  # upper left
            x_hh = x[:, :, y_h, :][:, :, :, x_h]  # upper right
            
            # Compute weighted sum
            # First along width, then along height
            x_l_interp = x_ll * x_weight_l.reshape(1, 1, 1, -1) + x_lh * x_weight_h.reshape(1, 1, 1, -1)
            x_h_interp = x_hl * x_weight_l.reshape(1, 1, 1, -1) + x_hh * x_weight_h.reshape(1, 1, 1, -1)
            x_interp = x_l_interp * y_weight_l.reshape(1, 1, -1, 1) + x_h_interp * y_weight_h.reshape(1, 1, -1, 1)
            
            return x_interp
          else
            raise ArgumentError, "Mode #{@mode} not implemented for 2D upsampling"
          end
        end
        
        def interpolate_3d(x, depth, height, width)
          # Similar implementation for 3D, expanding the 2D version with an additional dimension
          if @mode == 'nearest'
            # Nearest neighbor for 3D tensors
            batch_size, channels, in_d, in_h, in_w = x.shape
            
            d_scale = depth.to_f / in_d
            h_scale = height.to_f / in_h
            w_scale = width.to_f / in_w
            
            d_indices = MLX::Array.arange(depth).floor_divide(d_scale).to_i
            h_indices = MLX::Array.arange(height).floor_divide(h_scale).to_i
            w_indices = MLX::Array.arange(width).floor_divide(w_scale).to_i
            
            return x[:, :, d_indices, :, :][:, :, :, h_indices, :][:, :, :, :, w_indices]
          else
            raise ArgumentError, "Mode #{@mode} not implemented for 3D upsampling yet"
          end
        end
      end
    end
  end
end 