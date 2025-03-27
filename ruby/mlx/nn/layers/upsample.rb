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
          length = x.shape[-1]
          
          if @mode == 'nearest'
            # Use nearest neighbor upsampling
            scale = size.to_f / length
            indices = MLX::Array.arange(size).floor_divide(scale).to_i
            # Fix Python-style indexing
            return x[0...x.shape[0], 0...x.shape[1], indices]
          else
            # Use linear interpolation
            scale = size > 1 ? (length - 1) / (size - 1.0) : 0
            
            # Get indices and weights for interpolation
            idx_f = MLX::Array.arange(size) * scale
            idx_l = idx_f.floor.to_i
            idx_r = idx_l + 1
            idx_r = MLX.minimum(idx_r, MLX::Array.full_like(idx_r, length - 1))
            
            weight_r = idx_f - idx_l
            weight_l = 1 - weight_r
            
            # Get values and apply weights
            # Fix Python-style indexing
            left = x[0...x.shape[0], 0...x.shape[1], idx_l]
            right = x[0...x.shape[0], 0...x.shape[1], idx_r]
            
            return weight_l * left + weight_r * right
          end
        end
        
        def interpolate_2d(x, height, width)
          in_h, in_w = x.shape[-2], x.shape[-1]
          
          if @mode == 'nearest'
            # Use nearest neighbor upsampling
            h_scale = height.to_f / in_h
            w_scale = width.to_f / in_w
            
            h_indices = MLX::Array.arange(height).floor_divide(h_scale).to_i
            w_indices = MLX::Array.arange(width).floor_divide(w_scale).to_i
            
            # Fix Python-style indexing
            temp = x[0...x.shape[0], 0...x.shape[1], h_indices, 0...x.shape[3]]
            return temp[0...temp.shape[0], 0...temp.shape[1], 0...temp.shape[2], w_indices]
          elsif @mode == 'bilinear'
            # Use bilinear interpolation
            if height > 1
              h_scale = (in_h - 1) / (height - 1.0)
            else
              h_scale = 0
            end
            
            if width > 1
              w_scale = (in_w - 1) / (width - 1.0)
            else
              w_scale = 0
            end
            
            # Get y coordinates
            y_f = MLX::Array.arange(height) * h_scale
            y_l = y_f.floor.to_i
            y_h = y_l + 1
            y_h = MLX.minimum(y_h, MLX::Array.full_like(y_h, in_h - 1))
            
            # Get x coordinates
            x_f = MLX::Array.arange(width) * w_scale
            x_l = x_f.floor.to_i
            x_h = x_l + 1
            x_h = MLX.minimum(x_h, MLX::Array.full_like(x_h, in_w - 1))
            
            # Get weights
            w_yl = (y_f - y_l).reshape(height, 1)
            w_yh = (1 - w_yl)
            w_xl = (x_f - x_l).reshape(1, width)
            w_xh = (1 - w_xl)
            
            # Get the four corners and apply weights
            # Fix Python-style indexing
            temp1 = x[0...x.shape[0], 0...x.shape[1], y_l, 0...x.shape[3]]
            temp2 = x[0...x.shape[0], 0...x.shape[1], y_h, 0...x.shape[3]]
            
            x_ll = temp1[0...temp1.shape[0], 0...temp1.shape[1], 0...temp1.shape[2], x_l]  # lower left
            x_lh = temp1[0...temp1.shape[0], 0...temp1.shape[1], 0...temp1.shape[2], x_h]  # lower right
            x_hl = temp2[0...temp2.shape[0], 0...temp2.shape[1], 0...temp2.shape[2], x_l]  # upper left
            x_hh = temp2[0...temp2.shape[0], 0...temp2.shape[1], 0...temp2.shape[2], x_h]  # upper right
            
            # Apply bilinear interpolation formula
            output = (
              w_yh * w_xh * x_ll + 
              w_yh * w_xl * x_lh + 
              w_yl * w_xh * x_hl + 
              w_yl * w_xl * x_hh
            )
            
            return output
          else
            raise ArgumentError, "Unsupported interpolation mode: #{@mode} for 2D"
          end
        end
        
        def interpolate_3d(x, depth, height, width)
          in_d, in_h, in_w = x.shape[-3], x.shape[-2], x.shape[-1]
          
          if @mode == 'nearest'
            # Use nearest neighbor upsampling
            d_scale = depth.to_f / in_d
            h_scale = height.to_f / in_h
            w_scale = width.to_f / in_w
            
            d_indices = MLX::Array.arange(depth).floor_divide(d_scale).to_i
            h_indices = MLX::Array.arange(height).floor_divide(h_scale).to_i
            w_indices = MLX::Array.arange(width).floor_divide(w_scale).to_i
            
            # Fix Python-style indexing
            temp1 = x[0...x.shape[0], 0...x.shape[1], d_indices, 0...x.shape[3], 0...x.shape[4]]
            temp2 = temp1[0...temp1.shape[0], 0...temp1.shape[1], 0...temp1.shape[2], h_indices, 0...temp1.shape[4]]
            return temp2[0...temp2.shape[0], 0...temp2.shape[1], 0...temp2.shape[2], 0...temp2.shape[3], w_indices]
          else
            # Trilinear interpolation would be implemented here
            raise ArgumentError, "Unsupported interpolation mode: #{@mode} for 3D"
          end
        end
      end
    end
  end
end 