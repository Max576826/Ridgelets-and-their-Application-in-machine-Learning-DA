classdef RidgeletConvLayer < nnet.layer.Layer
    properties (Learnable)
        % Learnable parameters: arrays for 'a', 'b', and 'u' (u is a unit vector)
        a  % Array of scaling parameters
        b  % Array of translation parameters
        u  % Array of unit vectors, one per kernel
    end
    
    properties
        KernelSize  % Size of each kernel
        NumKernels  % Number of kernels
    end
    
    methods
        function layer = RidgeletConvLayer(kernelSize, numKernels, name)
            % Constructor for custom Ridglet convolution layer
            layer.Name = name;
            layer.Description = "Grayscale convolution layer with multiple Mexican Hat wavelet kernels";
            
            layer.KernelSize = kernelSize;
            layer.NumKernels = numKernels;
            
            % Initialize learnable parameters
            layer.a = randn(1, numKernels);  % Array of scaling parameters 'a'
            layer.b = randn(1, numKernels);  % Array of translation parameters 'b'
            layer.u = randn(numKernels, 2);  % Array of random vectors 'u', one per kernel
            
            % Normalize 'u' vectors to be unit vectors
            for i = 1:numKernels
                layer.u(i, :) = layer.u(i, :) / norm(layer.u(i, :));
            end
        end
        
        function Z = predict(layer, X)
    % Forward pass (Prediction)
    [h, w, ~, batchSize] = size(X);  % Get input size
    numKernels = layer.NumKernels;  % Number of kernels
    
    % Preallocate output array (one channel per kernel)
    Z_numeric = zeros(h, w, numKernels, batchSize, 'like', extractdata(X));
    
    % Apply each kernel to the input
    for k = 1:numKernels
        % Generate the Mexican Hat kernel for the k-th set of parameters
        kernel = layer.generateMexicanHatKernel(layer.KernelSize, k);
        
        % Apply convolution for each image in the batch with the k-th kernel
        for i = 1:batchSize
            Z_numeric(:, :, k, i) = conv2(extractdata(X(:, :, 1, i)), kernel, 'same');
        end
    end
    
    % Convert the numeric output back to dlarray without specifying format
    Z = dlarray(Z_numeric);  % Remove 'SSCB' format
end

        
        function kernel = generateMexicanHatKernel(layer, kernelSize, k)
            % Generate the Mexican hat wavelet kernel based on the k-th set of learnable parameters
            % Initialize the kernel
            kernel = zeros(kernelSize, 'single');  % Use 'single' or 'double'
            
            % Extract learnable parameters for the k-th kernel
            a = layer.a(k);
            b = layer.b(k);
            u = layer.u(k, :)';  % Ensure 'u' is a column vector

            % Create a grid of coordinates for the kernel
            [x, y] = meshgrid(1:kernelSize(2), 1:kernelSize(1));
            
            % Reshape (x, y) into position vectors
            pos = [x(:), y(:)]';
            
            % Compute dot product for each position in the kernel
            dot_product = u' * pos;
            
            % Compute the Mexican Hat wavelet value for each position
            pos_u = dot_product - b;  % Apply translation parameter 'b'
            scaled_pos_u = pos_u / a;  % Scale by parameter 'a'
            wavelet_values = (1 - scaled_pos_u.^2) .* exp(-0.5 * scaled_pos_u.^2);
            
            % Reshape the values into the 2D kernel
            kernel(:) = wavelet_values;
        end
    end
end
