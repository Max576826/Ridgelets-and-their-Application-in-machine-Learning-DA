classdef CustomSigmaLayer < nnet.layer.Layer
    properties (Learnable)
        d 
    end
    
    methods
        function layer = CustomSigmaLayer(name)
            
            layer.d = rand(); 
            
            
            if nargin > 0
                layer.Name = name;
            end
        end
        function Z = predict(layer, X)
            % Einbindung der Aktivierungsfunktion
            X = double(X);
            Z = layer.d*arrayfun(@sigma, X);
            if ~isa(Z, 'single')
                Z = single(Z);
            end
            
        end

        function [dLdX, dLdD] = backward(layer, X, Z, dLdZ, ~)
            % Backpropagation, Einbindung der Gradienten-Funktion
            X = single(X);
            grad_sigma = arrayfun(@d_sigma, X);
            grad_sigma = single(grad_sigma);

            % Kettenregel: dL/dX = dL/dZ * dZ/dX
            dLdX = dLdZ .* (layer.d * grad_sigma);

            % Gradient with respect to the learnable parameter 'd'
            dLdD = sum(dLdZ .* arrayfun(@sigma, X), 'all');
        end
       
    end
end