% Machine Vision Neural Network tutorial---Part 1: affine_layer
% Author: Daniel E. Worrall, 3 Dec 2016
%
% This script contains the class definition for an affine layer. It
% contains three functions a constructor, 'forward' and 'backward'. The
% constructor creates an appropriately sized and initialized weights
% matrix, which is stored as an object property obj.W. Forward and backward 
% compute the forward- and back-propagation steps respectively. You will 
% need to fill out the sections marked 'TODO'.

classdef affine_layer
    % The properties section lists the variables associated with this layer
    % which are stored whenever the forward or backward methods are called.
    properties
        W       % Weights and bias matrix (params)
        x       % Input
        y       % Output
        dLdW    % Gradient of loss wrt params
    end
    methods
        function obj = affine_layer(n_in, n_out)
            % Constructor
            % Initialise biases to 0.01
            b = 0.01*ones(1,n_out);
            % Initialise weights using stddev sqrt(1/n_in). This is known
            % as He initialisation (He et al., 2015), and is used to
            % prevent the variance of the forward- and back-prop passes
            % from exploding or vanishing to zero.
            obj.W = [sqrt(1/(n_in))*randn(n_in, n_out); b];
        end
        function [y, obj] = forward(obj, x)
            % Build the forward propagation step for an affine layer.
            
            % TODO 2.1: pad the input x with ones in the last (rightmost) dimension 
            % and compute affine transformation
            %x = 0;
            %y = 0;
            [nData,nDim] = size(x);
            ones_pad = ones(nData,1);
            x_padded = horzcat(x,ones_pad);
            y = x_padded*obj.W;
            % Store input/output to object
            obj.x = x_padded;
            obj.y = y;
        end
        function [dLdx, obj] = backward(obj, dLdy)
            % Compute the backpropagated gradients of this layer.
            
            % TODO 2.2: Implement the back-propagation step for the affine
            % layer. Remember to compute the gradient wrt the input 
            % (without bias), not the augmented input.
            [r,c] = size(obj.W);
            W_no_bias = obj.W(1:end-1,:);
            dydx = transpose(W_no_bias);
            dLdx = dLdy*dydx;
            
            % Store gradients to object
            obj.dLdW = obj.x'*dLdy;
        end
    end
end
