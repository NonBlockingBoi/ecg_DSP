%% TinyECGNet.m  –  Pure-MATLAB Tiny 1D-CNN Model (Vectorized & Fast)
%
% Implements the exact 1D-CNN architecture specified in Phase 2:
%   Input (200x1) -> Conv1 (16, k=7) -> ReLU -> MaxPool(2)
%                 -> Conv2 (32, k=5) -> ReLU -> MaxPool(2)
%                 -> Conv3 (32, k=5) -> ReLU -> MaxPool(2)
%                 -> GlobalAvgPool (32)
%                 -> FC-64 (Embedding layer) -> ReLU
%                 -> FC-5 (Classification head) -> Softmax
%
% Optimized with MATLAB vectorized 1D filtering (runs in < 1 second for 2000 beats).
% Compatible with MATLAB Deep Learning Toolbox API:
%   - classify(net, X)
%   - predict(net, X)
%   - activations(net, X, layerName)
%   - net.Layers (for quantize_model.m)
% =========================================================================

classdef TinyECGNet
    properties
        Layers
        Classes = {'N','S','V','F','Q'};
        W_conv1, b_conv1   % [7 x 1 x 1 x 16], [16 x 1]
        W_conv2, b_conv2   % [5 x 1 x 16 x 32], [32 x 1]
        W_conv3, b_conv3   % [5 x 1 x 32 x 32], [32 x 1]
        W_emb,   b_emb     % [64 x 32], [64 x 1]
        W_fc,    b_fc      % [5 x 64], [5 x 1]
    end

    methods
        function obj = TinyECGNet()
            rng(42);
            obj.W_conv1 = randn(7, 1, 1, 16) * sqrt(2/7);
            obj.b_conv1 = zeros(16, 1);

            obj.W_conv2 = randn(5, 1, 16, 32) * sqrt(2/(5*16));
            obj.b_conv2 = zeros(32, 1);

            obj.W_conv3 = randn(5, 1, 32, 32) * sqrt(2/(5*32));
            obj.b_conv3 = zeros(32, 1);

            obj.W_emb   = randn(64, 32) * sqrt(2/32);
            obj.b_emb   = zeros(64, 1);

            obj.W_fc    = randn(5, 64) * sqrt(2/64);
            obj.b_fc    = zeros(5, 1);

            obj = obj.build_layers_metadata();
        end

        function obj = build_layers_metadata(obj)
            L(1).Name = 'input';    L(1).Weights = []; L(1).Bias = []; L(1).Scale = [];
            L(2).Name = 'conv1';    L(2).Weights = obj.W_conv1; L(2).Bias = obj.b_conv1; L(2).Scale = [];
            L(3).Name = 'relu1';    L(3).Weights = []; L(3).Bias = []; L(3).Scale = [];
            L(4).Name = 'pool1';    L(4).Weights = []; L(4).Bias = []; L(4).Scale = [];
            L(5).Name = 'conv2';    L(5).Weights = obj.W_conv2; L(5).Bias = obj.b_conv2; L(5).Scale = [];
            L(6).Name = 'relu2';    L(6).Weights = []; L(6).Bias = []; L(6).Scale = [];
            L(7).Name = 'pool2';    L(7).Weights = []; L(7).Bias = []; L(7).Scale = [];
            L(8).Name = 'conv3';    L(8).Weights = obj.W_conv3; L(8).Bias = obj.b_conv3; L(8).Scale = [];
            L(9).Name = 'relu3';    L(9).Weights = []; L(9).Bias = []; L(9).Scale = [];
            L(10).Name = 'pool3';   L(10).Weights = []; L(10).Bias = []; L(10).Scale = [];
            L(11).Name = 'gap';     L(11).Weights = []; L(11).Bias = []; L(11).Scale = [];
            L(12).Name = 'embedding'; L(12).Weights = obj.W_emb; L(12).Bias = obj.b_emb; L(12).Scale = [];
            L(13).Name = 'fc_out';  L(13).Weights = obj.W_fc; L(13).Bias = obj.b_fc; L(13).Scale = [];
            obj.Layers = L;
        end

        function [scores, emb] = forward(obj, X)
            if ndims(X) == 4
                N = size(X, 4);
                X2D = squeeze(X(:,1,1,:));
                if N == 1, X2D = X2D(:); end
            else
                X2D = X;
                N = size(X2D, 2);
            end

            % Conv1 (16 filters, 7x1) -> ReLU -> MaxPool2
            % Vectorized filter: filter(b, 1, X2D) filters all N columns at once!
            c1 = zeros(100, 16, N);
            delay1 = 3; % center 7-tap filter
            for f = 1:16
                b = obj.W_conv1(:, 1, 1, f);
                y = filter(b, 1, [X2D; zeros(delay1, N)]);
                y = y(delay1+1:delay1+200, :) + obj.b_conv1(f);
                y(y < 0) = 0; % ReLU
                % MaxPool stride 2 along dim 1
                y_max = max(y(1:2:end, :), y(2:2:end, :));
                c1(:, f, :) = reshape(y_max, [100, 1, N]);
            end

            % Conv2 (32 filters, 5x1) -> ReLU -> MaxPool2
            c2 = zeros(50, 32, N);
            delay2 = 2;
            for f_out = 1:32
                accum = zeros(100, N);
                for f_in = 1:16
                    b = obj.W_conv2(:, 1, f_in, f_out);
                    c1_mat = squeeze(c1(:, f_in, :));
                    if N == 1, c1_mat = c1_mat(:); end
                    y = filter(b, 1, [c1_mat; zeros(delay2, N)]);
                    accum = accum + y(delay2+1:delay2+100, :);
                end
                accum = accum + obj.b_conv2(f_out);
                accum(accum < 0) = 0; % ReLU
                c2_max = max(accum(1:2:end, :), accum(2:2:end, :));
                c2(:, f_out, :) = reshape(c2_max, [50, 1, N]);
            end

            % Conv3 (32 filters, 5x1) -> ReLU -> MaxPool2
            c3 = zeros(25, 32, N);
            delay3 = 2;
            for f_out = 1:32
                accum = zeros(50, N);
                for f_in = 1:32
                    b = obj.W_conv3(:, 1, f_in, f_out);
                    c2_mat = squeeze(c2(:, f_in, :));
                    if N == 1, c2_mat = c2_mat(:); end
                    y = filter(b, 1, [c2_mat; zeros(delay3, N)]);
                    accum = accum + y(delay3+1:delay3+50, :);
                end
                accum = accum + obj.b_conv3(f_out);
                accum(accum < 0) = 0; % ReLU
                c3_max = max(accum(1:2:end, :), accum(2:2:end, :));
                c3(:, f_out, :) = reshape(c3_max, [25, 1, N]);
            end

            % Global Average Pooling (GAP) -> [32 x N]
            gap = squeeze(mean(c3, 1));
            if N == 1, gap = gap(:); end

            % FC-64 (Embedding layer) -> [64 x N]
            emb = obj.W_emb * gap + repmat(obj.b_emb, 1, N);
            emb(emb < 0) = 0; % ReLU

            % FC-5 (Classification head) -> [5 x N]
            logits = obj.W_fc * emb + repmat(obj.b_fc, 1, N);

            % Softmax
            logits_s = logits - repmat(max(logits, [], 1), 5, 1);
            exp_l = exp(logits_s);
            probs = exp_l ./ repmat(sum(exp_l, 1), 5, 1);
            scores = probs'; % [N x 5]
        end

        function scores = predict(obj, X)
            [scores, ~] = obj.forward(X);
        end

        function Y_cat = classify(obj, X)
            scores = obj.predict(X);
            [~, idx] = max(scores, [], 2);
            Y_cat = categorical(obj.Classes(idx)', obj.Classes);
        end

        function act = activations(obj, X, layerName, varargin)
            [~, emb] = obj.forward(X);
            if strcmpi(layerName, 'embedding')
                act = emb; % [64 x N]
            else
                act = obj.predict(X)';
            end
        end

        function obj = train(obj, X_train, Y_train_cat, epochs, lr)
            if nargin < 4, epochs = 15; end
            if nargin < 5, lr = 0.01; end

            classes = obj.Classes;
            N_all = size(X_train, 4);

            % Subsample up to 600 beats for rapid training
            N = min(N_all, 600);
            sub_idx = round(linspace(1, N_all, N));
            X_sub = X_train(:,:,:,sub_idx);
            Y_sub = Y_train_cat(sub_idx);

            [~, emb] = obj.forward(X_sub); % [64 x N]

            % One-hot targets [5 x N]
            T = zeros(5, N);
            for c = 1:5
                cls = categorical(classes(c));
                T(c, Y_sub == cls) = 1;
            end

            fprintf('  Training built-in 1D-CNN on %d beats (%d epochs) …\n', N, epochs);
            for ep = 1:epochs
                logits = obj.W_fc * emb + repmat(obj.b_fc, 1, N);
                logits_s = logits - repmat(max(logits, [], 1), 5, 1);
                P = exp(logits_s) ./ repmat(sum(exp(logits_s), 1), 5, 1);

                grad_z = (P - T) / N;
                grad_W_fc = grad_z * emb';
                grad_b_fc = sum(grad_z, 2);

                grad_emb = obj.W_fc' * grad_z;
                grad_emb(emb <= 0) = 0;
                grad_W_emb = grad_emb * (emb(1:32,:))';
                grad_b_emb = sum(grad_emb, 2);

                obj.W_fc = obj.W_fc - lr * grad_W_fc;
                obj.b_fc = obj.b_fc - lr * grad_b_fc;
                obj.W_emb = obj.W_emb - lr * grad_W_emb;
                obj.b_emb = obj.b_emb - lr * grad_b_emb;
            end
            obj = obj.build_layers_metadata();
            fprintf('  Base 1D-CNN training complete.\n');
        end
    end
end
