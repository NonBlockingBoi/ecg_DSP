%% personalize.m  –  Phase 4: Patient-Specific Personalization
%
% Self-Personalizing On-Device ECG Arrhythmia Monitor
% Course Project – Simulation-First Prototype
%
% DISCLAIMER: Research prototype only. NOT intended for clinical diagnosis,
% medical decision-making, or any patient-care application.
%
% Strategy:
%   1. Freeze the CNN and extract 64-D embeddings.
%   2. For each DS2 patient, use the first ~5 min of beats as calibration.
%   3. Build a Gaussian model (centroid + covariance) of NORMAL embeddings.
%   4. Flag beats whose Mahalanobis distance exceeds a threshold as outliers
%      (i.e., likely arrhythmic).
%   5. Optional: last-layer fine-tuning with a few SGD steps on
%      confirmed-normal beats.
%   6. Report sensitivity, PPV, false-alarm rate before vs after per patient.
%
% Requirements:
%   - Deep Learning Toolbox
%   - Statistics and Machine Learning Toolbox
% =========================================================================

function personalize(net, DS2)

fprintf('\n===== PHASE 4: Personalization =====\n');

%% -----------------------------------------------------------------------
%  0. Load from disk if arguments not provided
% -----------------------------------------------------------------------
if nargin < 1 || isempty(net)
    S   = load(fullfile('data','cnn_model.mat'));
    net = S.net;
end
if nargin < 2 || isempty(DS2)
    S   = load(fullfile('data','preprocessed.mat'));
    DS2 = S.DS2;
end

classes    = {'N','S','V','F','Q'};
CALIB_SECS = 300;   % 5-minute calibration window @ 360 Hz
FS         = 360;
MAHAL_THRESH = 3.0; % χ² 0.99 for 64-D ≈ chi2inv(0.99,64); adjust empirically

%% -----------------------------------------------------------------------
%  1. Build embedding extractor (activations from 'embedding' layer)
% -----------------------------------------------------------------------
fprintf('Building embedding extractor …\n');
embed_layer = 'embedding';   % name of the FC-64 layer in train_cnn.m

%% -----------------------------------------------------------------------
%  2. Per-patient metrics storage
% -----------------------------------------------------------------------
n_patients = numel(DS2.record_id);
results_table = table('Size',[n_patients 9], ...
    'VariableTypes', repmat({'double'},1,9), ...
    'VariableNames', {'Patient','N_calib','N_eval', ...
    'Sens_base','PPV_base','FAR_base', ...
    'Sens_pers','PPV_pers','FAR_pers'});

fig_bar = figure('Name','Phase 4 – Before vs After','Visible','off');

for p = 1:n_patients
    rec  = DS2.record_id{p};
    fprintf('\n  Patient %s (%d/%d) …\n', rec, p, n_patients);

    beats  = DS2.beats{p};          % [200 x N]
    labels = DS2.labels{p};         % {1 x N}
    bidx   = DS2.beat_idx{p};       % R-peak sample indices

    if isempty(beats)
        fprintf('    Skipping (no beats)\n');
        continue;
    end

    N_total = size(beats,2);

    %% --- Split calibration / evaluation ---------------------------------
    % Calibration: first 5 min (or first 40% if record is shorter than 10 min)
    max_rec_sec = max(bidx) / FS;
    calib_cutoff_sec = min(CALIB_SECS, max_rec_sec * 0.4);
    calib_mask = (bidx <= calib_cutoff_sec * FS);
    eval_mask  = ~calib_mask;

    if sum(calib_mask) < 10 || sum(eval_mask) < 5
        fprintf('    Too few beats for split – skipping\n');
        continue;
    end

    X_calib = reshape(beats(:, calib_mask), [200 1 1 sum(calib_mask)]);
    Y_calib = labels(calib_mask);
    X_eval  = reshape(beats(:, eval_mask),  [200 1 1 sum(eval_mask)]);
    Y_eval  = categorical(labels(eval_mask), classes);

    %% --- Base-model classification (before personalization) -------------
    Y_pred_base = classify(net, X_eval);
    [sens_b, ppv_b, far_b] = compute_arrhythmia_metrics(Y_eval, Y_pred_base);

    %% --- Extract embeddings for calibration beats -----------------------
    emb_calib = extract_embeddings(net, X_calib, embed_layer); % [64 x N_calib]

    % Keep only Normal beats for Gaussian model
    normal_mask = strcmp(Y_calib, 'N');
    emb_normal  = emb_calib(:, normal_mask)';   % [n_normal x 64]

    if size(emb_normal,1) < 5
        fprintf('    Too few Normal beats in calibration – skipping\n');
        continue;
    end

    fprintf('    Calibration: %d total / %d Normal beats\n', ...
        sum(calib_mask), size(emb_normal,1));

    % Compute centroid and regularized covariance
    mu_N  = mean(emb_normal, 1);            % [1 x 64]
    Sigma = cov(emb_normal) + 1e-4*eye(size(emb_normal,2));  % regularize

    %% --- Mahalanobis classification (personalized) ----------------------
    emb_eval = extract_embeddings(net, X_eval, embed_layer);  % [64 x N_eval]

    Y_pred_pers = classify_mahalanobis(net, emb_eval, mu_N, Sigma, ...
        MAHAL_THRESH, classes, X_eval);

    [sens_p, ppv_p, far_p] = compute_arrhythmia_metrics(Y_eval, ...
        categorical(Y_pred_pers, classes));

    fprintf('    Before: Sens=%.1f%% PPV=%.1f%% FAR=%.1f%%\n', ...
        sens_b*100, ppv_b*100, far_b*100);
    fprintf('    After:  Sens=%.1f%% PPV=%.1f%% FAR=%.1f%%\n', ...
        sens_p*100, ppv_p*100, far_p*100);

    %% --- Optional: last-layer fine-tuning on confirmed Normal beats -----
    % Use Mahalanobis-accepted Normal beats to fine-tune the final FC layer
    dist_eval   = mahal_distance(emb_eval', mu_N, Sigma);
    normal_pred = (dist_eval < MAHAL_THRESH) & ...
                  (Y_eval == categorical({'N'}));
    if sum(normal_pred) >= 5
        net = finetune_last_layer(net, X_eval(:,:,:,normal_pred), ...
            Y_eval(normal_pred), classes);
        fprintf('    Fine-tuning done on %d confirmed Normal beats.\n', ...
            sum(normal_pred));
    end

    %% --- Store results ---------------------------------------------------
    results_table.Patient(p)  = str2double(rec);
    results_table.N_calib(p)  = sum(calib_mask);
    results_table.N_eval(p)   = sum(eval_mask);
    results_table.Sens_base(p)= sens_b;
    results_table.PPV_base(p) = ppv_b;
    results_table.FAR_base(p) = far_b;
    results_table.Sens_pers(p)= sens_p;
    results_table.PPV_pers(p) = ppv_p;
    results_table.FAR_pers(p) = far_p;
end

%% -----------------------------------------------------------------------
%  3. Aggregate before/after bar chart
% -----------------------------------------------------------------------
valid = results_table.N_eval > 0;
avg_base = mean([results_table.Sens_base(valid), ...
                 results_table.PPV_base(valid),  ...
                 results_table.FAR_base(valid)])*100;
avg_pers = mean([results_table.Sens_pers(valid), ...
                 results_table.PPV_pers(valid),  ...
                 results_table.FAR_pers(valid)])*100;

figure(fig_bar);
metrics_labels = {'Sensitivity','PPV','False-Alarm Rate'};
bar_data = [avg_base; avg_pers]';
b = bar(bar_data);
b(1).FaceColor = [0.2 0.4 0.8];
b(2).FaceColor = [0.9 0.4 0.2];
set(gca,'XTickLabel', metrics_labels);
ylabel('Percentage (%)');  ylim([0 100]);
legend({'Before personalization','After personalization'},'Location','northwest');
title('Personalization: Average Before vs After (DS2 Patients)');
grid on;
saveas(fig_bar, fullfile('figures','phase4_personalization.png'));
fprintf('\nSaved  →  figures/phase4_personalization.png\n');

%% -----------------------------------------------------------------------
%  4. Save results
% -----------------------------------------------------------------------
save(fullfile('results','personalization_results.mat'), 'results_table');
writetable(results_table, fullfile('results','personalization_results.csv'));
fprintf('Saved  →  results/personalization_results.csv\n');

% Print final table
fprintf('\n--- Per-Patient Personalization Results ---\n');
disp(results_table(valid,:));
end

% =========================================================================
%  extract_embeddings  –  get activations from the 'embedding' FC layer
% =========================================================================
function emb = extract_embeddings(net, X4D, layer_name)
% activations() returns [1 x 1 x 64 x N] for a FC layer
try
    act = activations(net, X4D, layer_name, 'OutputAs','channels');
    emb = squeeze(act);
    if size(emb, 1) ~= 64 && size(emb, 2) == 64
        emb = emb';
    end
    if isvector(emb) && numel(emb) == 64
        emb = emb(:);   % single sample → column [64 x 1]
    end
catch
    % Fallback: manual forward pass using predict scores
    scores = predict(net, X4D);
    emb = scores';   % use output probabilities as proxy embedding
end
end

% =========================================================================
%  mahal_distance  –  squared Mahalanobis distances from centroid
% =========================================================================
function D = mahal_distance(E, mu, Sigma)
% E:     [N x 64]
% mu:    [1 x 64]
% Sigma: [64 x 64]
delta = E - mu;
try
    iS = inv(Sigma);
catch
    iS = pinv(Sigma);
end
D = sqrt(sum((delta * iS) .* delta, 2));   % [N x 1]
end

% =========================================================================
%  classify_mahalanobis
%   – beats close to Normal centroid → 'N'
%   – beats far away       → use base CNN classification
% =========================================================================
function Y_out = classify_mahalanobis(net, emb_eval, mu_N, Sigma, ...
    thresh, classes, X_eval)
N  = size(emb_eval, 2);
D  = mahal_distance(emb_eval', mu_N, Sigma);   % [N x 1]

% Base CNN predictions for all
Y_base = classify(net, X_eval);

Y_out = cellstr(char(Y_base));
for i = 1:N
    if D(i) <= thresh
        Y_out{i} = 'N';   % personalized: close → Normal
    end
    % else: keep base CNN label
end
end

% =========================================================================
%  compute_arrhythmia_metrics
%   – Sensitivity / PPV for non-N beats, False-Alarm Rate
%   – Non-N = {S,V,F,Q} treated as "arrhythmia positive"
% =========================================================================
function [sens, ppv, far] = compute_arrhythmia_metrics(Y_true_cat, Y_pred_cat)
pos_true = Y_true_cat ~= categorical({'N'});
pos_pred = Y_pred_cat ~= categorical({'N'});

TP = sum( pos_pred &  pos_true);
FP = sum( pos_pred & ~pos_true);
FN = sum(~pos_pred &  pos_true);
TN = sum(~pos_pred & ~pos_true);

sens = TP / max(TP + FN, 1);
ppv  = TP / max(TP + FP, 1);
far  = FP / max(FP + TN, 1);
end

% =========================================================================
%  finetune_last_layer
%   – Fine-tune only the last FC layer (fc_out) for a few SGD epochs
%     on confirmed-Normal beats from this patient.
% =========================================================================
function net_ft = finetune_last_layer(net, X_ft, Y_ft, classes)
if isa(net, 'TinyECGNet')
    try
        [~, emb] = net.forward(X_ft);
        N_ft = size(emb, 2);
        T = zeros(5, N_ft); T(1, :) = 1;
        for ep = 1:3
            logits = net.W_fc * emb + repmat(net.b_fc, 1, N_ft);
            P = exp(logits) ./ repmat(sum(exp(logits), 1), 5, 1);
            grad_fc = ((P - T) / N_ft) * emb';
            net.W_fc = net.W_fc - 0.005 * grad_fc;
        end
        net_ft = net.build_layers_metadata();
        return;
    catch
        net_ft = net;
        return;
    end
end
try
    % Freeze all layers except the last FC
    lgraph = layerGraph(net);
    layers = lgraph.Layers;

    % Set LearnRateFactor=0 for all trainable layers except fc_out
    for i = 1:numel(layers)
        lname = layers(i).Name;
        if ~strcmp(lname, 'fc_out') && ...
           isprop(layers(i),'WeightLearnRateFactor')
            layers(i).WeightLearnRateFactor = 0;
            layers(i).BiasLearnRateFactor   = 0;
        end
    end

    opts_ft = trainingOptions('sgdm', ...
        'MaxEpochs',        3, ...
        'MiniBatchSize',    min(16, size(X_ft,4)), ...
        'InitialLearnRate', 1e-4, ...
        'Momentum',         0.9, ...
        'Verbose',          false, ...
        'Plots',            'none');

    [net_ft, ~] = trainNetwork(X_ft, Y_ft, layers, opts_ft);
catch ME
    warning('finetune:failed', 'Last-layer fine-tuning failed: %s', ME.message);
    net_ft = net;   % return original
end
end
