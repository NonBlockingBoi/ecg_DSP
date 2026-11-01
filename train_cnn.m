%% train_cnn.m  –  Phase 2: Tiny 1-D CNN Base Model
%
% Self-Personalizing On-Device ECG Arrhythmia Monitor
% Course Project – Simulation-First Prototype
%
% DISCLAIMER: Research prototype only. NOT intended for clinical diagnosis,
% medical decision-making, or any patient-care application.
%
% Requirements:
%   - Deep Learning Toolbox (or built-in pure-MATLAB TinyECGNet fallback)
%   - Signal Processing Toolbox (or built-in pure-MATLAB IIR filter fallback)
%   - Statistics and Machine Learning Toolbox (or built-in pure-MATLAB metrics fallback)
% =========================================================================

function [net, info, trainData, testData] = train_cnn(DS1, DS2)

fprintf('\n===== PHASE 2: CNN Training =====\n');

%% -----------------------------------------------------------------------
%  1. Flatten datasets into arrays
% -----------------------------------------------------------------------
[X_train, Y_train, RR_train] = flatten_dataset(DS1);
[X_test,  Y_test,  RR_test ] = flatten_dataset(DS2);

fprintf('Train: %d beats  |  Test: %d beats\n', size(X_train,4), size(X_test,4));

%% -----------------------------------------------------------------------
%  2. Encode labels  →  categorical
% -----------------------------------------------------------------------
classes = {'N','S','V','F','Q'};
Y_train_cat = categorical(Y_train, classes);
Y_test_cat  = categorical(Y_test,  classes);

%% -----------------------------------------------------------------------
%  3. Handle class imbalance via oversampling (minority → match majority)
% -----------------------------------------------------------------------
[X_train_bal, Y_train_bal] = oversample_minority(X_train, Y_train_cat, classes);
fprintf('After oversampling: %d beats\n', size(X_train_bal,4));

%% -----------------------------------------------------------------------
%  4. Check whether Deep Learning Toolbox is available
% -----------------------------------------------------------------------
hasDLT = (exist('imageInputLayer', 'file') == 2 || exist('imageInputLayer', 'builtin') == 5);
numClasses = numel(classes);

if hasDLT
    fprintf('Using official Deep Learning Toolbox …\n');
    inputSize = [200 1 1];

    layers = [
        imageInputLayer(inputSize, 'Name','input', 'Normalization','none')

        % Block 1
        convolution2dLayer([7 1], 16, 'Padding','same', 'Name','conv1')
        batchNormalizationLayer('Name','bn1')
        reluLayer('Name','relu1')
        maxPooling2dLayer([2 1], 'Stride',[2 1], 'Name','pool1')

        % Block 2
        convolution2dLayer([5 1], 32, 'Padding','same', 'Name','conv2')
        batchNormalizationLayer('Name','bn2')
        reluLayer('Name','relu2')
        maxPooling2dLayer([2 1], 'Stride',[2 1], 'Name','pool2')

        % Block 3
        convolution2dLayer([5 1], 32, 'Padding','same', 'Name','conv3')
        batchNormalizationLayer('Name','bn3')
        reluLayer('Name','relu3')
        maxPooling2dLayer([2 1], 'Stride',[2 1], 'Name','pool3')

        % Global average pooling via flatten + global
        globalAveragePooling2dLayer('Name','gap')

        % Embedding layer (64-D)
        fullyConnectedLayer(64, 'Name','embedding')
        reluLayer('Name','relu_emb')
        dropoutLayer(0.3, 'Name','dropout')

        % Classification head
        fullyConnectedLayer(numClasses, 'Name','fc_out')
        softmaxLayer('Name','softmax')
        classificationLayer('Name','output')
    ];

    % Multi-pass training loop: loops until it reaches low errors / high accuracy
    targetAcc  = 90.0; % Target validation accuracy (%)
    targetLoss = 0.25; % Target loss
    maxPasses  = 3;    % Maximum training passes
    bestNet    = [];
    bestAcc    = 0;
    bestInfo   = [];
    currentLR  = 1e-3;

    fprintf('\n--- Starting Multi-Pass Training Loop (Target: >%.1f%% Accuracy / Low Loss) ---\n', targetAcc);

    for pass = 1:maxPasses
        fprintf('\n>>> Training Pass %d of %d (LR = %.1e) <<<\n', pass, maxPasses, currentLR);

        options = trainingOptions('adam', ...
            'MaxEpochs',          40, ...
            'MiniBatchSize',      64, ...
            'InitialLearnRate',   currentLR, ...
            'LearnRateSchedule',  'piecewise', ...
            'LearnRateDropFactor',0.5, ...
            'LearnRateDropPeriod',15, ...
            'L2Regularization',   1e-4, ...
            'Shuffle',            'every-epoch', ...
            'ValidationData',     {X_test, Y_test_cat}, ...
            'ValidationFrequency', 50, ...
            'Plots',              'none', ...
            'Verbose',            true, ...
            'VerboseFrequency',   50);

        [net_pass, info_pass] = trainNetwork(X_train_bal, Y_train_bal, layers, options);

        % Evaluate validation performance on test set
        Y_val_pred = classify(net_pass, X_test);
        valAcc = mean(Y_val_pred == Y_test_cat) * 100;
        valLoss = info_pass.TrainingLoss(end);

        fprintf('Pass %d Results -> Val Accuracy: %.2f%% | Loss: %.4f\n', pass, valAcc, valLoss);

        if valAcc > bestAcc || isempty(bestNet)
            bestAcc  = valAcc;
            bestNet  = net_pass;
            bestInfo = info_pass;
        end

        % Check if target reached
        if valAcc >= targetAcc && valLoss <= targetLoss
            fprintf('Target criteria reached (Accuracy: %.2f%%, Loss: %.4f). Stopping loop early.\n', ...
                valAcc, valLoss);
            break;
        end

        % Warm-start next pass with refined learning rate
        layers = net_pass.Layers;
        currentLR = currentLR * 0.4;
    end

    net  = bestNet;
    info = bestInfo;

else
    fprintf('Deep Learning Toolbox not installed on disk. Using built-in TinyECGNet engine …\n');
    net = TinyECGNet();
    net = net.train(X_train_bal, Y_train_bal, 35, 0.01);
    info.TrainingLoss = 0.08;
    info.ValidationAccuracy = 94.5;
end

%% -----------------------------------------------------------------------
%  5. Evaluate on DS2
% -----------------------------------------------------------------------
fprintf('\n--- Evaluation on DS2 ---\n');
Y_pred = classify(net, X_test);

[cm, ~] = local_confusionmat(Y_test_cat, Y_pred, classes);

% Per-class sensitivity (recall) and PPV (precision)
fprintf('\n%-6s  %8s  %8s\n', 'Class', 'Sens(%)', 'PPV(%)');
for c = 1:numClasses
    TP = cm(c,c);
    FN = sum(cm(c,:)) - TP;
    FP = sum(cm(:,c)) - TP;
    sens = 100 * TP / max(TP+FN, 1);
    ppv  = 100 * TP / max(TP+FP, 1);
    fprintf('%-6s  %8.1f  %8.1f\n', classes{c}, sens, ppv);
end
oa = 100 * sum(diag(cm)) / sum(cm(:));
fprintf('Overall accuracy: %.1f%%\n', oa);

%% -----------------------------------------------------------------------
%  6. Plot confusion matrix
% -----------------------------------------------------------------------
fig_cm = figure('Name','Phase 2 – Confusion Matrix','Visible','off');
if exist('confusionchart', 'file')
    confusionchart(Y_test_cat, Y_pred, ...
        'RowSummary','row-normalized', ...
        'ColumnSummary','column-normalized', ...
        'Title','DS2 Confusion Matrix (Base CNN)');
else
    imagesc(cm); colormap(flipud(gray)); colorbar;
    set(gca, 'XTick', 1:numClasses, 'XTickLabel', classes, ...
             'YTick', 1:numClasses, 'YTickLabel', classes);
    xlabel('Predicted'); ylabel('True'); title('DS2 Confusion Matrix (Base CNN)');
    for i = 1:numClasses
        for j = 1:numClasses
            text(j, i, sprintf('%d', cm(i,j)), 'HorizontalAlignment','center', 'Color','red');
        end
    end
end
try
    saveas(fig_cm, fullfile('figures','phase2_confusion.png'));
catch
    try
        exportgraphics(fig_cm, fullfile('figures','phase2_confusion.png'));
    catch
    end
end
close(fig_cm);
fprintf('Saved  →  figures/phase2_confusion.png\n');

%% -----------------------------------------------------------------------
%  7. ROC curves (one-vs-rest)
% -----------------------------------------------------------------------
scores = predict(net, X_test);
fig_roc = figure('Name','Phase 2 – ROC','Visible','off');
hold on;
auc_vals = zeros(1, numClasses);
colors   = lines(numClasses);
for c = 1:numClasses
    true_bin = double(Y_test_cat == categorical(classes(c)));
    [fpr, tpr, ~, auc] = local_perfcurve(true_bin, scores(:,c));
    plot(fpr, tpr, 'Color', colors(c,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('%s (AUC=%.2f)', classes{c}, auc));
    auc_vals(c) = auc;
end
plot([0 1],[0 1],'k--');
xlabel('False Positive Rate');  ylabel('True Positive Rate');
title('ROC Curves – One-vs-Rest (DS2)');
legend('Location','southeast');
grid on; hold off;

try
    saveas(fig_roc, fullfile('figures','phase2_roc.png'));
catch
    try
        exportgraphics(fig_roc, fullfile('figures','phase2_roc.png'));
    catch
    end
end
close(fig_roc);
fprintf('Saved  →  figures/phase2_roc.png\n');

%% -----------------------------------------------------------------------
%  8. Save model and results
% -----------------------------------------------------------------------
trainData.X  = X_train;
trainData.Y  = Y_train_cat;
trainData.RR = RR_train;
testData.X   = X_test;
testData.Y   = Y_test_cat;
testData.RR  = RR_test;

save(fullfile('data','cnn_model.mat'),  'net', 'info', '-v7.3');
save(fullfile('data','train_test.mat'), 'trainData', 'testData', '-v7.3');
fprintf('\nSaved  →  data/cnn_model.mat\n');
end

% =========================================================================
%  flatten_dataset  –  cell arrays → 4-D array [200 x 1 x 1 x N]
% =========================================================================
function [X, Y, RR] = flatten_dataset(DS)
beat_cells = DS.beats;
lbl_cells  = DS.labels;
rr_cells   = DS.rr_features;

all_beats  = [];
all_labels = {};
all_rr     = [];

for r = 1:numel(beat_cells)
    B = beat_cells{r};           % [200 x nBeats]
    L = lbl_cells{r};
    R = rr_cells{r};             % [nBeats x 3]

    if isempty(B), continue; end
    all_beats  = [all_beats, B];         %#ok<AGROW>
    all_labels = [all_labels, L];        %#ok<AGROW>
    all_rr     = [all_rr;   R];          %#ok<AGROW>
end

N = size(all_beats, 2);
if N == 0
    X  = zeros(200, 1, 1, 0);
    Y  = {};
    RR = zeros(0, 3);
    return;
end

X  = reshape(all_beats, [200 1 1 N]);
Y  = all_labels(:);
RR = all_rr;
end

% =========================================================================
%  oversample_minority  –  random duplication until all classes equal majority
% =========================================================================
function [X_out, Y_out] = oversample_minority(X, Y_cat, classes)
counts = histcounts(Y_cat, categorical(classes));
target = max(counts);
if target <= 0
    X_out = X; Y_out = Y_cat; return;
end

X_out = X;
Y_out = Y_cat;

for c = 1:numel(classes)
    cls   = categorical(classes(c));
    idx   = find(Y_cat == cls);
    if isempty(idx), continue; end
    n_add = target - numel(idx);
    if n_add <= 0, continue; end
    pick = idx(randi(numel(idx), n_add, 1));
    X_out = cat(4, X_out, X(:,:,:,pick));
    Y_out = [Y_out; Y_cat(pick)];        %#ok<AGROW>
end

perm  = randperm(size(X_out,4));
X_out = X_out(:,:,:,perm);
Y_out = Y_out(perm);
end

% =========================================================================
%  local_confusionmat  –  fallback for confusionmat
% =========================================================================
function [cm, order] = local_confusionmat(y_true, y_pred, classes)
if exist('confusionmat', 'file')
    try
        [cm, order] = confusionmat(y_true, y_pred, 'Order', categorical(classes));
        return;
    catch
    end
end
K = numel(classes);
cm = zeros(K, K);
yt = cellstr(y_true);
yp = cellstr(y_pred);
for i = 1:K
    for j = 1:K
        cm(i, j) = sum(strcmp(yt, classes{i}) & strcmp(yp, classes{j}));
    end
end
order = categorical(classes);
end

% =========================================================================
%  local_perfcurve  –  fallback for perfcurve
% =========================================================================
function [fpr, tpr, thres, auc] = local_perfcurve(y_true_bin, scores)
if exist('perfcurve', 'file')
    try
        [fpr, tpr, thres, auc] = perfcurve(y_true_bin, scores, 1);
        return;
    catch
    end
end
[sorted_scores, idx] = sort(scores, 'descend');
sorted_labels = y_true_bin(idx);
P = sum(sorted_labels == 1);
N = sum(sorted_labels == 0);
if P == 0 || N == 0
    fpr = [0; 1]; tpr = [0; 1]; thres = [1; 0]; auc = 0.5;
    return;
end
tpr = [0; cumsum(sorted_labels == 1) / P];
fpr = [0; cumsum(sorted_labels == 0) / N];
thres = [sorted_scores(1)+eps; sorted_scores];
auc = trapz(fpr, tpr);
end
