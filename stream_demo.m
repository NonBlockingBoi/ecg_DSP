%% stream_demo.m  –  Phase 5: Real-Time Streaming Simulation
%
% Self-Personalizing On-Device ECG Arrhythmia Monitor
% Course Project – Simulation-First Prototype
%
% DISCLAIMER: Research prototype only. NOT intended for clinical diagnosis,
% medical decision-making, or any patient-care application.
%
% Replays a chosen record, runs Pan-Tompkins detection and CNN classification
% in a streaming loop, then outputs the ECG with colour-coded beat labels,
% alarms, and per-beat latency statistics.
% =========================================================================

function stream_demo(net, DS2, record_choice)

fprintf('\n===== PHASE 5: Streaming Simulation =====\n');

%% -----------------------------------------------------------------------
%  0. Load artefacts if not provided
% -----------------------------------------------------------------------
if nargin < 1 || isempty(net)
    S   = load(fullfile('data','cnn_model.mat'));
    net = S.net;
end
if nargin < 2 || isempty(DS2)
    S   = load(fullfile('data','preprocessed.mat'));
    DS2 = S.DS2;
end
if nargin < 3 || isempty(record_choice)
    record_choice = 1;   % first DS2 record
end

%% -----------------------------------------------------------------------
%  1. Select a patient record for demo
% -----------------------------------------------------------------------
if isfield(DS2, 'record_id') && ~isempty(DS2.record_id)
    p = min(record_choice, numel(DS2.record_id));
    rec = DS2.record_id{p};
    beats = DS2.beats{p};
    labels = DS2.labels{p};
    beat_idx = DS2.beat_idx{p};
else
    rec = 'demo';
    beats = [];
    labels = {};
    beat_idx = [];
end

if isempty(beats)
    fprintf('No cached beats found – generating synthetic ECG for demo.\n');
    [beats, labels, beat_idx] = generate_demo_signal();
    rec = 'synthetic_DS2';
end

fprintf('Streaming record: %s  (%d beats)\n', rec, size(beats,2));

%% -----------------------------------------------------------------------
%  2. Reconstruct continuous ECG trace for display
% -----------------------------------------------------------------------
WIN_PRE  = 90;
WIN_POST = 110;
FS       = 360;

total_len = max(beat_idx) + WIN_POST + 200;
ecg_full  = zeros(total_len, 1);
count_map = zeros(total_len, 1);
for b = 1:size(beats,2)
    i0 = beat_idx(b) - WIN_PRE;
    i1 = beat_idx(b) + WIN_POST - 1;
    if i0 < 1 || i1 > total_len, continue; end
    ecg_full(i0:i1) = ecg_full(i0:i1) + beats(:,b);
    count_map(i0:i1)= count_map(i0:i1) + 1;
end
mask = count_map > 0;
ecg_full(mask) = ecg_full(mask) ./ count_map(mask);

%% -----------------------------------------------------------------------
%  3. Colour map for AAMI classes
% -----------------------------------------------------------------------
% N=blue, S=green, V=red, F=orange, Q=purple
class_colors = containers.Map( ...
    {'N','S','V','F','Q'}, ...
    {[0.2 0.4 0.8], [0.1 0.7 0.2], [0.9 0.1 0.1], [0.95 0.5 0.1], [0.6 0.1 0.8]});

%% -----------------------------------------------------------------------
%  4. Streaming loop (beat-by-beat simulation)
% -----------------------------------------------------------------------
N_beats     = size(beats,2);
latencies   = zeros(N_beats, 1);
pred_labels = cell(N_beats, 1);

fprintf('Running real-time streaming inference loop …\n');
for b = 1:N_beats
    beat_seg = beats(:, b);
    X_in     = reshape(beat_seg, [200 1 1 1]);

    t0 = tic;
    Y_pred = classify(net, X_in);
    lat    = toc(t0) * 1000; % ms

    latencies(b)   = lat;
    pred_labels{b} = char(Y_pred);
end

%% -----------------------------------------------------------------------
%  5. Latency statistics
% -----------------------------------------------------------------------
valid_lat = latencies(latencies > 0);
fprintf('\n--- Streaming Latency Statistics ---\n');
fprintf('  Mean:   %.2f ms\n', mean(valid_lat));
fprintf('  Median: %.2f ms\n', median(valid_lat));
fprintf('  Max:    %.2f ms\n', max(valid_lat));
fprintf('  P95:    %.2f ms\n', prctile(valid_lat, 95));

%% -----------------------------------------------------------------------
%  6. Accuracy of streaming predictions vs. ground truth
% -----------------------------------------------------------------------
if ~isempty(labels)
    Y_true_s   = categorical(labels(:), {'N','S','V','F','Q'});
    Y_pred_s   = categorical(pred_labels(:), {'N','S','V','F','Q'});
    acc_stream = mean(Y_true_s == Y_pred_s) * 100;
    fprintf('  Streaming accuracy across all %d beats: %.1f%%\n', N_beats, acc_stream);
end

%% -----------------------------------------------------------------------
%  7. Save figures
% -----------------------------------------------------------------------
% 1) Latency histogram
fig_lat = figure('Name','Phase 5 – Latency','Visible','off');
histogram(valid_lat, 20, 'FaceColor',[0.2 0.6 0.9], 'EdgeColor','none');
hold on;
xline(mean(valid_lat),  'r-',  sprintf('Mean %.1f ms', mean(valid_lat)),  'LineWidth',2);
xline(median(valid_lat),'g--', sprintf('Med %.1f ms',  median(valid_lat)),'LineWidth',2);
xlabel('Latency (ms)'); ylabel('Count');
title('Per-Beat Inference Latency Distribution');
grid on; hold off;
saveas(fig_lat, fullfile('figures','phase5_latency.png'));
fprintf('Saved  →  figures/phase5_latency.png\n');

% 2) Annotated streaming ECG snapshot
fig_ecg = figure('Name','Phase 5 – ECG Snapshot','Visible','off', 'Position', [100 100 1200 450]);
t_full  = (0:numel(ecg_full)-1) / FS;
show_secs = min(30, t_full(end));
mask_t  = t_full <= show_secs;
plot(t_full(mask_t), ecg_full(mask_t), 'Color', [0.2 0.2 0.2], 'LineWidth', 1.0);
hold on;

cls_list = {'N','S','V','F','Q'};
for b = 1:N_beats
    tb = beat_idx(b)/FS;
    if tb > show_secs, continue; end
    if isempty(pred_labels{b}), continue; end
    col = class_colors(pred_labels{b});
    y_val = ecg_full(min(beat_idx(b), numel(ecg_full)));
    plot(tb, y_val + 0.15, 'v', 'MarkerFaceColor', col, 'MarkerEdgeColor', 'k', 'MarkerSize', 8);
    % If arrhythmia, add alarm marker
    if ~strcmp(pred_labels{b}, 'N')
        text(tb, y_val + 0.35, sprintf('⚠ %s', pred_labels{b}), ...
            'Color', [0.85 0.1 0.1], 'FontSize', 8, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    end
end

xlabel('Time (seconds)'); ylabel('Normalized Amplitude');
title(sprintf('Simulated Real-Time ECG Arrhythmia Monitor – Record %s', rec), 'FontSize', 12);
grid on;
ylim([min(ecg_full(mask_t))-0.5, max(ecg_full(mask_t))+0.8]);

% Add legend
legend_h = [];
legend_t = {'ECG Trace'};
legend_h(1) = plot(NaN, NaN, 'Color', [0.2 0.2 0.2], 'LineWidth', 1);
for c = 1:numel(cls_list)
    legend_h(end+1) = plot(NaN, NaN, 'v', 'MarkerFaceColor', class_colors(cls_list{c}), ...
        'MarkerEdgeColor', 'k', 'MarkerSize', 8); %#ok<AGROW>
    legend_t{end+1} = sprintf('Class %s', cls_list{c}); %#ok<AGROW>
end
legend(legend_h, legend_t, 'Location', 'northeast');
hold off;
saveas(fig_ecg, fullfile('figures','phase5_ecg_snapshot.png'));
fprintf('Saved  →  figures/phase5_ecg_snapshot.png\n');

fprintf('Streaming simulation completed successfully.\n');
end

% =========================================================================
%  generate_demo_signal
% =========================================================================
function [beats, labels, beat_idx] = generate_demo_signal()
FS      = 360;
N_beats = 60;
WIN_PRE = 90; WIN_POST = 110;
WIN_LEN = WIN_PRE + WIN_POST;

beats    = zeros(WIN_LEN, N_beats);
labels   = cell(1, N_beats);
beat_idx = zeros(1, N_beats);
label_pool = {'N','N','N','N','N','V','S','N','N','F'};

t = (0:WIN_LEN-1)' / FS;
for b = 1:N_beats
    lbl = label_pool{mod(b-1,numel(label_pool))+1};
    beat = 0.3*exp(-((t-0.25).^2)/(2*0.04^2)) + ...
           1.0*exp(-((t-0.25).^2)/(2*0.01^2)) + ...
           0.3*exp(-((t-0.50).^2)/(2*0.06^2));
    if strcmp(lbl,'V')
        beat = -0.7*exp(-((t-0.25).^2)/(2*0.05^2));
    end
    beat = beat + 0.02*randn(size(beat));
    beats(:,b)   = (beat - mean(beat)) / (std(beat) + 1e-6);
    labels{b}    = lbl;
    beat_idx(b)  = WIN_PRE + (b-1)*round(0.8*FS);
end
end

