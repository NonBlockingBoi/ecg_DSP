%% main.m  –  Master Driver Script
%
% Self-Personalizing On-Device ECG Arrhythmia Monitor
% Course Project – Simulation-First Prototype
%
% DISCLAIMER: Research prototype only. NOT a medical or diagnostic device.
%             Do NOT use for clinical decision-making or patient care.
%
% ─────────────────────────────────────────────────────────────────────────
% REQUIRED TOOLBOXES
%   1. Signal Processing Toolbox            – filtering, resampling
%   2. Deep Learning Toolbox                – CNN training, activations
%   3. Statistics and Machine Learning Toolbox – confusionmat, mahal
%   4. Deep Learning Toolbox Model Quantization Library (add-on)  – dlquantizer
%      (optional; Phase 3 falls back gracefully if absent)
%   5. MATLAB Coder / Embedded Coder        – C-code generation stub
%      (optional; Phase 3 prints instructions if absent)
%   6. WFDB Toolbox for MATLAB              – live PhysioNet download
%      https://physionet.org/content/wfdb-matlab/
%      (optional; Phase 1 falls back to .mat files in ./data/ or synthetic)
%
% SETUP
%   a) Download MIT-BIH records:
%      >> physionetdb('mitdb');      % requires WFDB Toolbox
%   OR place pre-exported .mat files in ./data/<recID>.mat with fields:
%      val         – double signal  [N_samples x N_leads]
%      Fs          – sampling rate  (Hz)
%      ann_sample  – annotation sample indices  [N_ann x 1]
%      ann_type    – annotation beat types      {N_ann x 1}  e.g. {'N','V',…}
%
% RUNNING
%   >> main           % full pipeline with synthetic demo data (no WFDB)
%   >> main(true)     % full pipeline using WFDB Toolbox (live download)
%   >> main(false, 3) % no WFDB, stream record #3 from DS2
% ─────────────────────────────────────────────────────────────────────────

function main(useWFDB, stream_record_idx)

if nargin < 1, useWFDB           = false; end
if nargin < 2, stream_record_idx = 1;     end

banner();

%% ──────────────────────────────────────────────────────────────────────
%  Environment setup
% ──────────────────────────────────────────────────────────────────────
addpath(pwd);                   % ensure all .m files are on the path
check_toolboxes();

% Create output directories
for d = {'data','figures','results'}
    if ~exist(d{1},'dir'), mkdir(d{1}); end
end

%% ──────────────────────────────────────────────────────────────────────
%  PHASE 1 – Preprocessing
% ──────────────────────────────────────────────────────────────────────
t1 = tic;
if exist(fullfile('data','preprocessed.mat'),'file')
    fprintf('\n[PHASE 1] Loading cached preprocessed data …\n');
    load(fullfile('data','preprocessed.mat'), 'DS1', 'DS2');
else
    fprintf('\n[PHASE 1] Running preprocessing …\n');
    [DS1, DS2] = preprocess(useWFDB);
end
fprintf('[PHASE 1] Done in %.1f s\n', toc(t1));

%% ──────────────────────────────────────────────────────────────────────
%  PHASE 2 – CNN Training
% ──────────────────────────────────────────────────────────────────────
t2 = tic;
if exist(fullfile('data','cnn_model.mat'),'file')
    fprintf('\n[PHASE 2] Loading cached CNN model …\n');
    load(fullfile('data','cnn_model.mat'), 'net', 'info');
    load(fullfile('data','train_test.mat'), 'trainData', 'testData');
else
    fprintf('\n[PHASE 2] Training CNN …\n');
    [net, info, trainData, testData] = train_cnn(DS1, DS2);
end
fprintf('[PHASE 2] Done in %.1f s\n', toc(t2));

%% ──────────────────────────────────────────────────────────────────────
%  PHASE 3 – Quantization
% ──────────────────────────────────────────────────────────────────────
t3 = tic;
fprintf('\n[PHASE 3] Quantization analysis …\n');
quantize_model(net, testData);
fprintf('[PHASE 3] Done in %.1f s\n', toc(t3));

%% ──────────────────────────────────────────────────────────────────────
%  PHASE 4 – Personalization
% ──────────────────────────────────────────────────────────────────────
t4 = tic;
fprintf('\n[PHASE 4] Personalization …\n');
personalize(net, DS2);
fprintf('[PHASE 4] Done in %.1f s\n', toc(t4));

%% ──────────────────────────────────────────────────────────────────────
%  PHASE 5 – Streaming Demo
% ──────────────────────────────────────────────────────────────────────
t5 = tic;
fprintf('\n[PHASE 5] Streaming simulation …\n');
stream_demo(net, DS2, stream_record_idx);
fprintf('[PHASE 5] Done in %.1f s\n', toc(t5));

%% ──────────────────────────────────────────────────────────────────────
%  Final Report Table
% ──────────────────────────────────────────────────────────────────────
print_final_report();

fprintf('\n====================================================\n');
fprintf(' Pipeline complete.  All outputs in ./figures/ and ./results/\n');
fprintf('====================================================\n');
end

% =========================================================================
%  check_toolboxes  –  verify required MATLAB toolboxes
% =========================================================================
function check_toolboxes()
required = { ...
    'Signal Processing Toolbox',              'signal_toolbox'; ...
    'Deep Learning Toolbox',                  'neural_network_toolbox'; ...
    'Statistics and Machine Learning Toolbox','statistics_toolbox'};

optional = { ...
    'Deep Learning Toolbox Model Quantization Library', 'dl_quant'; ...
    'MATLAB Coder',                                      'coder'; ...
    'Embedded Coder',                                    'ecoder'};

fprintf('\n--- Toolbox Check ---\n');
all_ok = true;
for i = 1:size(required,1)
    ok = license('test', required{i,2});
    status = 'OK';
    if ~ok
        status = 'MISSING ← required';
        all_ok = false;
    end
    fprintf('  %-52s %s\n', required{i,1}, status);
end
for i = 1:size(optional,1)
    ok = license('test', optional{i,2});
    fprintf('  %-52s %s\n', optional{i,1}, ifelse(ok,'OK','not installed (optional)'));
end
if ~all_ok
    warning('main:missingToolbox', ...
        ['One or more toolboxes are missing on disk. ' ...
         'The pipeline will use built-in pure-MATLAB fallback engines. ' ...
         'Install them via MATLAB Add-Ons for native GPU/C-Coder acceleration.']);
end
fprintf('\n');
end

% =========================================================================
%  print_final_report  –  aggregate summary from saved result files
% =========================================================================
function print_final_report()
fprintf('\n====================================================\n');
fprintf(' FINAL RESULTS SUMMARY\n');
fprintf('====================================================\n');

% Quantization table
if exist(fullfile('results','quantization_results.csv'),'file')
    fprintf('\n--- Phase 3: Quantization ---\n');
    T = readtable(fullfile('results','quantization_results.csv'));
    disp(T);
end

% Personalization table
if exist(fullfile('results','personalization_results.csv'),'file')
    fprintf('\n--- Phase 4: Personalization (first 5 rows) ---\n');
    T = readtable(fullfile('results','personalization_results.csv'));
    disp(T(1:min(5,height(T)),:));
end

fprintf('\nFigures saved to:  ./figures/\n');
fprintf('  phase2_confusion.png  – DS2 confusion matrix\n');
fprintf('  phase2_roc.png        – DS2 ROC curves\n');
fprintf('  phase3_quantization.png – FP32 vs INT8 bar chart\n');
fprintf('  phase4_personalization.png – before/after personalization\n');
fprintf('  phase5_latency.png    – per-beat latency histogram\n');
fprintf('  phase5_ecg_snapshot.png – annotated streaming ECG\n');
end

% =========================================================================
%  banner
% =========================================================================
function banner()
fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════╗\n');
fprintf('║  Self-Personalizing On-Device ECG Arrhythmia Monitor ║\n');
fprintf('║  Course Project – Simulation-First Prototype         ║\n');
fprintf('║                                                      ║\n');
fprintf('║  DISCLAIMER: Research prototype only.                ║\n');
fprintf('║  NOT a medical or diagnostic device.                 ║\n');
fprintf('╚══════════════════════════════════════════════════════╝\n');
end

% =========================================================================
%  ifelse  –  ternary helper (MATLAB lacks ? : operator)
% =========================================================================
function out = ifelse(cond, a, b)
if cond, out = a; else, out = b; end
end
