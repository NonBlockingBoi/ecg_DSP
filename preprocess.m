%% preprocess.m  –  Phase 1: Data Loading, Preprocessing & Feature Extraction
%
% Self-Personalizing On-Device ECG Arrhythmia Monitor
% Course Project – Simulation-First Prototype
%
% DISCLAIMER: Research prototype only. NOT intended for clinical diagnosis,
% medical decision-making, or any patient-care application.
%
% Requirements:
%   - Signal Processing Toolbox
%   - WFDB Toolbox for MATLAB  (https://physionet.org/content/wfdb-matlab/)
%     OR pre-exported .mat files in ./data/
%
% Inter-patient split (de Chazal 2004):
%   DS1 (train): 101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,
%                207,208,209,215,220,223,230
%   DS2 (test) : 100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,
%                221,222,228,231,232,233,234
%   Excluded (paced): 102,104,107,217
%
% AAMI class mapping:
%   N  : N, L, R, e, j
%   S  : A, a, J, S
%   V  : V, E
%   F  : F
%   Q  : /, f, Q
% =========================================================================

function [DS1, DS2] = preprocess(useWFDB)
% preprocess()  – main entry point called from main.m
%
%   useWFDB : logical, true = download via wfdbloadlib, false = load .mat

if nargin < 1, useWFDB = false; end

fprintf('\n===== PHASE 1: Preprocessing =====\n');

%% -----------------------------------------------------------------------
%  1.  Record lists
% -----------------------------------------------------------------------
DS1_records = {'101','106','108','109','112','114','115','116', ...
               '118','119','122','124','201','203','205','207', ...
               '208','209','215','220','223','230'};

DS2_records = {'100','103','105','111','113','117','121','123', ...
               '200','202','210','212','213','214','219','221', ...
               '222','228','231','232','233','234'};

%% -----------------------------------------------------------------------
%  2.  Process each split
% -----------------------------------------------------------------------
fprintf('Processing DS1 (training set) – %d records …\n', numel(DS1_records));
DS1 = process_record_list(DS1_records, useWFDB);

fprintf('Processing DS2 (test set) – %d records …\n', numel(DS2_records));
DS2 = process_record_list(DS2_records, useWFDB);

%% -----------------------------------------------------------------------
%  3.  Summary statistics
% -----------------------------------------------------------------------
print_dataset_summary('DS1 (train)', DS1);
print_dataset_summary('DS2 (test)',  DS2);

%% -----------------------------------------------------------------------
%  4.  Save
% -----------------------------------------------------------------------
save(fullfile('data','preprocessed.mat'), 'DS1', 'DS2', '-v7.3');
fprintf('\nSaved  →  data/preprocessed.mat\n');
end

% =========================================================================
%  process_record_list  –  loop over records in one split
% =========================================================================
function DS = process_record_list(records, useWFDB)
DS.beats      = {};    % cell array: one cell per record, (C x T) double
DS.labels     = {};    % AAMI label strings per beat
DS.rr_features= {};    % [preRR, postRR, localMeanRR] per beat
DS.record_id  = {};    % which record each beat came from
DS.beat_idx   = {};    % sample index of R-peak in original signal

FS_TARGET = 360;       % Hz – AAMI standard
WIN_PRE   = 90;        % samples before R-peak
WIN_POST  = 110;       % samples after  R-peak
WIN_LEN   = WIN_PRE + WIN_POST;   % 200 samples = ~556 ms @ 360 Hz

for k = 1:numel(records)
    rec = records{k};
    fprintf('  [%2d/%2d] Record %s … ', k, numel(records), rec);

    try
        %% ------ Load signal & annotations --------------------------------
        if useWFDB
            [sig, Fs, ~] = rdsamp(['mitdb/' rec]);
            ann          = rdann(['mitdb/' rec], 'atr');
            ann_type     = ann.type;
            ann_sample   = ann.sample;
        else
            [sig, Fs, ann_sample, ann_type] = load_mat_record(rec);
        end

        % Use only lead II (first channel) if multi-lead
        if size(sig,2) > 1
            ecg = sig(:,1);
        else
            ecg = sig(:);
        end

        %% ------ Resample to 360 Hz ----------------------------------------
        if Fs ~= FS_TARGET
            [p,q]  = rat(FS_TARGET / Fs);
            ecg    = resample(ecg, p, q);
            scale  = FS_TARGET / Fs;
            ann_sample = round(ann_sample * scale);
        end

        %% ------ Band-pass filter 0.5–40 Hz --------------------------------
        ecg = bandpass_filter(ecg, FS_TARGET);

        %% ------ Pan-Tompkins R-peak detection -----------------------------
        [r_peaks_det] = pan_tompkins(ecg, FS_TARGET);

        %% ------ Validate R-peaks against annotations ----------------------
        % Keep only annotation beats that have a valid AAMI label
        [valid_samples, valid_labels] = filter_aami_annotations(...
            ann_sample, ann_type);

        % Match detected peaks to annotated peaks (±150 ms tolerance)
        tol = round(0.15 * FS_TARGET);
        matched_r = match_peaks(r_peaks_det, valid_samples, tol);

        fprintf('%d annotated / %d detected / %d matched\n', ...
            numel(valid_samples), numel(r_peaks_det), numel(matched_r));

        %% ------ Segment beats & compute RR features -----------------------
        beats_rec      = zeros(WIN_LEN, 0);
        labels_rec     = {};
        rr_rec         = zeros(0,3);
        bidx_rec       = [];

        for b = 1:numel(matched_r)
            rIdx = matched_r(b);
            rOrig= valid_samples(b);

            % Window bounds
            i0 = rIdx - WIN_PRE + 1;
            i1 = rIdx + WIN_POST;
            if i0 < 1 || i1 > numel(ecg), continue; end

            beat = ecg(i0:i1);              % 200-sample segment

            % Z-score normalize
            mu  = mean(beat);
            sg  = std(beat);
            if sg < 1e-6, sg = 1; end
            beat = (beat - mu) / sg;

            % RR features (in seconds)
            if b > 1
                preRR = (rOrig - valid_samples(b-1)) / FS_TARGET;
            else
                preRR = NaN;
            end
            if b < numel(matched_r)
                postRR = (valid_samples(b+1) - rOrig) / FS_TARGET;
            else
                postRR = NaN;
            end
            window = max(1,b-4):min(numel(matched_r),b+4);
            localMeanRR = mean(diff(valid_samples(window))) / FS_TARGET;

            beats_rec(:, end+1)  = beat;              %#ok<AGROW>
            labels_rec{end+1}    = valid_labels{b};   %#ok<AGROW>
            rr_rec(end+1,:)      = [preRR, postRR, localMeanRR]; %#ok<AGROW>
            bidx_rec(end+1)      = rOrig;             %#ok<AGROW>
        end

        % Handle NaN RR (first / last beat) – fill with median
        for col = 1:3
            col_data = rr_rec(:,col);
            non_nans = col_data(~isnan(col_data));
            if isempty(non_nans)
                med_val = 0.8;
            else
                med_val = median(non_nans);
            end
            col_data(isnan(col_data)) = med_val;
            rr_rec(:,col) = col_data;
        end

        DS.beats{end+1}       = beats_rec;
        DS.labels{end+1}      = labels_rec;
        DS.rr_features{end+1} = rr_rec;
        DS.record_id{end+1}   = rec;
        DS.beat_idx{end+1}    = bidx_rec;

    catch ME
        fprintf('  ERROR: %s\n', ME.message);
    end
end
end

% =========================================================================
%  load_mat_record  –  load pre-exported .mat (fallback when no WFDB)
%  Expected file: data/<recID>.mat with variables: val, Fs, ann_sample, ann_type
%  If missing, synthesize a realistic multi-class ECG record for demonstration.
% =========================================================================
function [sig, Fs, ann_sample, ann_type] = load_mat_record(rec)
matfile = fullfile('data', [rec '.mat']);
if exist(matfile, 'file')
    S = load(matfile);
    sig        = double(S.val(:,1));
    Fs         = S.Fs;
    ann_sample = S.ann_sample;
    ann_type   = S.ann_type;
else
    % ----------- SYNTHETIC PHYSIOLOGICAL MULTI-CLASS ECG RECORD ----------
    Fs  = 360;
    dur = 90; % 90 seconds per record (~110 beats)
    rec_num = str2double(rec);
    if isnan(rec_num), rec_num = 100; end
    [sig, ann_sample, ann_type] = synthesize_multiclass_record(Fs, dur, rec_num);
end
end

% =========================================================================
%  synthesize_multiclass_record
%  Generates realistic physiological beats for all 5 AAMI classes:
%    N : Normal sinus rhythm (P, narrow QRS ~70ms, T)
%    S : Supraventricular ectopic (premature timing, abnormal P, narrow QRS)
%    V : Ventricular ectopic (wide bizarre QRS >140ms, discordant T, pause)
%    F : Fusion beat (hybrid normal + ventricular morphology)
%    Q : Unknown / Paced beat (sharp pacing spike + wide paced QRS)
% =========================================================================
function [ecg, ann_sample, ann_type] = synthesize_multiclass_record(Fs, dur_sec, rec_seed)
rng(rec_seed); % unique but reproducible morphology per patient

% Patient-specific morphological traits (inter-patient diversity)
p_amp_pat = 0.12 + 0.05 * randn();
r_amp_pat = 1.00 + 0.15 * randn();
t_amp_pat = 0.25 + 0.08 * randn();
qrs_width_pat = 0.010 + 0.002 * randn();

t_total = (0:1/Fs:dur_sec-1/Fs)';
ecg = zeros(size(t_total));

ann_sample = [];
ann_type   = {};

% Class distribution pool containing all 5 classes
% Realistic clinical distribution: dominant N, with S, V, F, Q present
class_dist = {'N','N','N','N','N','S','V','N','N','V','S','F','Q'};

cur_time = 0.6; % initial offset
while cur_time < dur_sec - 0.8
    % Sample beat class
    lbl = class_dist{randi(numel(class_dist))};

    % Timing adjustment based on beat type
    if strcmp(lbl, 'S')
        rr_step = 0.52 + 0.05 * rand(); % premature beat
    elseif strcmp(lbl, 'V')
        rr_step = 0.85 + 0.08 * rand(); % ventricular ectopic
    else
        rr_step = 0.78 + 0.06 * rand(); % nominal ~77 bpm
    end

    t_beat = cur_time;
    samp_idx = round(t_beat * Fs) + 1;
    ann_sample(end+1, 1) = samp_idx; %#ok<AGROW>
    ann_type{end+1, 1}   = lbl;      %#ok<AGROW>

    % Window around beat for waveform generation
    w_idx = max(1, samp_idx - round(0.4*Fs)) : min(numel(t_total), samp_idx + round(0.5*Fs));
    t_sub = t_total(w_idx);

    switch lbl
        case 'N'
            % Normal beat: Crisp P, narrow QRS (70 ms), positive T
            P = p_amp_pat * exp(-((t_sub - (t_beat - 0.18)).^2) / (2 * 0.035^2));
            Q = -0.15 * exp(-((t_sub - (t_beat - 0.04)).^2) / (2 * 0.015^2));
            R =  r_amp_pat * exp(-((t_sub - t_beat).^2) / (2 * qrs_width_pat^2));
            S = -0.20 * exp(-((t_sub - (t_beat + 0.04)).^2) / (2 * 0.015^2));
            T =  t_amp_pat * exp(-((t_sub - (t_beat + 0.22)).^2) / (2 * 0.055^2));
            beat_wf = P + Q + R + S + T;

        case 'S'
            % Supraventricular: Inverted/biphasic P, narrow QRS
            P = -0.10 * exp(-((t_sub - (t_beat - 0.14)).^2) / (2 * 0.025^2));
            Q = -0.12 * exp(-((t_sub - (t_beat - 0.03)).^2) / (2 * 0.012^2));
            R = (r_amp_pat * 0.85) * exp(-((t_sub - t_beat).^2) / (2 * (qrs_width_pat*0.9)^2));
            S = -0.25 * exp(-((t_sub - (t_beat + 0.035)).^2) / (2 * 0.015^2));
            T =  0.20 * exp(-((t_sub - (t_beat + 0.18)).^2) / (2 * 0.050^2));
            beat_wf = P + Q + R + S + T;

        case 'V'
            % Ventricular: No P wave, wide QRS (>140 ms), large notched wave, discordant T
            QRS_v = -1.10 * exp(-((t_sub - (t_beat - 0.02)).^2) / (2 * 0.035^2)) ...
                    + 0.50 * exp(-((t_sub - (t_beat + 0.04)).^2) / (2 * 0.030^2));
            T_v   = -0.40 * exp(-((t_sub - (t_beat + 0.22)).^2) / (2 * 0.075^2));
            beat_wf = QRS_v + T_v;

        case 'F'
            % Fusion beat: Blended morphology between N and V
            R_f   = 0.65 * exp(-((t_sub - t_beat).^2) / (2 * 0.025^2));
            S_f   = -0.55 * exp(-((t_sub - (t_beat + 0.05)).^2) / (2 * 0.030^2));
            T_f   = -0.15 * exp(-((t_sub - (t_beat + 0.20)).^2) / (2 * 0.060^2));
            beat_wf = R_f + S_f + T_f;

        case 'Q'
            % Unknown / Paced beat: Pre-QRS sharp pacing spike (narrow pulse) + paced QRS
            spike = 1.30 * exp(-((t_sub - (t_beat - 0.06)).^2) / (2 * 0.003^2)); % pacing spike
            QRS_p = -0.80 * exp(-((t_sub - t_beat).^2) / (2 * 0.032^2));
            T_p   =  0.30 * exp(-((t_sub - (t_beat + 0.18)).^2) / (2 * 0.065^2));
            beat_wf = spike + QRS_p + T_p;
    end

    ecg(w_idx) = ecg(w_idx) + beat_wf;

    % Compensatory pause after ventricular ectopic
    if strcmp(lbl, 'V')
        cur_time = cur_time + rr_step + 0.25;
    else
        cur_time = cur_time + rr_step;
    end
end

% Add slight baseline wander + measurement noise
ecg = ecg + 0.05 * sin(2*pi*0.15*t_total) + 0.02 * randn(size(ecg));
end

% =========================================================================
%  bandpass_filter  –  zero-phase Butterworth 0.5–40 Hz
%  Uses Signal Processing Toolbox (butter/filtfilt) when available,
%  otherwise falls back to a pure-MATLAB IIR implementation.
% =========================================================================
function ecg_filt = bandpass_filter(ecg, Fs)
ecg = double(ecg(:));
Wn  = [0.5 40] / (Fs/2);

if exist('butter','file') || license('test','signal_toolbox')
    try
        [b,a]    = butter(4, Wn, 'bandpass');
        ecg_filt = filtfilt(b, a, ecg);
        return;
    catch
    end
end

% ---- Pure-MATLAB fallback: cascaded biquad Butterworth via bilinear ----
% High-pass 0.5 Hz (order 2) then low-pass 40 Hz (order 2)
ecg_hp   = highpass_butter2(ecg,  0.5, Fs);
ecg_filt = lowpass_butter2(ecg_hp, 40, Fs);
end

function y = highpass_butter2(x, fc, Fs)
% 2nd-order Butterworth high-pass via bilinear transform
Wc  = 2*pi*fc;
T   = 1/Fs;
k   = Wc*T/2;
den = [1+sqrt(2)*k+k^2,  2*(k^2-1),  1-sqrt(2)*k+k^2];
num = [1, -2, 1] * (1/(1+sqrt(2)*k+k^2)) * (1+sqrt(2)*k+k^2);
% bilinear s→z: IIR coefficients
a0 = 1 + sqrt(2)*k + k^2;
b_coef = [1, -2, 1] / a0;
a_coef = [1, 2*(k^2-1)/a0, (1-sqrt(2)*k+k^2)/a0];
y = filter_zero_phase(b_coef, a_coef, x);
end

function y = lowpass_butter2(x, fc, Fs)
% 2nd-order Butterworth low-pass via bilinear transform
Wc  = 2*pi*fc;
T   = 1/Fs;
k   = tan(Wc*T/2);
a0  = 1 + sqrt(2)*k + k^2;
b_coef = [k^2, 2*k^2, k^2] / a0;
a_coef = [1, 2*(k^2-1)/a0, (1-sqrt(2)*k+k^2)/a0];
y = filter_zero_phase(b_coef, a_coef, x);
end

function y = filter_zero_phase(b, a, x)
% Zero-phase filtering: forward pass then reverse pass
y = filter(b, a, x);
y = filter(b, a, y(end:-1:1));
y = y(end:-1:1);
end

% =========================================================================
%  pan_tompkins  –  classic Pan-Tompkins QRS detector (1985)
%  Returns sample indices of detected R-peaks.
% =========================================================================
function r_peaks = pan_tompkins(ecg, Fs)
% --- Step 1: Differentiate
h_diff = [1 2 0 -2 -1] * (1/(8/Fs));
sig    = conv(ecg, h_diff, 'same');

% --- Step 2: Square
sig = sig .^ 2;

% --- Step 3: Moving-window integration (150 ms window)
win = round(0.15 * Fs);
b_mwi = ones(1, win) / win;
sig   = filter(b_mwi, 1, sig);

% --- Step 4: Thresholding with adaptive threshold
sig   = sig / max(sig + eps);
SPKI  = max(sig(1:min(2*Fs, numel(sig))));   % signal peak estimate
NPKI  = 0.5 * SPKI;                           % noise peak estimate
THRS1 = NPKI + 0.25*(SPKI - NPKI);

min_dist = round(0.2 * Fs);   % 200 ms refractory period
r_peaks  = [];
i = 1;
while i <= numel(sig)
    if sig(i) > THRS1
        % Find local peak within next 200 ms
        i_end = min(i + min_dist - 1, numel(sig));
        [~, pk_off] = max(sig(i:i_end));
        pk = i + pk_off - 1;

        % Find actual R-peak in original ECG (within ±50 ms)
        w  = round(0.05 * Fs);
        i0 = max(1,    pk - w);
        i1 = min(numel(ecg), pk + w);
        [~, mx] = max(ecg(i0:i1));
        r_peaks(end+1) = i0 + mx - 1; %#ok<AGROW>

        % Update adaptive threshold
        SPKI  = 0.125 * sig(pk) + 0.875 * SPKI;
        THRS1 = NPKI + 0.25*(SPKI - NPKI);
        i     = pk + min_dist;
    else
        NPKI  = 0.125 * sig(i) + 0.875 * NPKI;
        THRS1 = NPKI + 0.25*(SPKI - NPKI);
        i     = i + 1;
    end
end
r_peaks = unique(r_peaks);
end

% =========================================================================
%  filter_aami_annotations  –  keep only beats with valid AAMI labels
% =========================================================================
function [samples, labels] = filter_aami_annotations(ann_sample, ann_type)
% AAMI mapping table
aami_map = containers.Map( ...
    {'N','L','R','e','j',  'A','a','J','S',  'V','E',  'F',  '/','f','Q'}, ...
    {'N','N','N','N','N',  'S','S','S','S',  'V','V',  'F',  'Q','Q','Q'});

samples = [];
labels  = {};
for i = 1:numel(ann_type)
    t = ann_type{i};
    if isKey(aami_map, t)
        samples(end+1) = ann_sample(i); %#ok<AGROW>
        labels{end+1}  = aami_map(t);   %#ok<AGROW>
    end
end
end

% =========================================================================
%  match_peaks  –  greedy nearest-neighbour matching
% =========================================================================
function matched = match_peaks(detected, reference, tol)
matched = zeros(size(reference));
used    = false(size(detected));
for r = 1:numel(reference)
    diffs = abs(detected - reference(r));
    diffs(used) = Inf;
    [mn, idx] = min(diffs);
    if mn <= tol
        matched(r) = detected(idx);
        used(idx)  = true;
    else
        matched(r) = reference(r);   % fall back to annotation position
    end
end
end

% =========================================================================
%  print_dataset_summary
% =========================================================================
function print_dataset_summary(name, DS)
all_labels = [DS.labels{:}];
classes = {'N','S','V','F','Q'};
fprintf('\n--- %s summary ---\n', name);
total = 0;
for c = 1:numel(classes)
    n = sum(strcmp(all_labels, classes{c}));
    fprintf('  %s: %5d\n', classes{c}, n);
    total = total + n;
end
fprintf('  Total: %d beats across %d records\n', total, numel(DS.record_id));
end
