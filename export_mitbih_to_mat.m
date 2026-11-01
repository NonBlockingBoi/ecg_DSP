%% export_mitbih_to_mat.m  –  Helper: Export MIT-BIH records to .mat
%
% Run this script ONCE (requires WFDB Toolbox OR Python with wfdb package).
% It produces ./data/<recID>.mat files that preprocess.m can load offline.
%
% DISCLAIMER: Research prototype only. NOT a diagnostic device.
%
% Usage:
%   Option 1 – WFDB Toolbox installed:
%     export_mitbih_to_mat('wfdb')
%
%   Option 2 – Python wfdb installed (run from MATLAB):
%     export_mitbih_to_mat('python')
%
%   Option 3 – Already-downloaded WFDB .dat/.hea/.atr files in ./data/wfdb/:
%     export_mitbih_to_mat('local')
% =========================================================================

function export_mitbih_to_mat(method)

if nargin < 1, method = 'wfdb'; end

% All DS1 + DS2 records (paced records excluded)
all_records = { ...
    '100','101','103','105','106','108','109','111','112','113', ...
    '114','115','116','117','118','119','121','122','123','124', ...
    '200','201','202','203','205','207','208','209','210','212', ...
    '213','214','215','219','220','221','222','223','228','230', ...
    '231','232','233','234'};

if ~exist('data','dir'), mkdir('data'); end

switch lower(method)

    % ------------------------------------------------------------------
    case 'wfdb'
        % Requires WFDB Toolbox: https://physionet.org/content/wfdb-matlab/
        fprintf('Exporting via WFDB Toolbox …\n');
        for k = 1:numel(all_records)
            rec = all_records{k};
            fprintf('  %s … ', rec);
            try
                db_rec = ['mitdb/' rec];
                [sig, Fs] = rdsamp(db_rec);
                ann       = rdann(db_rec, 'atr');
                save_mat(rec, sig, Fs, ann.sample, ann.type);
                fprintf('OK\n');
            catch ME
                fprintf('FAILED: %s\n', ME.message);
            end
        end

    % ------------------------------------------------------------------
    case 'python'
        % Uses Python's wfdb package via MATLAB's py interface
        fprintf('Exporting via Python wfdb …\n');
        try
            py.importlib.import_module('wfdb');
        catch
            error('Python wfdb not available. Install: pip install wfdb');
        end
        for k = 1:numel(all_records)
            rec = all_records{k};
            fprintf('  %s … ', rec);
            try
                pyRec = py.wfdb.rdrecord(sprintf('mitdb/%s', rec), ...
                    pyargs('pn_dir','mitdb'));
                pyAnn = py.wfdb.rdann(sprintf('mitdb/%s', rec), 'atr', ...
                    pyargs('pn_dir','mitdb'));
                sig        = double(pyRec.p_signal);
                Fs         = double(pyRec.fs);
                ann_sample = double(py.array.array('d', pyAnn.sample))';
                ann_type   = cellfun(@char, cell(pyAnn.symbol), ...
                    'UniformOutput',false)';
                save_mat(rec, sig, Fs, ann_sample, ann_type);
                fprintf('OK\n');
            catch ME
                fprintf('FAILED: %s\n', ME.message);
            end
        end

    % ------------------------------------------------------------------
    case 'local'
        % Parse locally downloaded WFDB files using basic binary reader
        fprintf('Exporting from local WFDB files in ./data/wfdb/ …\n');
        fprintf('(Ensure .dat, .hea, .atr files are present)\n');
        wfdb_dir = fullfile('data','wfdb');
        for k = 1:numel(all_records)
            rec = all_records{k};
            fprintf('  %s … ', rec);
            try
                [sig, Fs, ann_sample, ann_type] = read_wfdb_local(rec, wfdb_dir);
                save_mat(rec, sig, Fs, ann_sample, ann_type);
                fprintf('OK\n');
            catch ME
                fprintf('FAILED: %s\n', ME.message);
            end
        end

    otherwise
        error('Unknown method: %s  (use wfdb|python|local)', method);
end

fprintf('\nExport complete. .mat files saved in ./data/\n');
end

% =========================================================================
function save_mat(rec, sig, Fs, ann_sample, ann_type)
outfile = fullfile('data', [rec '.mat']);
val = double(sig);
save(outfile, 'val','Fs','ann_sample','ann_type', '-v7');
end

% =========================================================================
%  read_wfdb_local  –  minimal WFDB format-16 reader (no toolbox needed)
%  Only supports MIT-BIH format 212 (12-bit, 2 samples packed per 3 bytes)
%  and format 16 (16-bit signed).
% =========================================================================
function [sig, Fs, ann_sample, ann_type] = read_wfdb_local(rec, wfdb_dir)
hea_file = fullfile(wfdb_dir, [rec '.hea']);
dat_file = fullfile(wfdb_dir, [rec '.dat']);
atr_file = fullfile(wfdb_dir, [rec '.atr']);

%% -- Parse header
fid  = fopen(hea_file,'r');
line = fgetl(fid);
fclose(fid);
parts = strsplit(strtrim(line));
% parts: recname nlead Fs N_samps
Fs      = str2double(parts{3});
N_samps = str2double(parts{4});

% Read second line to get format
fid  = fopen(hea_file,'r');
fgetl(fid);  % skip first line
sig_line = fgetl(fid);
fclose(fid);
sig_parts = strsplit(strtrim(sig_line));
fmt = str2double(sig_parts{2});

%% -- Read signal
fid = fopen(dat_file,'rb','ieee-le');
raw = fread(fid,'uint8=>uint8');
fclose(fid);

if fmt == 212
    % Format 212: 3 bytes encode 2 x 12-bit samples
    n3 = floor(numel(raw)/3)*3;
    raw = raw(1:n3);
    raw = reshape(raw, 3, []);
    s1 = int16(raw(1,:)) + int16(bitshift(bitand(raw(2,:), uint8(15)),8));
    s2 = int16(raw(3,:)) + int16(bitshift(raw(2,:), -4)) * int16(16);
    sig = double([s1(:); s2(:)]);
    sig = reshape(sig(1:N_samps*2), 2, N_samps)';
elseif fmt == 16
    data = typecast(raw, 'int16');
    sig  = double(reshape(data, 2, N_samps)');
else
    error('Unsupported WFDB format: %d', fmt);
end

%% -- Parse annotation file (.atr)
fid = fopen(atr_file,'rb','ieee-le');
raw_ann = fread(fid,'uint16=>uint16');
fclose(fid);

ann_sample = [];
ann_type   = {};
aami_chars = 'NLReJjAaFVEfQ/';   % subset of MIT-BIH codes

sample_acc = 0;
for i = 1:numel(raw_ann)
    word  = raw_ann(i);
    anntyp= bitshift(word, -10);
    diff  = double(bitand(word, uint16(1023)));
    if anntyp == 0 && diff == 0, break; end  % end-of-file marker
    sample_acc = sample_acc + diff;
    if anntyp >= 1 && anntyp <= 49
        % Map WFDB anntyp number → character
        wfdb_sym = wfdb_anntype_to_char(anntyp);
        if ~isempty(wfdb_sym)
            ann_sample(end+1) = sample_acc; %#ok<AGROW>
            ann_type{end+1}   = wfdb_sym;   %#ok<AGROW>
        end
    end
end
end

% =========================================================================
function sym = wfdb_anntype_to_char(t)
% WFDB anntype number to character (WFDB standard table)
map = {1,'N'; 2,'L'; 3,'R'; 4,'a'; 5,'V'; 6,'F'; 7,'J'; 8,'A'; ...
       9,'S'; 10,'E'; 11,'j'; 12,'/'; 13,'Q'; 14,'~'; 16,'|'; ...
       17,'s'; 18,'T'; 19,'*'; 20,'D'; 21,'='; 22,'"'; 23,'@'; ...
       25,'x'; 26,'f'; 27,'('; 28,')'; 29,'p'; 30,'t'; 31,'u'; 32,'`'; ...
       33,'\'';34,'!'; 35,'^'; 36,'|'; 37,'+'; 38,'~'; 39,'Q'; 40,'?'; ...
       41,'_'; 42,'['; 43,'!'; 44,']'; 41,'e'};
sym = '';
for i = 1:size(map,1)
    if map{i,1} == t
        sym = map{i,2};
        return;
    end
end
end
