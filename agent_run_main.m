%% agent_run_main.m  –  Agent-driven full pipeline execution
% Called by: matlab -batch "cd('...ecg'); run('agent_run_main.m')"

fprintf('=== Agent: Initialising MATLAB R2026b toolboxes ===\n');

%% Force-load required toolbox paths (needed in -batch mode on some installs)
matlab_root = matlabroot;

% Signal Processing Toolbox
sp_path = fullfile(matlab_root, 'toolbox', 'signal', 'signal');
if exist(sp_path, 'dir'), addpath(sp_path); end
sp_path2 = fullfile(matlab_root, 'toolbox', 'signal', 'sigtools');
if exist(sp_path2, 'dir'), addpath(sp_path2); end

% Deep Learning Toolbox
dl_paths = { ...
    fullfile(matlab_root,'toolbox','nnet','nnet'), ...
    fullfile(matlab_root,'toolbox','nnet','nnetutils'), ...
    fullfile(matlab_root,'toolbox','deeplearning'), ...
    fullfile(matlab_root,'toolbox','nnet','cnn')};
for i = 1:numel(dl_paths)
    if exist(dl_paths{i},'dir'), addpath(dl_paths{i}); end
end

% Statistics & ML Toolbox
stats_path = fullfile(matlab_root,'toolbox','stats','stats');
if exist(stats_path,'dir'), addpath(stats_path); end

% Verify butter is now accessible
if ~exist('butter','file')
    % Try restoredefaultpath + rehash
    restoredefaultpath;
    rehash toolboxcache;
end

fprintf('butter available    : %s\n', mat2str(exist('butter','file')>0));
fprintf('imageInputLayer     : %s\n', mat2str(exist('imageInputLayer','file')>0));
fprintf('confusionmat        : %s\n', mat2str(exist('confusionmat','file')>0));

fprintf('\n=== Agent: Starting main.m pipeline (synthetic demo mode) ===\n');
try
    main(false, 1);
    fprintf('\n=== Agent: Pipeline completed successfully ===\n');
catch ME
    fprintf('\n=== Agent: Pipeline ERROR ===\n');
    fprintf('Message : %s\n', ME.message);
    fprintf('Stack:\n');
    for k = 1:numel(ME.stack)
        fprintf('  %s  line %d\n', ME.stack(k).name, ME.stack(k).line);
    end
    rethrow(ME);
end