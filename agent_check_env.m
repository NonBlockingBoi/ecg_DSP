%% agent_check_env.m  –  Agent-run environment + toolbox checker
% Called by the agent via: matlab -batch "run('agent_check_env.m')"
% Outputs go to stdout so the agent can read them.

fprintf('\n========================================\n');
fprintf('  MATLAB Environment & Toolbox Report\n');
fprintf('========================================\n');
fprintf('MATLAB Version : %s\n', version);
fprintf('Working Dir    : %s\n', pwd);

%% All installed toolboxes
v = ver;
fprintf('\n--- Installed Toolboxes (%d) ---\n', numel(v));
for k = 1:numel(v)
    fprintf('  %-52s  v%s\n', v(k).Name, v(k).Version);
end

%% License checks for the ECG project
fprintf('\n--- License Checks ---\n');
required = { ...
    'Signal Processing Toolbox',              'signal_toolbox'; ...
    'Deep Learning Toolbox',                  'neural_network_toolbox'; ...
    'Statistics and Machine Learning Toolbox','statistics_toolbox'};

optional = { ...
    'Deep Learning Toolbox Model Quantization Library','dl_quant'; ...
    'MATLAB Coder',                                    'coder'; ...
    'Embedded Coder',                                  'ecoder'};

all_ok = true;
for i = 1:size(required,1)
    ok = license('test', required{i,2});
    if ok
        tag = 'PRESENT  ✓ (required)';
    else
        tag = 'MISSING  ✗ (required)';
        all_ok = false;
    end
    fprintf('  %-52s  %s\n', required{i,1}, tag);
end
for i = 1:size(optional,1)
    ok = license('test', optional{i,2});
    if ok
        tag = 'present  (optional)';
    else
        tag = 'absent   (optional)';
    end
    fprintf('  %-52s  %s\n', optional{i,1}, tag);
end

if all_ok
    fprintf('\n[RESULT] All required toolboxes PRESENT. Ready to run main.m\n');
else
    fprintf('\n[RESULT] Some required toolboxes MISSING. Install them first.\n');
end
fprintf('========================================\n');
