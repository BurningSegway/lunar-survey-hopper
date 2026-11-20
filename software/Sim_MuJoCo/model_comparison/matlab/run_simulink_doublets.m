function run_simulink_doublets(slx_file, mlx_file, wheel_inertia, x0)
% Run the user's Simulink model (state_feedback_and_nonlinear.slx) on a
% patched copy, for model_comparison (--matlab). The originals are not
% changed.
% slx_file      : the Simulink model; its two dynamics MATLAB Function blocks
%                 hard-code J_w = 0.00225 * eye(3), so J_wi_val in the live
%                 script does not reach them
% mlx_file      : the live script (angular_momentum_tau_wheelspeeds.mlx),
%                 run first for A6, B6 and K (the model's gain blocks)
% wheel_inertia : J_w,i for the copy [kg*m^2] (CAD: 4.422e-3)
% x0            : Integrator2's initial state (logged as x_val)
% Writes doublets_cad_variable.csv (the model's own variable step) and
% doublets_cad_ode4_1ms.csv (fixed step, ode4 at 1 ms) next to this file.
here = fileparts(mfilename('fullpath'));
work = tempname; mkdir(work);
copyfile(slx_file, fullfile(work, 'sfn_copy.slx'));
copyfile(mlx_file, fullfile(work, 'live_copy.mlx'));
old = cd(work); cleanup = onCleanup(@() cd(old));
evalin('base', 'run(''live_copy.mlx'')');
mdl = 'sfn_copy';
load_system(mdl);
for c = sfroot().find('-isa', 'Stateflow.EMChart')'
    if contains(c.Script, 'J_w = 0.00225 * eye(3);')
        c.Script = strrep(c.Script, 'J_w = 0.00225 * eye(3);', ...
            sprintf('J_w = %.10g * eye(3);', wheel_inertia));
        fprintf('patched %s\n', c.Path);
    end
end
set_param([mdl '/Integrator2'], 'InitialCondition', mat2str(x0));
runs = {'variable', {}; ...
        'ode4_1ms', {'SolverType', 'Fixed-step', 'Solver', 'ode4', 'FixedStep', '1e-3'}};
for r = 1:size(runs, 1)
    out = sim(mdl, runs{r, 2}{:});
    ts = out.logsout.get('x_val').Values;
    x = squeeze(ts.Data); if size(x, 2) ~= 10, x = x.'; end
    T = array2table([ts.Time x], 'VariableNames', {'t', 'q1', 'q2', 'q3', 'q4', ...
        'wx', 'wy', 'wz', 'Omega_x', 'Omega_y', 'Omega_z'});
    writetable(T, fullfile(here, ['doublets_cad_' runs{r, 1} '.csv']));
    fprintf('%s: %d samples, end %s\n', runs{r, 1}, numel(ts.Time), mat2str(x(end, :), 5));
end
close_system(mdl, 0);
end
