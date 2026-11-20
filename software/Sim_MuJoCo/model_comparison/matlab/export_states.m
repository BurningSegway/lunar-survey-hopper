function export_states(mat_file, csv_file)
% Write a Simulink run's states to CSV for model_comparison (--matlab).
% mat_file : MAT-file with one timeseries of the 10 states
%            [q1 q2 q3 q4 wx wy wz Omega_x Omega_y Omega_z] (e.g. x_val)
% csv_file : output, columns t, q1..q4, wx..wz, Omega_x..Omega_z
s = load(mat_file);
names = fieldnames(s);
ts = s.(names{1});
x = squeeze(ts.Data);
if size(x, 2) ~= 10, x = x.'; end
T = array2table([ts.Time x], 'VariableNames', {'t', 'q1', 'q2', 'q3', 'q4', ...
    'wx', 'wy', 'wz', 'Omega_x', 'Omega_y', 'Omega_z'});
writetable(T, csv_file);
end
