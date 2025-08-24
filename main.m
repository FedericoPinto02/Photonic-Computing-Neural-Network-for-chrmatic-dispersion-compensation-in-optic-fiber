% main_pnn_page45.m
% Esegue training PSO/valutazione per PNN con MZI parametrizzati (page45)

clear; close all; clc;
rng(1234);

%% --- Scenario / Parametri
params.Baud = 10e9;
params.Nsps = 8;
params.Fs = params.Baud * params.Nsps;
params.Nsym_train = 2000;   % per training (rapido)
params.Nsym_eval  = 20000;  % per valutazione finale
params.modOrder = 4;

% fiber
params.beta2_ps2_per_m = -0.021; % ps^2/m
params.beta2 = params.beta2_ps2_per_m * 1e-24; % s^2/m
params.L = 125e3; % [m]

params.OSNR_dB = 34;
params.noise_model = 'AWGN';

% PNN architecture (paper: N=8 taps, dt=25ps)
params.N = 8;
params.dt = 25e-12;
k_db = [-19.0,-15.5,-14.8,-14.7,-21.4,-16.0,-18.0,-20.0];
params.k = 10.^(k_db/20); % linear amplitudes, 1xN

% PSO options (tune for speed/quality)
pso_opts.numParticles = 30;
pso_opts.maxIter = 120;
pso_opts.w = 0.7; pso_opts.c1 = 1.5; pso_opts.c2 = 1.5;

%% --- Generate training dataset (PRBS PAM4)
[tx_symbols_train, tx_wave_train, ~] = genPAM4_prbs(params.Nsym_train, params, 'prbsOrder',11, 'seed', 2025);

%% --- Sanity check: bare fiber (no PNN)
fprintf('Sanity check: bare fiber (no PNN) ...\n');
E_rx0 = fiberPropagate_freqdomain(tx_wave_train, params.Fs, params.beta2, params.L);
P_det0 = photodetect(E_rx0);
P_noisy0 = addNoise_OSNR(P_det0, params.OSNR_dB, params.Fs);
[yk0, tx_al0] = sample_and_align(P_noisy0, tx_symbols_train, params.Nsps);
[loss0, ~, ~] = loss_L2(yk0, tx_al0);
ber0 = evaluate_BER(yk0, tx_al0);
fprintf('Bare fiber -> loss = %.6g , BER ~ %.3g\n', loss0, ber0);

%% --- Setup objective for PSO (3N parameters)
objfun = @(x) loss(x, params, tx_symbols_train, tx_wave_train);

% initial vector: start near zero (theta small, phases zero)
init_theta = 0.1 * ones(params.N,1);
init_phi_u = zeros(params.N,1);
init_phi_d = zeros(params.N,1);
init_x = [init_theta; init_phi_u; init_phi_d];

fprintf('Starting PSO (3N params, N=%d)...\n', params.N);
[best_x, history] = trainer_PSO(objfun, init_x, pso_opts);

%% --- Extract best params
theta_opt = mod(best_x(1:params.N), 2*pi);
phi_u_opt = mod(best_x(params.N+1:2*params.N)+pi,2*pi)-pi;
phi_d_opt = mod(best_x(2*params.N+1:3*params.N)+pi,2*pi)-pi;
param_matrix_opt = [theta_opt(:), phi_u_opt(:), phi_d_opt(:)];

fprintf('PSO done. Best training loss = %.6g\n', min(history.bestFitness));

%% --- Final evaluation (larger dataset)
[tx_symbols_eval, tx_wave_eval, tvec_eval] = genPAM4_prbs(params.Nsym_eval, params, 'prbsOrder',11, 'seed', 777);
E_pnn_eval = PNN(tx_wave_eval, params.Fs, params.dt, params.k, param_matrix_opt);
E_rx_eval = fiberPropagate_freqdomain(E_pnn_eval, params.Fs, params.beta2, params.L);
P_det_eval = photodetect(E_rx_eval);
P_noisy_eval = addNoise_OSNR(P_det_eval, params.OSNR_dB, params.Fs);
[yk_eval, tx_sym_aligned_eval] = sample_and_align(P_noisy_eval, tx_symbols_eval, params.Nsps);
[final_loss, EL, ER] = loss_L2(yk_eval, tx_sym_aligned_eval);
final_BER = evaluate_BER(yk_eval, tx_sym_aligned_eval);

fprintf('Final eval -> loss = %.6g , BER = %.6g\n', final_loss, final_BER);

%% --- Plots
figure;
plot(history.iter, history.bestFitness, '-o'); xlabel('PSO iter'); ylabel('best loss'); title('PSO convergence');
figure;
eyewindow = params.Nsps*200; idx = 1:min(length(P_noisy_eval), eyewindow);
plot(tvec_eval(idx)*1e9, P_noisy_eval(idx)); xlabel('time (ns)'); ylabel('Detected power'); title('Eye snippet (detected power)');

% save parameters
save('pnn_page45_bestparams.mat','param_matrix_opt','history','final_loss','final_BER');

fprintf('Script main_pnn_page45 finished.\n');
