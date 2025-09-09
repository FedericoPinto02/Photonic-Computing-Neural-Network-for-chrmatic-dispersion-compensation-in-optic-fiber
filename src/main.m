clear; close all; clc;
rng(1234);  


%% 1) PARAMETRI DI CANALE / PNN

params.Baud = 10e9;
params.Nsps = 8;
params.Fs   = params.Baud*params.Nsps;
params.mode = 'PO';

% fibra
params.beta2_ps2_per_m = -0.021;                
params.beta2 = params.beta2_ps2_per_m * 1e-24; 
params.L = 125e3;                               

% rumore
params.OSNR_dB = 34;

% PNN (come nel paper: 8 tap, 25 ps)
params.N  = 8;
params.dt = 25e-12;

% perdite per tap (dB -> lineare)
k_db     = [-19.0,-15.5,-14.8,-14.7,-21.4,-16.0,-18.0,-20.0];
params.k = 10.^(k_db/20);


%% 2) Caricamento dataset


train = load('DataSets/dataset_val.mat'); 
val   = load('DataSets/dataset_val.mat');
test  = load('DataSets/dataset_test.mat');

% Allineo timing params a quanto salvato nei dataset
params.Fs   = train.Fs;
params.Nsps = train.Nsps;
params.Baud = train.Baud;

theta_id = (pi/2)*ones(params.N,1);  
phi_u_id = zeros(params.N,1);
phi_d_id = zeros(params.N,1);
PM_id = [theta_id, phi_u_id, phi_d_id];

% Streams di rumore separati e riproducibili:
rs_train = RandStream('mt19937ar','Seed', 12345);  % fisso durante PSO/ADAM
rs_val   = RandStream('mt19937ar','Seed', 22345);  % per validation
rs_test  = RandStream('mt19937ar','Seed', 32345);  % per test finale 

lm_opts = struct('pLow',0.25,'pHigh',0.75, ...
                 'lambda_range',1.5, ...
                 'margin_pct',0.10, ...
                 'range_min_pct',0.90);


%% 3) fiber su VALIDATION 

P_noisy0 = forward_rx_chain(train.tx_wave, PM_id, params);
[yk0, tx_al0, off_est] = sample_and_align_auto(P_noisy0, train.tx_symbols, params.Nsps, train.ref_power_wave);
[loss0,~] = loss_margin(yk0, tx_al0, lm_opts);
ber0 = evaluate_BER_MAP_verbose(yk0, tx_al0); 
fprintf('Sanity check: bare fiber (TRAIN) -> loss = %.6f , BER ~ %.3f\n', loss0, ber0);


%% 4) DEFINIZIONE OBIETTIVO 

P_noisy_tr_id = forward_rx_chain(train.tx_wave, PM_id, params);
[~, ~, off_star_train] = sample_and_align_auto( ...
    P_noisy_tr_id, train.tx_symbols, params.Nsps, train.ref_power_wave, 6, 3, lm_opts);

align_fixed = struct('fixed_offset', off_star_train, 'do_resample', false);

objfun = @(x) loss(x, params, train.tx_symbols, train.tx_wave, ...
    struct('noiseStream', rs_train, 'refPowerWave', train.ref_power_wave, ...
           'lossOpts', lm_opts, 'alignOpts', align_fixed));


%% 5) INIZIALIZZAZIONE PARAMETRI PSO (3N)

init_x = zeros(params.N,1); 

% Opzioni PSO
pso_opts.numParticles = 30;
pso_opts.maxIter = 120;
pso_opts.w = 0.7; pso_opts.c1 = 1.5; pso_opts.c2 = 1.5;


%% 6) TRAINING con PSO

pso_opts.initSpread = 0.5*pi;
fprintf('Starting PSO (%d params, N=%d, mode=%s)\n', numel(init_x), params.N, params.mode);
[best_x, history] = trainer_PSO(objfun, init_x, pso_opts);
fprintf('PSO done. Best training loss = %.6g\n', min(history.bestFitness));


%% 7) REFINEMENT con ADAM 

use_adam = true;  % metti a false per skippare ADAM

if use_adam
    fprintf('Starting ADAM refinement...\n');
    adam_opts.maxIter = 60;          
    adam_opts.alpha   = 0.02;       
    adam_opts.grad_estimator = 'spsa'; 
    adam_opts.c_spsa  = 0.01;
    adam_opts.wrap_2pi = true;
    adam_opts.clip_norm = 0.0;
    adam_opts.verbose = true;

    [x_refined, hist_adam] = trainer_ADAM(objfun, best_x, adam_opts);
    x_use = x_refined;
end


%% 8) ESTRAGO PARAMETRI MZI DAI VETTORI x

N = params.N;
wrapPi = @(a) mod(a + pi, 2*pi) - pi; 

mode = 'FULL';
if isfield(params,'mode') && ~isempty(params.mode)
    mode = upper(params.mode);
end

switch mode
    case 'PO'   % Phase-Only: ottimizziamo solo le N fasi del ramo up
        theta_opt = (pi/2)*ones(N,1);
        phi_u_opt = wrapPi(x_use(:));
        phi_d_opt = zeros(N,1);

    case 'FULL' % 3N parametri [theta, phi_u, phi_d]
        x_use = x_use(:);
        theta_opt = wrapPi(x_use(1:N));
        phi_u_opt = wrapPi(x_use(N+1:2*N));
        phi_d_opt = wrapPi(x_use(2*N+1:3*N));

    otherwise
        error('Main: params.mode sconosciuto: %s', mode);
end

param_matrix_opt = [theta_opt, phi_u_opt, phi_d_opt];


%% 9) VALUTAZIONE SU VALIDATION 


P_noisy_val = forward_rx_chain(val.tx_wave, param_matrix_opt, params);

[~, ~, off_star_val] = sample_and_align_auto( ...
    P_noisy_val, val.tx_symbols, params.Nsps, val.ref_power_wave, 6, 3, lm_opts);

loss_val = compute_loss_L2(P_noisy_val, val.tx_symbols, params.Nsps, val.ref_power_wave, lm_opts, struct());

[yk_val, tx_al_val, off_val, info_align_val] = sample_and_align_auto( ...
    P_noisy_val, val.tx_symbols, params.Nsps, val.ref_power_wave, 2*params.Nsps, 2);

[val_loss,~] = loss_margin(yk_val, tx_al_val, lm_opts);
[val_BER, thr_val, rank_map_val, info_ber_val] = evaluate_BER_MAP_verbose(yk_val, tx_al_val, true);

fprintf('Validation -> off0=%d, off*=%d, loss=%.3e, BER=%.6f\n', ...
    info_align_val.off0, off_val, val_loss, val_BER);


%% 10) VALUTAZIONE SU TEST 
P_noisy_test = forward_rx_chain(test.tx_wave, param_matrix_opt, params);

[~, ~, off_star_tst] = sample_and_align_auto( ...
    P_noisy_test, test.tx_symbols, params.Nsps, test.ref_power_wave, 6, 3, lm_opts);

a1 = struct('fixed_offset', off_star_tst, 'do_resample', false);
[yk_tst, tx_al_tst] = sample_and_align_auto( ...
    P_noisy_test, test.tx_symbols, params.Nsps, test.ref_power_wave, 0, 0, lm_opts, a1);

loss_tst = compute_loss_L2(P_noisy_test, test.tx_symbols, params.Nsps, test.ref_power_wave, lm_opts, struct());


%% Final TEST 
% 1) Forward RX 
P_noisy_test = forward_rx_chain(test.tx_wave, param_matrix_opt, params);  

[~, ~, off_star_test] = sample_and_align_auto( ...
    P_noisy_test, test.tx_symbols, params.Nsps, test.ref_power_wave, 6, 3, lm_opts);

% 3) Campioni per BER (k = off*)
a1 = struct('fixed_offset', off_star_test, 'do_resample', false);
[yk_test, tx_al_test] = sample_and_align_auto( ...
    P_noisy_test, test.tx_symbols, params.Nsps, test.ref_power_wave, 0, 0, lm_opts, a1);

% 4) Loss L2 
loss_test = compute_loss_L2(P_noisy_test, test.tx_symbols, params.Nsps, test.ref_power_wave, lm_opts, struct());


[yk_t, tx_al_t, off_t, info_align_t] = sample_and_align_auto( ...
    P_noisy_test, test.tx_symbols, params.Nsps, test.ref_power_wave, 4*params.Nsps, 2);

[final_loss, ~] = loss_margin(yk_t, tx_al_t, lm_opts);


[final_BER, thr_t, rank_map_t, info_ber_t] = evaluate_BER_MAP_verbose(yk_t, tx_al_t, true);

fprintf('Final TEST -> off0=%d, off*=%d, loss_margin=%.3e, BER=%.6f\n', ...
    info_align_t.off0, off_t, final_loss, final_BER);

nRuns     = 20;
seed_base = 50000;   
prevRS = RandStream.getGlobalStream();

loss_val_runs = zeros(nRuns,1);
ber_val_runs  = zeros(nRuns,1);
off_val_runs  = zeros(nRuns,1);

for r = 1:nRuns
    % seed diverso → cambia SOLO l'ASE/rumore
    rs = RandStream('mt19937ar','Seed', seed_base + r);
    RandStream.setGlobalStream(rs);

    % forward RX (pesi fissi = param_matrix_opt)
    P_val_r = forward_rx_chain(val.tx_wave, param_matrix_opt, params);

    % offset migliore per questa acquisizione
    [~, ~, off_r] = sample_and_align_auto( ...
        P_val_r, val.tx_symbols, params.Nsps, val.ref_power_wave, 6, 3, lm_opts);
    off_val_runs(r) = off_r;

    % campioni a k = off* per BER
    a1 = struct('fixed_offset', off_r, 'do_resample', false);
    [yk_val_r, tx_val_r] = sample_and_align_auto( ...
        P_val_r, val.tx_symbols, params.Nsps, val.ref_power_wave, 0, 0, lm_opts, a1);

    % loss L2 
    loss_val_runs(r) = compute_loss_L2(P_val_r, val.tx_symbols, params.Nsps, ...
                                       val.ref_power_wave, lm_opts, a1);

    % BER con il tuo MAP gaussiano 
    [ber_val_runs(r), ~, ~] = evaluate_BER_MAP_verbose(yk_val_r, tx_val_r, false);
end

% TEST 
loss_tst_runs = zeros(nRuns,1);
ber_tst_runs  = zeros(nRuns,1);
off_tst_runs  = zeros(nRuns,1);

for r = 1:nRuns
    rs = RandStream('mt19937ar','Seed', seed_base + 10000 + r); 
    RandStream.setGlobalStream(rs);

    P_tst_r = forward_rx_chain(test.tx_wave, param_matrix_opt, params);

    [~, ~, off_r] = sample_and_align_auto( ...
        P_tst_r, test.tx_symbols, params.Nsps, test.ref_power_wave, 6, 3, lm_opts);
    off_tst_runs(r) = off_r;

    a1 = struct('fixed_offset', off_r, 'do_resample', false);
    [yk_tst_r, tx_tst_r] = sample_and_align_auto( ...
        P_tst_r, test.tx_symbols, params.Nsps, test.ref_power_wave, 0, 0, lm_opts, a1);

    loss_tst_runs(r) = compute_loss_L2(P_tst_r, test.tx_symbols, params.Nsps, ...
                                       test.ref_power_wave, lm_opts, a1);

    [ber_tst_runs(r), ~, ~] = evaluate_BER_MAP_verbose(yk_tst_r, tx_tst_r, false);
end


RandStream.setGlobalStream(prevRS);


muL_val = mean(loss_val_runs);  sdL_val = std(loss_val_runs);
muB_val = mean(ber_val_runs);   sdB_val = std(ber_val_runs);
muO_val = mean(off_val_runs);   sdO_val = std(off_val_runs);

muL_tst = mean(loss_tst_runs);  sdL_tst = std(loss_tst_runs);
muB_tst = mean(ber_tst_runs);   sdB_tst = std(ber_tst_runs);
muO_tst = mean(off_tst_runs);   sdO_tst = std(off_tst_runs);

fprintf('\n==== Multi-run (Validation, n=%d) ====\n', nRuns);
fprintf('Loss_L2  = %.3e  ±  %.3e\n', muL_val, sdL_val);
fprintf('BER      = %.3e  ±  %.3e\n', muB_val, sdB_val);
fprintf('off*     = %.2f  ±  %.2f\n', muO_val, sdO_val);

fprintf('\n==== Multi-run (Test, n=%d) ====\n', nRuns);
fprintf('Loss_L2  = %.3e  ±  %.3e\n', muL_tst, sdL_tst);
fprintf('BER      = %.3e  ±  %.3e\n', muB_tst, sdB_tst);
fprintf('off*     = %.2f  ±  %.2f\n\n', muO_tst, sdO_tst);


%%  11) SWEEP OSNR @ 0.1 nm — BER vs OSNR 

OSNR_grid_dB = 28:2:40;   % scegli i punti che ti interessano
nRunsOSNR    = 20;        % media su 20 acquisizioni per ciascuna OSNR

% Salva stato RNG e OSNR per ripristinare a fine sweep
prevRS     = RandStream.getGlobalStream();
prevOSNR_dB = [];
if isfield(params, 'OSNR_dB'), prevOSNR_dB = params.OSNR_dB; end

nO = numel(OSNR_grid_dB);
muL_val_osnr = zeros(nO,1); sdL_val_osnr = zeros(nO,1);
muB_val_osnr = zeros(nO,1); sdB_val_osnr = zeros(nO,1);

muL_tst_osnr = zeros(nO,1); sdL_tst_osnr = zeros(nO,1);
muB_tst_osnr = zeros(nO,1); sdB_tst_osnr = zeros(nO,1);

for io = 1:nO
    OSNRdB = OSNR_grid_dB(io);
    params.OSNR_dB = OSNRdB;   

  
    loss_val_runs = zeros(nRunsOSNR,1);
    ber_val_runs  = zeros(nRunsOSNR,1);

    loss_tst_runs = zeros(nRunsOSNR,1);
    ber_tst_runs  = zeros(nRunsOSNR,1);

    for r = 1:nRunsOSNR
       
        rs = RandStream('mt19937ar','Seed', 90000 + 1000*io + r);
        RandStream.setGlobalStream(rs);

        
        P_val = forward_rx_chain(val.tx_wave, param_matrix_opt, params);

        % offset migliore per questa acquisizione 
        [~, ~, off_val] = sample_and_align_auto( ...
            P_val, val.tx_symbols, params.Nsps, val.ref_power_wave, 6, 3, lm_opts);

        % campioni a k = off* per BER
        a1 = struct('fixed_offset', off_val, 'do_resample', false);
        [yk_val, tx_val] = sample_and_align_auto( ...
            P_val, val.tx_symbols, params.Nsps, val.ref_power_wave, 0, 0, lm_opts, a1);

        % loss L2 e BER 
        loss_val_runs(r) = compute_loss_L2(P_val, val.tx_symbols, params.Nsps, val.ref_power_wave, lm_opts, a1);
        [ber_val_runs(r), ~, ~] = evaluate_BER_MAP_verbose(yk_val, tx_val, false);

        % TEST 
        P_tst = forward_rx_chain(test.tx_wave, param_matrix_opt, params);

        [~, ~, off_tst] = sample_and_align_auto( ...
            P_tst, test.tx_symbols, params.Nsps, test.ref_power_wave, 6, 3, lm_opts);

        a1 = struct('fixed_offset', off_tst, 'do_resample', false);
        [yk_tst, tx_tst] = sample_and_align_auto( ...
            P_tst, test.tx_symbols, params.Nsps, test.ref_power_wave, 0, 0, lm_opts, a1);

        loss_tst_runs(r) = compute_loss_L2(P_tst, test.tx_symbols, params.Nsps, test.ref_power_wave, lm_opts, a1);
        [ber_tst_runs(r), ~, ~] = evaluate_BER_MAP_verbose(yk_tst, tx_tst, false);
    end

    % statistiche per questa OSNR
    muL_val_osnr(io) = mean(loss_val_runs);  sdL_val_osnr(io) = std(loss_val_runs);
    muB_val_osnr(io) = mean(ber_val_runs);   sdB_val_osnr(io) = std(ber_val_runs);

    muL_tst_osnr(io) = mean(loss_tst_runs);  sdL_tst_osnr(io) = std(loss_tst_runs);
    muB_tst_osnr(io) = mean(ber_tst_runs);   sdB_tst_osnr(io) = std(ber_tst_runs);
end

% ripristina RNG e OSNR originali
RandStream.setGlobalStream(prevRS);
if ~isempty(prevOSNR_dB), params.OSNR_dB = prevOSNR_dB; end

% REPORT 
fprintf('\n==== SWEEP OSNR @ 0.1 nm (Validation) ====\n');
for io = 1:nO
    fprintf('OSNR=%2d dB | Loss=%.3e ± %.1e | BER=%.3e ± %.1e\n', ...
        OSNR_grid_dB(io), muL_val_osnr(io), sdL_val_osnr(io), muB_val_osnr(io), sdB_val_osnr(io));
end
fprintf('\n==== SWEEP OSNR @ 0.1 nm (Test) ====\n');
for io = 1:nO
    fprintf('OSNR=%2d dB | Loss=%.3e ± %.1e | BER=%.3e ± %.1e\n', ...
        OSNR_grid_dB(io), muL_tst_osnr(io), sdL_tst_osnr(io), muB_tst_osnr(io), sdB_tst_osnr(io));
end

% PLOT 
figure; hold on; grid on; box on;
errorbar(OSNR_grid_dB, muB_val_osnr, sdB_val_osnr, '-o', 'DisplayName','Validation');
errorbar(OSNR_grid_dB, muB_tst_osnr, sdB_tst_osnr, '-s', 'DisplayName','Test');
set(gca,'YScale','log');
yline(2e-3, '--', 'Pre-FEC 2e-3', 'LabelHorizontalAlignment','left');
xlabel('OSNR @ 0.1 nm (dB)'); ylabel('BER'); 
title('BER vs OSNR (media \pm std su nRuns)');
legend('Location','southwest');

figure; hold on; grid on; box on;
errorbar(OSNR_grid_dB, muL_val_osnr, sdL_val_osnr, '-o', 'DisplayName','Validation');
errorbar(OSNR_grid_dB, muL_tst_osnr, sdL_tst_osnr, '-s', 'DisplayName','Test');
xlabel('OSNR @ 0.1 nm (dB)'); ylabel('Loss L_2');
title('Loss L_2 vs OSNR (media \pm std su nRuns)');
legend('Location','northeast');


figure; plot(history.iter, history.bestFitness, '-o');
xlabel('PSO iter'); ylabel('best loss'); title('PSO convergence');

if use_adam
    figure;
    plot(hist_adam.loss,'-o'); hold on; plot(hist_adam.bestLoss,'-x');
    xlabel('ADAM iter'); ylabel('loss'); legend('cur','best'); title('ADAM refinement');
end

eyewindow = params.Nsps*200;
idx = 1:min(length(P_noisy_test), eyewindow);
figure; plot(test.tvec(idx)*1e9, P_noisy_test(idx));
xlabel('time (ns)'); ylabel('Detected power'); title('Eye snippet (TEST)');


figure; 
subplot(1,2,1); histogram(off_val_runs, 'BinMethod','integers');
title(sprintf('off* VAL (mean=%.2f, std=%.2f)', mean(off_val_runs), std(off_val_runs)));
subplot(1,2,2); histogram(off_tst_runs, 'BinMethod','integers');
title(sprintf('off* TEST (mean=%.2f, std=%.2f)', mean(off_tst_runs), std(off_tst_runs)));





function lossL2 = compute_loss_L2(P_noisy, tx_symbols, Nsps, ref_wave, lm_opts, alignOpts)
    if nargin<6, alignOpts = struct(); end

   
    if isfield(alignOpts,'fixed_offset') && ~isempty(alignOpts.fixed_offset)
        off_star = round(alignOpts.fixed_offset);
    else
        [~,~,off_star] = sample_and_align_auto(P_noisy, tx_symbols, Nsps, ref_wave, 6, 3, lm_opts);
    end

    % k e k+1 senza ricerca
    a1 = struct('fixed_offset', off_star,   'do_resample', false);
    a2 = struct('fixed_offset', off_star+1, 'do_resample', false);
    [yk1, tx1] = sample_and_align_auto(P_noisy, tx_symbols, Nsps, ref_wave, 0, 0, lm_opts, a1);
    [yk2, ~  ] = sample_and_align_auto(P_noisy, tx_symbols, Nsps, ref_wave, 0, 0, lm_opts, a2);

    valid = isfinite(yk1) & isfinite(yk2);
    y1 = yk1(valid); y2 = yk2(valid); txa = tx1(valid);

    [L1,~] = loss_margin(y1, txa, lm_opts);
    [L2,~] = loss_margin(y2, txa, lm_opts);
    lossL2 = 0.5*(L1+L2);
end
