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
ber0 = evaluate_BER_MAP(yk0, tx_al0); 
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
[val_BER, thr_val, rank_map_val, info_ber_val] = evaluate_BER_MAP(yk_val, tx_al_val);

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


[final_BER, thr_t, rank_map_t, info_ber_t] = evaluate_BER_MAP(yk_t, tx_al_t);

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
    [ber_val_runs(r), ~, ~] = evaluate_BER_MAP(yk_val_r, tx_val_r);
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

    [ber_tst_runs(r), ~, ~] = evaluate_BER_MAP(yk_tst_r, tx_tst_r);
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
        [ber_val_runs(r), ~, ~] = evaluate_BER_MAP(yk_val, tx_val);

        % TEST 
        P_tst = forward_rx_chain(test.tx_wave, param_matrix_opt, params);

        [~, ~, off_tst] = sample_and_align_auto( ...
            P_tst, test.tx_symbols, params.Nsps, test.ref_power_wave, 6, 3, lm_opts);

        a1 = struct('fixed_offset', off_tst, 'do_resample', false);
        [yk_tst, tx_tst] = sample_and_align_auto( ...
            P_tst, test.tx_symbols, params.Nsps, test.ref_power_wave, 0, 0, lm_opts, a1);

        loss_tst_runs(r) = compute_loss_L2(P_tst, test.tx_symbols, params.Nsps, test.ref_power_wave, lm_opts, a1);
        [ber_tst_runs(r), ~, ~] = evaluate_BER_MAP(yk_tst, tx_tst);
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


OSNR_grid_dB = 28:2:40;
nRunsOSNR    = 20;

[muB_val_osnr,sdB_val_osnr,muB_tst_osnr,sdB_tst_osnr, ...
 muL_val_osnr,sdL_val_osnr,muL_tst_osnr,sdL_tst_osnr] = ...
    sweep_OSNR_and_eval(param_matrix_opt, params, val, test, lm_opts, OSNR_grid_dB, nRunsOSNR);

%  BER plot 
figure('Name','Figure 1'); hold on; grid on; box on;
errorbar(OSNR_grid_dB, muB_val_osnr, sdB_val_osnr, '-o', 'DisplayName','Validation');
errorbar(OSNR_grid_dB, muB_tst_osnr, sdB_tst_osnr, '-s', 'DisplayName','Test');
set(gca,'YScale','log'); ylim([5e-4 2e-2]); xlim([min(OSNR_grid_dB)-0.5, max(OSNR_grid_dB)+0.5]);
yline(2e-3,'--','Pre-FEC 2e-3','LabelHorizontalAlignment','left');
xlabel('OSNR @ 0.1 nm (dB)'); ylabel('BER');
title('BER vs OSNR (mean \pm std over runs)');
legend('Location','southwest');
set(gcf,'Color','w'); 
saveas(gcf,'Figure1_BER_vs_OSNR.png');

% Loss plot 
figure('Name','Figure 2'); hold on; grid on; box on;
errorbar(OSNR_grid_dB, muL_val_osnr, sdL_val_osnr, '-o', 'DisplayName','Validation');
errorbar(OSNR_grid_dB, muL_tst_osnr, sdL_tst_osnr, '-s', 'DisplayName','Test');
xlabel('OSNR @ 0.1 nm (dB)'); ylabel('Loss L_2');
title('Loss L_2 vs OSNR (mean \pm std over runs)');
legend('Location','northeast'); set(gcf,'Color','w');
saveas(gcf,'Figure2_L2_vs_OSNR.png');


%% PSO convergence
figure('Name','Figure 3'); hold on; grid on; box on;
plot(history.iter, history.bestFitness, '-o','MarkerFaceColor','w');
xlabel('PSO iter'); ylabel('best loss'); title('PSO convergence'); set(gcf,'Color','w');
saveas(gcf,'Figure3_PSO_convergence.png');

%% ADAM refinement
if use_adam
    figure('Name','Figure 4'); hold on; grid on; box on;
    plot(hist_adam.loss,'-o','DisplayName','cur'); 
    plot(hist_adam.bestLoss,'-x','DisplayName','best');
    xlabel('ADAM iter'); ylabel('loss'); title('ADAM refinement'); legend('Location','northeast');
    set(gcf,'Color','w');
    saveas(gcf,'Figure4_ADAM_refinement.png');
end


%% [ Eye diagrams: BTB vs Fiber vs PNN (TEST set)
eyewin_sym = 200;  

params_BTB = params; params_BTB.L = 0;  
E_btb = test.tx_wave(:).';
E_btb_f = opt_bpf_field(E_btb, params.Fs, 30e9);
P_btb = rx_lpf_elec(abs(E_btb_f).^2, params.Fs, 16e9);

% --- Solo fibra (no PNN) ---
E_fib = fiber_propagate_freqdomain(test.tx_wave(:).', params.Fs, params.beta2, params.L);
E_fib = opt_bpf_field(E_fib, params.Fs, 30e9);
P_fib = rx_lpf_elec(abs(E_fib).^2, params.Fs, 16e9);

% --- PNN + fibra (pipeline completa con i pesi ottimi) ---
P_pnn = forward_rx_chain(test.tx_wave, param_matrix_opt, params);

figure('Name','Figure 5'); set(gcf,'Color','w');
subplot(1,3,1); plot_eye(P_btb, params.Nsps, eyewin_sym); title('BTB');
subplot(1,3,2); plot_eye(P_fib, params.Nsps, eyewin_sym); title(sprintf('Fiber (L=%dkm)', round(params.L/1e3)));
subplot(1,3,3); plot_eye(P_pnn, params.Nsps, eyewin_sym); title('PNN + Fiber');
sgtitle('Eye diagrams (TEST)');
saveas(gcf,'Figure5_Eyes_BTB_Fiber_PNN.png');


%% CD-induced power penalty (empirical RF sweep)
fRF = linspace(1e9, 40e9, 60);     
m   = 0.05;                        
[pen_BTB, pen_FIB, pen_PNN] = measure_penalty_RF(fRF, m, params, test.tx_wave, param_matrix_opt);

figure('Name','Figure 6'); hold on; grid on; box on;
plot(fRF/1e9, pen_FIB, '-o', 'DisplayName','Fiber only');
plot(fRF/1e9, pen_PNN, '-s', 'DisplayName','PNN + Fiber');
yline(0,'--','BTB'); xlabel('RF frequency (GHz)'); ylabel('Penalty (dB)');
title(sprintf('CD-induced power penalty (L=%dkm, OSNR=%gdB)', round(params.L/1e3), params.OSNR_dB));
legend('Location','southwest'); set(gcf,'Color','w');
saveas(gcf,'Figure6_CD_penalty.png');

% curva teorica
beta2 = params.beta2; L = params.L; w = 2*pi*fRF;
Pth = -20*log10(abs(cos(0.5*beta2*L.*(w.^2))));
plot(fRF/1e9, Pth, ':', 'DisplayName','Theory (CD only)');
legend show;


function lossL2 = compute_loss_L2(P_noisy, tx_symbols, Nsps, ref_wave, lm_opts, alignOpts)
    if nargin<6, alignOpts = struct(); end

    % determina off_star (fisso se passato, altrimenti cerca)
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

%% secondary help functions for plotting

function [muB_val,sdB_val,muB_tst,sdB_tst,muL_val,sdL_val,muL_tst,sdL_tst] = ...
    sweep_OSNR_and_eval(PM, params, val, test, lm_opts, OSNR_grid_dB, nRuns)


prevRS = RandStream.getGlobalStream();
prevOSNR = params.OSNR_dB; nO = numel(OSNR_grid_dB);

muL_val = zeros(nO,1); sdL_val = zeros(nO,1); muB_val = zeros(nO,1); sdB_val = zeros(nO,1);
muL_tst = zeros(nO,1); sdL_tst = zeros(nO,1); muB_tst = zeros(nO,1); sdB_tst = zeros(nO,1);

for io = 1:nO
    params.OSNR_dB = OSNR_grid_dB(io);
    Lval = zeros(nRuns,1); Bval = zeros(nRuns,1);
    Ltst = zeros(nRuns,1); Btst = zeros(nRuns,1);
    for r = 1:nRuns
        RandStream.setGlobalStream(RandStream('mt19937ar','Seed', 90000 + 1000*io + r));
        % --- Validation
        P_val = forward_rx_chain(val.tx_wave, PM, params);
        [~,~,off_v] = sample_and_align_auto(P_val, val.tx_symbols, params.Nsps, val.ref_power_wave, 6,3,lm_opts);
        a = struct('fixed_offset',off_v,'do_resample',false);
        [yk_v, tx_v] = sample_and_align_auto(P_val, val.tx_symbols, params.Nsps, val.ref_power_wave, 0,0,lm_opts,a);
        Lval(r) = compute_loss_L2(P_val, val.tx_symbols, params.Nsps, val.ref_power_wave, lm_opts, a);
        Bval(r) = evaluate_BER_MAP(yk_v, tx_v);

        % --- Test
        P_t = forward_rx_chain(test.tx_wave, PM, params);
        [~,~,off_t] = sample_and_align_auto(P_t, test.tx_symbols, params.Nsps, test.ref_power_wave, 6,3,lm_opts);
        a = struct('fixed_offset',off_t,'do_resample',false);
        [yk_t, tx_t] = sample_and_align_auto(P_t, test.tx_symbols, params.Nsps, test.ref_power_wave, 0,0,lm_opts,a);
        Ltst(r) = compute_loss_L2(P_t, test.tx_symbols, params.Nsps, test.ref_power_wave, lm_opts, a);
        Btst(r) = evaluate_BER_MAP(yk_t, tx_t);
    end
    muL_val(io)=mean(Lval); sdL_val(io)=std(Lval); muB_val(io)=mean(Bval); sdB_val(io)=std(Bval);
    muL_tst(io)=mean(Ltst); sdL_tst(io)=std(Ltst); muB_tst(io)=mean(Btst); sdB_tst(io)=std(Btst);
end

RandStream.setGlobalStream(prevRS); params.OSNR_dB = prevOSNR;
end


%%
function Eo = fiber_propagate_freqdomain(Ei, Fs, beta2, L)
% Propagazione di sola CD in frequenza: H(w) = exp(-j * 0.5 * beta2 * L * w^2)
% Ei: campo baseband (complesso)  row vector o col -> restituiamo row.
    v = Ei(:).';
    N = numel(v);
    [w, ~] = make_freq_axis(N, Fs);                 % rad/s, ordinata per fftshift
    V  = fftshift(fft(v));                          % -> F(w)
    H  = exp(-1j * 0.5 * beta2 * L .* (w.^2));      % filtro di fase CD
    v2 = ifft(ifftshift(V .* H));                   % <- campo dopo fibra
    Eo = v2;
end

function Eo = opt_bpf_field(Ei, Fs, f3dB)
% Gaussian low-pass sul campo ottico (complesso), -3 dB @ f3dB
    v = Ei(:).';
    N = numel(v);
    [~, f] = make_freq_axis(N, Fs);                 % Hz
    sigma = f3dB / sqrt(log(2));                    % -3 dB @ f3dB
    H = exp(-(f.^2)/(2*sigma^2));
    V = fftshift(fft(v));
    v2 = ifft(ifftshift(V .* H));
    Eo = v2;
end

function Po = rx_lpf_elec(Pi, Fs, f3dB)
% Gaussian low-pass su segnale elettrico reale, -3 dB 
    x = Pi(:).';
    N = numel(x);
    [~, f] = make_freq_axis(N, Fs);                 
    sigma = f3dB / sqrt(log(2));
    H = exp(-(f.^2)/(2*sigma^2));
    X = fftshift(fft(x));
    x2 = ifft(ifftshift(X .* H),'symmetric');
    Po = x2;
end

function A = single_tone_amp(x, Fs, f0)
% Stima l'ampiezza del tono a f0 
    x = x(:).';
    N = numel(x);
    w = hann_local(N).';
    L = 2^nextpow2(N);
    X = fft(x .* w, L);
    f = (0:L-1)*(Fs/L);
    [~,i] = min(abs(f - f0));
    A = 2*abs(X(i))/sum(w);   % ampiezza (picco) normalizzata
end

function plot_eye(P, Nsps, nSym)
% Eye plot semplice su 2 UI (nSym finestre)
    L = Nsps*nSym;
    N = floor(numel(P)/L);
    if N<1, plot(P); return; end
    X = reshape(P(1:N*L), L, N);
    t = (0:L-1)/Nsps;
    plot(t, X, 'LineWidth', 0.7);
    grid on; box on; xlim([0 2]);
    xlabel('time (symbols)'); ylabel('Detected power');
end

function [w, f] = make_freq_axis(N, Fs)
% Restituisce asse angolare 
    df = Fs/N;
    if mod(N,2)==0
        f = (-N/2:N/2-1)*df;    
    else
        f = (-(N-1)/2:(N-1)/2)*df;
    end
    w = 2*pi*f;                 
end

function y = hann_local(N)
    try
        y = hann(N);
    catch
        n = (0:N-1).';
        y = 0.5*(1 - cos(2*pi*n/(N-1)));
    end
end


function [penBTB, penFIB, penPNN] = measure_penalty_RF(fRF, m, params, tx_wave, PM)

Fs = params.Fs;
t  = (0:numel(tx_wave)-1)/Fs;

% Per una misura più stabile setto OSNR molto alto 
params_pure = params; 
if isfield(params,'OSNR_dB'), params_pure.OSNR_dB = 100; end

ampBTB = zeros(size(fRF));
ampFIB = zeros(size(fRF));
ampPNN = zeros(size(fRF));

for k=1:numel(fRF)
    f = fRF(k);

    % Piccola AM sul campo trasmesso
    Ein = tx_wave(:).'.*(1 + m*cos(2*pi*f*t));

    %  BTB: solo filtri ottico + elettrico + fotodiodo (no fibra, no PNN)
    E0 = opt_bpf_field(Ein, Fs, 30e9);
    P0 = rx_lpf_elec(abs(E0).^2, Fs, 16e9);
    ampBTB(k) = single_tone_amp(P0, Fs, f);

    % Solo fibra (no PNN)
    Ef = fiber_propagate_freqdomain(Ein, Fs, params_pure.beta2, params_pure.L);
    Ef = opt_bpf_field(Ef, Fs, 30e9);
    Pf = rx_lpf_elec(abs(Ef).^2, Fs, 16e9);
    ampFIB(k) = single_tone_amp(Pf, Fs, f);

    %  PNN + fibra: passo dalla tua catena ufficiale
    Pp = forward_rx_chain(Ein, PM, params_pure);   % include PNN, fibra, ricevitore
    ampPNN(k) = single_tone_amp(Pp, Fs, f);
end

% Penalty rispetto al BTB (dB): valori negativi = attenuazione
penBTB = zeros(size(fRF));
penFIB = 20*log10(ampFIB./ampBTB + eps);
penPNN = 20*log10(ampPNN./ampBTB + eps);
end
