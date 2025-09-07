function [loss_val, diag] = loss(x, params, tx_symbols, tx_wave, opts)
% LOSS — funzione obiettivo per il training della tua PNN
% - Supporta FULL (3N) e PO (N) via params.mode
% - Usa sample_and_align_auto con alignOpts passati da Main
% - Valuta loss_margin (smooth, non si azzera)
%
% INPUT:
%   x           : vettore parametri (N se PO, 3N se FULL)
%   params      : struct con almeno .N e .Nsps (+ il resto che usi nel forward)
%   tx_symbols  : etichette simboli (0..3) per PAM4
%   tx_wave     : forma d'onda di riferimento TX (come già usi nel forward)
%   opts        : struct opzionale
%       .lossOpts     -> opzioni per loss_margin (lm_opts)
%       .alignOpts    -> fixed_offset / do_resample per sample_and_align_auto
%       .refPowerWave -> traccia di riferimento per la xcorr
%       .noiseStream  -> (opz.) RandStream per riproducibilità
%
% OUTPUT:
%   loss_val    : scalare della loss
%   diag        : diagnostica (facoltativa)

    if nargin < 5 || ~isstruct(opts), opts = struct(); end

    % ---------- 1) Modalità e validazione lunghezza x ----------
    N = params.N;
    mode = 'FULL';
    if isfield(params,'mode') && ~isempty(params.mode)
        mode = upper(params.mode);
    end

    switch mode
        case 'PO'   % Phase-Only: ottimizziamo N fasi (phi_u)
            if numel(x) ~= N
                error('loss(PO): attesi N=%d parametri, ricevuti %d.', N, numel(x));
            end
        case 'FULL' % 3N parametri [theta, phi_u, phi_d]
            if numel(x) ~= 3*N
                error('loss(FULL): attesi 3N=%d parametri, ricevuti %d.', 3*N, numel(x));
            end
        otherwise
            error('loss: params.mode sconosciuto: %s', mode);
    end

    % Matrice Nx3 [theta, phi_u, phi_d] coerente con il resto della pipeline
    param_matrix = expand_params(x, N, mode);

    % ---------- 2) Opzioni loss / allineamento ----------
    if isfield(opts,'lossOpts') && ~isempty(opts.lossOpts)
        lm_opts = opts.lossOpts;
    else
        lm_opts = struct('pLow',0.25,'pHigh',0.75, ...
                         'pTailLow',0.05,'pTailHigh',0.95, ...
                         'lambda_range',1.5,'lambda_tail',0.5, ...
                         'margin_pct',0.10,'range_min_pct',0.90);
    end

    alignOpts = struct();
    if isfield(opts,'alignOpts') && ~isempty(opts.alignOpts)
        alignOpts = opts.alignOpts;
    end

    % ---------- 3) (Opz.) Rumore riproducibile ----------
    prevRS = [];
    if isfield(opts,'noiseStream') && ~isempty(opts.noiseStream)
        try
            prevRS = RandStream.getGlobalStream();
            RandStream.setGlobalStream(opts.noiseStream);
        catch
            % se non usi RandStream globale, ignora
        end
    end

P_noisy = forward_rx_chain(tx_wave, param_matrix, params);



    % Ripristina (opz.) lo stream di rumore precedente
    if ~isempty(prevRS)
        try, RandStream.setGlobalStream(prevRS); catch, end
    end

    % ---------- Allineamento & loss L2: due campioni (k e k+1) ----------
coarseSpan = 6;   % usa i tuoi default
K          = 3;

% 1) Decidi l'offset "base" off_star UNA volta
if isfield(alignOpts,'fixed_offset') && ~isempty(alignOpts.fixed_offset)
    % TRAINING: offset fissato dal Main
    off_star = round(alignOpts.fixed_offset);
else
    % VAL/TEST: cerca l'offset con la routine standard
    [~, ~, off_star, ~] = sample_and_align_auto( ...
        P_noisy, tx_symbols, params.Nsps, opts.refPowerWave, ...
        coarseSpan, K, lm_opts, alignOpts);
end

% 2) Campiona SENZA RICERCA a off_star (k) e off_star+1 (k+1)
align_fixed1 = struct('fixed_offset', off_star,   'do_resample', false);
align_fixed2 = struct('fixed_offset', off_star+1, 'do_resample', false);

[yk1, tx1] = sample_and_align_auto( ...
    P_noisy, tx_symbols, params.Nsps, opts.refPowerWave, ...
    0, 0, lm_opts, align_fixed1);   % no search
[yk2, tx2] = sample_and_align_auto( ...
    P_noisy, tx_symbols, params.Nsps, opts.refPowerWave, ...
    0, 0, lm_opts, align_fixed2);   % no search

% 3) Usa solo i campioni validi in comune
valid = isfinite(yk1) & isfinite(yk2);
if ~any(valid)
    % fallback estremo (se fuori range): usa una sola passata
    [loss_val, lm_diag] = loss_margin(yk1(isfinite(yk1)), tx1(isfinite(yk1)), lm_opts);
    if nargout > 1
        diag = struct('mode',mode,'param_matrix',param_matrix,'loss_margin',lm_diag, ...
                      'off_star',off_star,'alignOpts',alignOpts,'note','fallback L2');
    end
    return
end
yk1 = yk1(valid);
yk2 = yk2(valid);
txa = tx1(valid);    % tx1 e tx2 coincidono per indice simbolo

% 4) Calcola le due loss e fai la media (L2)
[L1, d1] = loss_margin(yk1, txa, lm_opts);
[L2, d2] = loss_margin(yk2, txa, lm_opts);
loss_val  = 0.5*(L1 + L2);

% 5) Diagnostica (facoltativa)
if nargout > 1
    diag = struct();
    diag.mode          = mode;
    diag.param_matrix  = param_matrix;
    diag.loss_margin_1 = d1;
    diag.loss_margin_2 = d2;
    diag.off_star      = off_star;
    diag.alignOpts     = alignOpts;
end

end

% ====== SUBFUNZIONI DI SUPPORTO =========================================
function E_noisy = add_ASE_OSNR_field(E_sig, OSNR_dB, Fs, Bopt_Hz, RBW_nm)
% Aggiunge ASE complesso a banda ottica, coerente con OSNR @ RBW_nm (default 0.1 nm).
% E_sig: campo ottico complesso (vettore riga/colonna)
% OSNR_dB: OSNR misurata con RBW_nm (tipicamente 0.1 nm)
% Fs: sample rate del campo [Hz]
% Bopt_Hz: larghezza del filtro ottico [Hz] (es. 30e9)
% RBW_nm: resolution bandwidth dell'OSA in nanometri (default 0.1 nm)

    if nargin < 5 || isempty(RBW_nm), RBW_nm = 0.1; end
    c = 299792458; lambda0 = 1550e-9;
    RBW_Hz = c/(lambda0^2) * (RBW_nm*1e-9);   % ~12.5 GHz @ 1550 nm
    OSNR_lin = 10^(OSNR_dB/10);

    % Potenza media del segnale (|E|^2 ∝ potenza)
    Psig = mean(abs(E_sig).^2);

    % Potenza di rumore riferita a RBW di riferimento
    Pn_ref = Psig / OSNR_lin;        % [W] nella RBW dell'OSA
    S_ASE  = Pn_ref / RBW_Hz;        % densità spettrale (W/Hz)

    % Rumore bianco complesso discreto: var = S_ASE * Fs
    sigma2 = S_ASE * Fs;
    n = sqrt(sigma2/2) * (randn(size(E_sig)) + 1j*randn(size(E_sig)));

    % Somma al campo
    E_noisy = E_sig + n;

    % ASE limitato dal filtro ottico
    if ~isempty(Bopt_Hz) && isfinite(Bopt_Hz) && Bopt_Hz > 0 && Bopt_Hz < Fs/2
        E_noisy = opt_bpf_field(E_noisy, Fs, Bopt_Hz);
    end
end

function E_filt = opt_bpf_field(E_in, Fs, Bopt_Hz)
% Low-pass ideale (campo complesso) con fc = Bopt_Hz (one-sided)
    if Bopt_Hz >= Fs/2
        E_filt = E_in; return;
    end
    N  = 257;                   % ordine FIR (puoi aumentare se vuoi roll-off più ripido)
    fc = Bopt_Hz/(Fs/2);        % normalizzato (Nyquist)
    h  = fir1(N-1, fc, 'low');  % LPF FIR
    Er = filtfilt(h,1, real(E_in));
    Ei = filtfilt(h,1, imag(E_in));
    E_filt = complex(Er, Ei);
end

function y = rx_lpf_elec(x, Fs, f3dB)
% LPF elettrico (Butterworth 5°) con f3dB (es. 16 GHz)
    Wn = min(f3dB/(Fs/2), 0.999);
    [b,a] = butter(5, Wn, 'low');
    y = filtfilt(b,a, x(:));
end
% ========================================================================
