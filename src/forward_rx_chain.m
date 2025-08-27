function P_noisy = forward_rx_chain(tx_wave, param_matrix, params)
% FORWARD_RX_CHAIN
% PNN (pre) -> fibra -> filtro ottico 30 GHz -> ASE @ 0.1 nm (sul CAMPO)
%           -> photodetection -> LPF elettrico 16 GHz -> power floor

    Fs = params.Fs;

    % 1) PNN PRE-compensatore
    E_eq = PNN(tx_wave(:).', Fs, params.dt, params.k(:).', param_matrix);

    % 2) Fibra (CD)
    E_rx = fiberPropagate_freqdomain(E_eq, Fs, params.beta2, params.L);

    % 3) Filtro OTTICO 30 GHz
    E_f = opt_bpf_field(E_rx, Fs, 30e9);

    % 4) ASE ottico coerente con OSNR @ 0.1 nm (sul CAMPO)
    if isfield(params,'OSNR_dB') && ~isempty(params.OSNR_dB)
        E_n = add_ASE_OSNR_field(E_f, params.OSNR_dB, Fs, 30e9, 0.1);
    else
        E_n = E_f;
    end

    % 5) Photodetection + LPF elettrico 16 GHz
    P_det = photodetect(E_n);
    P_rx  = rx_lpf_elec(P_det, Fs, 16e9);

    % 6) Robustezza numerica
    P_noisy = apply_power_floor(P_rx, 0.01);
    P_noisy = P_noisy(:);
end

% ===================== SUBFUNZIONI =====================

function E_noisy = add_ASE_OSNR_field(E_sig, OSNR_dB, Fs, Bopt_Hz, RBW_nm)
% Aggiunge ASE complesso coerente con OSNR @ RBW_nm (default 0.1 nm) sul CAMPO.
    if nargin < 5 || isempty(RBW_nm), RBW_nm = 0.1; end
    c = 299792458; lambda0 = 1550e-9;
    RBW_Hz  = c/(lambda0^2) * (RBW_nm*1e-9);   % ~12.5 GHz @ 1550 nm
    OSNRlin = 10^(OSNR_dB/10);

    % Potenza media del segnale (|E|^2 ∝ potenza)
    Psig  = mean(abs(E_sig).^2);
    PnRef = Psig / OSNRlin;      % potenza rumore nella RBW di riferimento
    S_ASE = PnRef / RBW_Hz;      % densità spettrale [W/Hz]

    % Rumore bianco complesso discreto: var = S_ASE * Fs
    sigma2 = S_ASE * Fs;
    n = sqrt(sigma2/2) * (randn(size(E_sig)) + 1j*randn(size(E_sig)));

    E_noisy = E_sig + n;

    % Ri-limita alla banda ottica (se utile)
    if ~isempty(Bopt_Hz) && isfinite(Bopt_Hz) && Bopt_Hz > 0 && Bopt_Hz < Fs/2
        E_noisy = opt_bpf_field(E_noisy, Fs, Bopt_Hz);
    end
end

function E_filt = opt_bpf_field(E_in, Fs, Bopt_Hz)
% Low-pass sul CAMPO con fc = Bopt_Hz (one-sided).
% Usa fir1/filtfilt se ci sono; fallback a finestra rettangolare.

    if Bopt_Hz >= Fs/2
        E_filt = E_in; return;
    end

    try
        % FIR sinc finestrato + zero-phase
        N  = 257;
        fc = Bopt_Hz/(Fs/2);           % normalizzato (0..1)
        h  = fir1(N-1, fc, 'low');
        Er = filtfilt(h,1, real(E_in));
        Ei = filtfilt(h,1, imag(E_in));
        E_filt = complex(Er, Ei);
    catch
        % Fallback senza Signal Processing Toolbox: moving-average LPF
        L = max(3, 2*round(Fs/Bopt_Hz) + 1);   % finestra ~ Fs/fc
        w = ones(L,1)/L;
        Er = conv(real(E_in), w, 'same');
        Ei = conv(imag(E_in), w, 'same');
        E_filt = complex(Er, Ei);
    end
end

function y = rx_lpf_elec(x, Fs, f3dB)
% LPF elettrico con f3dB (idealmente Butterworth 5°).
    try
        Wn = min(f3dB/(Fs/2), 0.999);
        [b,a] = butter(5, Wn, 'low');
        y = filtfilt(b,a, x(:));
    catch
        % Fallback FIR se butter/filtfilt non esistono
        N  = 257;
        fc = min(f3dB/(Fs/2), 0.999);
        h  = fir1(N-1, fc, 'low');
        y  = filtfilt(h,1, x(:));
    end
end
