function P_noisy = forward_rx_chain(tx_wave, param_matrix, params)
% PNN (pre) -> fibra -> filtro ottico 30 GHz -> ASE @ 0.1 nm (sul CAMPO)
%           -> photodetection -> LPF elettrico 16 GHz -> power floor

    Fs = params.Fs;

    % 1) PNN PRE-compensatore
    E_eq = PNN(tx_wave(:).', Fs, params.dt, params.k(:).', param_matrix);

    % 2) Fibra (CD)
    E_rx = fiber_propagate_freqdomain(E_eq, Fs, params.beta2, params.L);

    % 3) Filtro OTTICO 30 GHz
    E_f = opt_bpf_field(E_rx, Fs, 30e9);

    % 4) ASE ottico coerente con OSNR @ 0.1 nm (sul CAMPO)
    if isfield(params,'OSNR_dB') && ~isempty(params.OSNR_dB)
        E_n = add_ASE_OSNR_field(E_f, params.OSNR_dB, Fs, 30e9, 0.1);
    else
        E_n = E_f;
    end

    % 5) Photodetection (|E|^2) + LPF elettrico 16 GHz
    P_det = abs(E_n).^2;   % photodetection (|E|^2)
    P_rx  = rx_lpf_elec(P_det, Fs, 16e9);

    % 6) Robustezza numerica
    P_noisy = apply_power_floor(P_rx, 0.01); % floor = quantile 1%
    P_noisy = P_noisy(:);
end

% ===================== SUBFUNZIONI =====================

function E_rx = fiber_propagate_freqdomain(E_in, Fs, beta2, L)
% fiber_propagate_freqdomain - propaga campo E_in su fibra length L con beta2
% E_rx = fiber_propagate_freqdomain(E_in, Fs, beta2, L)
% beta2 in [s^2/m], L in [m]

N = length(E_in);
Nfft = 2^nextpow2(N);
Epad = [E_in, zeros(1, Nfft-N)]; % zero-pad
Efreq = fftshift(fft(Epad));

% frequency vector (Hz)
f = (-Nfft/2:Nfft/2-1)*(Fs/Nfft);
omega = 2*pi*f;

% transfer function H(omega)
H = exp(-1j*0.5 * beta2 * L .* (omega.^2));

Eout_freq = Efreq .* H;
Eout_time = ifft(ifftshift(Eout_freq));
Eout_time = Eout_time(1:N); % trim padding

E_rx = Eout_time;
end

function PNN = PNN(E_in, Fs, dt, k_vect, param_matrix)
%   Creates delayed replicas of E_in (spacing dt at sampling Fs), applies per-tap
%   amplitudes k_vect and MZI params param_matrix [theta, phi_u, phi_d], then coherently

if size(param_matrix,2) ~= 3
    error('param_matrix must be N x 3: [theta, phi_u, phi_d]');
end

N = size(param_matrix,1);
Nsamples = length(E_in);
PNN = zeros(1, Nsamples);

for i=1:N
    theta_i = param_matrix(i,1);
    phi_u_i = param_matrix(i,2);
    phi_d_i = param_matrix(i,3);
    
    % compute tap MZI matrix (only top-top gain - U(1,1) )
    G11 = (-1j .* exp(-1j*theta_i/2)) .* sin(theta_i/2) .* exp(-1j*phi_u_i);

    % delay (shift)
    shift_s = (i-1)*dt;
    shift_samples = round(shift_s * Fs);
    sig_shifted = zeros(1, Nsamples);
    if shift_samples < Nsamples
        sig_shifted(shift_samples+1:end) = E_in(1:Nsamples-shift_samples);
    end

    % accumulate (apply tap loss k_vect(i) and MZI gain)
    PNN = PNN + k_vect(i) * (G11 .* sig_shifted);
end
end

function P_f = apply_power_floor(P, alpha)
% evita massicci zeri dopo il rumore: clippa a un percentile basso
if nargin<2 || isempty(alpha), alpha = 0.01; end
flo = quantile(P(:), alpha);
P_f = max(P, flo);
end

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

    % Rilimita alla banda ottica (check se utile)
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
