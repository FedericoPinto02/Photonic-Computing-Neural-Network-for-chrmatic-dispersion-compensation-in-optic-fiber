function loss = loss(x, params, tx_symbols, tx_wave)

N = params.N;
if numel(x) ~= 3*N
    error('Parameter vector length must be 3*N');
end

% unpack & normalize into sensible intervals
theta = mod(x(1:N), 2*pi); % theta in [0,2pi)
phi_u = mod(x(N+1:2*N) + pi, 2*pi) - pi; % phases in [-pi,pi]
phi_d = mod(x(2*N+1:3*N) + pi, 2*pi) - pi;

param_matrix = [theta(:), phi_u(:), phi_d(:)];

% forward: PNN (pre-comp)
E_pnn = PNN(tx_wave, params.Fs, params.dt, params.k, param_matrix);

% propagate through fiber (paper places PNN before fiber)
E_rx = fiberPropagate_freqdomain(E_pnn, params.Fs, params.beta2, params.L);

% photodetection
P_det = photodetect(E_rx);

% add noise (AWGN approx via OSNR)
P_noisy = addNoise_OSNR(P_det, params.OSNR_dB, params.Fs);

% sample & align
[yk, tx_sym_aligned] = sample_and_align(P_noisy, tx_symbols, params.Nsps);

% compute loss L2
[loss, ~, ~] = loss_L2(yk, tx_sym_aligned);

% safety penalty
if ~isreal(loss) || isnan(loss) || isinf(loss)
    loss = 1e6;
end
end
