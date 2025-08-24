function PNN = PNN(E_in, Fs, dt, k_vect, param_matrix)


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
    % compute tap MZI matrix (page45)
    U2 = MZI(theta_i, phi_u_i, phi_d_i); % keep global phase
    G11 = U2(1,1); % top->top complex gain for this tap

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