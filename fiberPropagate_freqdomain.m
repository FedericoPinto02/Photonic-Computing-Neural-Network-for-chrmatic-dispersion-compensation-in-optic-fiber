function E_rx = fiberPropagate_freqdomain(E_in, Fs, beta2, L)
% fiberPropagate_freqdomain - propaga campo E_in su fibra length L con beta2
% E_rx = fiberPropagate_freqdomain(E_in, Fs, beta2, L)
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
