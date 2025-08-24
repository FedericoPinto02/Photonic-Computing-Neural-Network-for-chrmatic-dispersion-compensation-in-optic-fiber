function P_noisy = addNoise_OSNR(P_signal, OSNR_dB, Fs)
% addNoise_OSNR - aggiunge rumore AWGN a segnale di potenza P_signal (campionato)
% Approssimazione: noisePower = signalPower / 10^(OSNR/10)
Pavg = mean(P_signal);
SNR_linear = 10^(OSNR_dB/10);
noisePower = Pavg / SNR_linear;
% AWGN (real) added to detected power samples
sigma = sqrt(noisePower);
P_noisy = P_signal + sigma * randn(size(P_signal));
end
