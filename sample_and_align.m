function [yk, tx_symbols_aligned] = sample_and_align(P_noisy, tx_symbols, Nsps)
% sample_and_align - trova offset tramite xcorr e campiona al centro di simbolo
% P_noisy: 1xNsamples ; tx_symbols: Nsym x 1 (integers)
Nsym = length(tx_symbols);
Nsamples = length(P_noisy);

% Build reference waveform from tx_symbols (rect pulses)
ref = zeros(1, Nsamples);
for k=1:Nsym
    idx = (k-1)*Nsps + (1:Nsps);
    ref(idx) = tx_symbols(k); % using integer pattern as template is OK for alignment
end

% cross-correlation
[c, lags] = xcorr(P_noisy, ref);
[~,I] = max(abs(c));
lag = lags(I);
% compute offset: if lag>0 P leads ref, adjust
offset = -lag;
% compute sampling indices (center sample)
center = round(Nsps/2);
yk = zeros(Nsym,1);
for k=1:Nsym
    samp_idx = (k-1)*Nsps + center + offset;
    if samp_idx < 1 || samp_idx > Nsamples
        yk(k) = NaN;
    else
        yk(k) = P_noisy(samp_idx);
    end
end

% remove NaN if any (edge effects)
valid = ~isnan(yk);
yk = yk(valid);
tx_symbols_aligned = tx_symbols(valid);
end
