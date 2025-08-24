function BER = evaluate_BER(yk, tx_symbols_aligned)
% evaluate_BER - valuta BER su PAM4 usando threshold midpoints from percentiles
[~, EL, ER] = loss_L2(yk, tx_symbols_aligned); % reuse percentiles
% if any NaN, fallback simple thresholds based on histogram
numLevels = 4;
T = zeros(numLevels-1,1);
for n=1:(numLevels-1)
    T(n) = 0.5*(ER(n) + EL(n+1));
end

% quantize yk to symbols 0..3 using thresholds
decoded = zeros(size(yk));
for k=1:length(yk)
    s = yk(k);
    if s < T(1)
        decoded(k) = 0;
    elseif s < T(2)
        decoded(k) = 1;
    elseif s < T(3)
        decoded(k) = 2;
    else
        decoded(k) = 3;
    end
end

% BER (symbol error rate)
BER = sum(decoded ~= tx_symbols_aligned) / length(tx_symbols_aligned);
end
