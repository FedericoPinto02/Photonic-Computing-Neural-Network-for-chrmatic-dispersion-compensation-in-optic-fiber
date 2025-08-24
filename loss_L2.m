function [loss, EL, ER] = loss_L2(yk, tx_symbols_aligned)
% loss_L2 - approssimazione della loss L2 del paper usando percentili delle code
% tx_symbols_aligned: integers 0..3 (PAM4)
levels = [-3 -1 1 3];
numLevels = numel(levels);
EL = zeros(numLevels,1);
ER = zeros(numLevels,1);

% map transmitted integer labels to level index
for n=0:numLevels-1
    idx = (tx_symbols_aligned == n);
    samples_n = yk(idx);
    if isempty(samples_n)
        EL(n+1) = NaN; ER(n+1) = NaN;
        continue;
    end
    % 10th and 90th percentiles as left/right edges
    EL(n+1) = prctile(samples_n, 10);
    ER(n+1) = prctile(samples_n, 90);
end

% compute overlap-based loss: for adjacent pairs (n,n+1)
loss = 0;
for n=1:(numLevels-1)
    if isnan(ER(n)) || isnan(EL(n+1))
        continue;
    end
    overlap = ER(n) - EL(n+1); % if negative -> separated
    if overlap > 0
        loss = loss + overlap^2;
    end
end

% Also add a small regularization to keep numbers stable
loss = loss + 1e-12;
end
