function [BER, thresholds, rank_map, info] = evaluate_BER_ordered_verbose(yk, tx_symbols_aligned, verbose)
% EVALUATE_BER_ORDERED_VERBOSE - decoder con log di mediane/threshold/fallback
% USO:
%   [BER, thr, rank_map, info] = evaluate_BER_ordered_verbose(yk, tx_al, true)

if nargin<3, verbose = false; end
numLevels = 4;
info = struct();

% conteggi per classe
counts = zeros(numLevels,1);
for n=0:numLevels-1
    counts(n+1) = sum(tx_symbols_aligned==n);
end
info.counts = counts;

% mediane per classe
med = nan(numLevels,1);
for n=0:numLevels-1
    s = yk(tx_symbols_aligned==n);
    if ~isempty(s), med(n+1) = median(s); end
end
info.medians = med;

fallback = any(isnan(med));
info.fallback = fallback;

if fallback
    % fallback robusto: percentili globali 25/50/75
    q = quantile(yk, [0.25 0.5 0.75]);
    thresholds = q(:);
    rank_map = (0:3)'+1;
    pred_rank = zeros(size(yk));
    for k=1:numel(yk)
        s = yk(k);
        if s < thresholds(1)
            pred_rank(k) = 1;
        elseif s < thresholds(2)
            pred_rank(k) = 2;
        elseif s < thresholds(3)
            pred_rank(k) = 3;
        else
            pred_rank(k) = 4;
        end
    end
    true_rank = rank_map(tx_symbols_aligned+1);
    BER = mean(pred_rank ~= true_rank);

    if verbose
        fprintf('[BER-VERBOSE] Fallback attivo (classe mancante). Counts=%s, thr=%s\n', ...
            mat2str(counts.'), mat2str(thresholds.',3));
    end
    return;
end

% ordina le mediane → mappa classe->rango
[med_sorted, ord] = sort(med,'ascend');
rank_map = zeros(numLevels,1);
for r=1:numLevels
    rank_map(ord(r)) = r;
end

% soglie a metà tra mediane ordinate
thresholds = zeros(numLevels-1,1);
for r=1:numLevels-1
    thresholds(r) = 0.5*(med_sorted(r)+med_sorted(r+1));
end
info.thresholds = thresholds;

% quantizzazione
pred_rank = zeros(size(yk));
for k=1:numel(yk)
    s = yk(k);
    if s < thresholds(1)
        pred_rank(k) = 1;
    elseif s < thresholds(2)
        pred_rank(k) = 2;
    elseif s < thresholds(3)
        pred_rank(k) = 3;
    else
        pred_rank(k) = 4;
    end
end

true_rank = rank_map(tx_symbols_aligned+1);
BER = mean(pred_rank ~= true_rank);

if verbose
    fprintf('[BER-VERBOSE] Counts=%s | Med=%s | Thr=%s | BER=%.6f\n', ...
        mat2str(counts.'), mat2str(med.',3), mat2str(thresholds.',3), BER);
end
end
