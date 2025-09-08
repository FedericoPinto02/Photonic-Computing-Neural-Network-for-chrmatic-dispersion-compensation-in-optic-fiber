function [BER, thresholds, rank_map, info] = evaluate_BER_MAP_verbose(yk, tx_aligned, verbose)
% Decoder MAP con gaussiane robuste per i 4 livelli
% Soglie = punti di pari densità tra adiacenti. Fallback su "ordered" se classi mancanti.
if nargin<3, verbose=false; end
L=4; info=struct(); 
counts = zeros(L,1); med = nan(L,1); q10 = nan(L,1); q90 = nan(L,1);
for c=0:L-1
    s = yk(tx_aligned==c);
    counts(c+1)=numel(s);
    if ~isempty(s)
        med(c+1)=median(s);
        q10(c+1)=quantile(s,0.10);
        q90(c+1)=quantile(s,0.90);
    end
end
info.counts=counts;
if any(counts==0) || any(isnan(med))
    if verbose, fprintf('[MAP] Fallback: classe mancante.\n'); end
    [BER, thresholds, rank_map, info2] = evaluate_BER_ordered_verbose(yk, tx_aligned, verbose);
    info.fallback=true; info.ordered=info2; return;
end

% ordina per mediana
[med_sorted, ord] = sort(med,'ascend');
rank_map = zeros(L,1);
for r=1:L, rank_map(ord(r))=r; end

% sigma robusta da q90-q10 (≈ 2*1.28155*sigma → sigma ≈ (q90-q10)/2.5631)
sigma = (q90 - q10) / 2.563103; sigma = sigma(ord);
mu = med_sorted;

% soglie tra adiacenti risolvendo uguaglianza di pdf gaussiane
thresholds = zeros(L-1,1);
for r=1:L-1
    thresholds(r) = thr_two_gauss(mu(r),sigma(r), mu(r+1),sigma(r+1));
end
info.mu=mu; info.sigma=sigma; info.thresholds=thresholds;

% classificazione
pred_rank = zeros(numel(yk),1);
for k=1:numel(yk)
    s=yk(k);
    if s<thresholds(1), pred_rank(k)=1;
    elseif s<thresholds(2), pred_rank(k)=2;
    elseif s<thresholds(3), pred_rank(k)=3;
    else, pred_rank(k)=4;
    end
end
true_rank = rank_map(tx_aligned+1);
BER = mean(pred_rank ~= true_rank);

if verbose
    fprintf('[MAP] counts=%s\n', mat2str(counts.'));
    fprintf('[MAP] mu(sorted)=%s\n', mat2str(mu.',3));
    fprintf('[MAP] sigma(sorted)=%s\n', mat2str(sigma.',3));
    fprintf('[MAP] thr=%s | BER=%.6f\n', mat2str(thresholds.',3), BER);
end
end

function t = thr_two_gauss(m1,s1,m2,s2)
% Punto (reale) dove N(m1,s1) e N(m2,s2) hanno stessa pdf (m1<m2).
% Include clamp su sigma e gestione robusta del discriminante.

    % evita sigma=0
    s1 = max(s1, 1e-12);
    s2 = max(s2, 1e-12);

    a = (1/(2*s1^2)) - (1/(2*s2^2));
    b = (-m1/(s1^2)) + (m2/(s2^2));
    c = (m1^2/(2*s1^2)) - (m2^2/(2*s2^2)) + log(s2/s1);

    if abs(a) < 1e-18
        % sigma quasi uguali -> midpoint
        t = (m1 + m2)/2;
        return;
    end

    disc = b^2 - 4*a*c;
    disc = max(disc, 0);  % evita sqrt di numeri negativi per arrotondamenti

    r1 = (-b + sqrt(disc)) / (2*a);
    r2 = (-b - sqrt(disc)) / (2*a);

    % scegli la radice tra m1 e m2 se esiste; altrimenti la più vicina al midpoint
    mid = (m1 + m2)/2;
    candidates = [r1, r2];
    in_between = (candidates > m1) & (candidates < m2);
    if any(in_between)
        t = candidates(find(in_between,1,'first'));
    else
        [~,ix] = min(abs(candidates - mid));
        t = candidates(ix);
    end
end


% ******** ordered BER function utility
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