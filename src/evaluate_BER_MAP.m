function [BER, thresholds, rank_map, info] = evaluate_BER_MAP(yk, tx_aligned)
% Decoder MAP per i 4 livelli


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
    %  inline ordered fallback (
    
    q = quantile(yk, [0.25 0.50 0.75]);
    thresholds = q(:);
    rank_map = (1:4)';              
   
    pred_rank = zeros(numel(yk),1);
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
    true_rank = rank_map(tx_aligned+1);
    BER = mean(pred_rank ~= true_rank);
    info.fallback = true;
    info.ordered  = struct('method','inline-ordered','thresholds',thresholds);
    return;
end


% ordina per mediana
[med_sorted, ord] = sort(med,'ascend');
rank_map = zeros(L,1);
for r=1:L, rank_map(ord(r))=r; end

% sigma 
sigma = (q90 - q10) / 2.563103; sigma = sigma(ord);
mu = med_sorted;

% soglie 
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


end

function t = thr_two_gauss(m1,s1,m2,s2)

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



