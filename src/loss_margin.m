function [loss, diag] = loss_margin(yk, labels, opts)
% loss con margine tra classi PAM4, penalizza code e collasso
%
% USO:
%   [loss, diag] = loss_margin(yk, labels)
%   [loss, diag] = loss_margin(yk, labels, opts)
%
% opts (tutti opzionali)
%   .pLow           (default 0.25)   % quartile basso per parte centrale
%   .pHigh          (default 0.75)   % quartile alto per parte centrale
%   .pTailLow       (default 0.05)   % percentile basso per code
%   .pTailHigh      (default 0.95)   % percentile alto per code
%   .margin_pct     (default 0.10)   % margine richiesto = margin_pct * IQR(10–90)
%   .range_min_pct  (default 0.90)   % dinamica min tra mediane = range_min_pct * IQR
%   .margin         (default [])     % se dato, usa valore assoluto (override margin_pct)
%   .range_min      (default [])     % se dato, usa valore assoluto (override range_min_pct)
%   .lambda_range   (default 1.5)    % peso penale dinamica
%   .lambda_tail    (default 0.5)    % peso penale code/overlap
%
% Ritorna:
%   loss  : scalare >= 0 da minimizzare (mai "piatto" a zero)
%   diag  : struct con diagnostica (mediane ordinate, gaps, ecc.)

    if nargin < 3, opts = struct(); end
    yk = yk(:);
    labels = labels(:);
    numLevels = 4;

    % --- default opzioni ---
    pLow          = getf(opts,'pLow',          0.25);
    pHigh         = getf(opts,'pHigh',         0.75);
    pTailLow      = getf(opts,'pTailLow',      0.05);
    pTailHigh     = getf(opts,'pTailHigh',     0.95);
    margin_pct    = getf(opts,'margin_pct',    0.10);
    range_min_pct = getf(opts,'range_min_pct', 0.90);
    lambda_range  = getf(opts,'lambda_range',  1.5);
    lambda_tail   = getf(opts,'lambda_tail',   0.5);

    % --- statistiche base per classe (mediane per ordering) ---
    med = nan(numLevels,1);
    counts = zeros(numLevels,1);
    for c = 0:numLevels-1
        s = yk(labels == c);
        s = s(isfinite(s));
        counts(c+1) = numel(s);
        if ~isempty(s), med(c+1) = median(s); end
    end

    % fallback se manca qualche classe
    if any(counts==0) || any(isnan(med))
        loss = 1e3;
        diag = struct('counts',counts,'fallback',true);
        return;
    end

    % --- ordina classi per mediana (rango fisico 1..4) ---
    [med_sorted, ord] = sort(med,'ascend');

    % --- scala globale: IQR(10–90) per rendere i target adimensionali ---
    iqr_global = quantile(yk,0.90) - quantile(yk,0.10);
    if iqr_global <= 0
        iqr_global = max(eps, std(yk));
    end

    % margine richiesto e dinamica minima (assoluti o percentuali)
    margin    = getf(opts,'margin',    margin_pct    * iqr_global);
    range_min = getf(opts,'range_min', range_min_pct * iqr_global);

    % --- quartili CENTRALI per i gap (pLow–pHigh), per ciascuna classe ordinata ---
    qLc = nan(numLevels,1); qHc = nan(numLevels,1);
    q10 = nan(numLevels,1); q90 = nan(numLevels,1);  % per sigma robusta
    for i = 1:numLevels
        cls = (labels == (ord(i)-1));
        si  = yk(cls); si = si(isfinite(si));
        qLc(i) = quantile(si, pLow);
        qHc(i) = quantile(si, pHigh);
        q10(i) = quantile(si, 0.10);
        q90(i) = quantile(si, 0.90);
    end

    % ====== PATCH 1: HINGE 'SOFT' sui gap centrali (mai zero) ======
    % softplus(x) = tau*log(1+exp(x/tau)); scegli tau in scala all'IQR globale
    tau = max(1e-12, 0.02 * iqr_global);
    hinge_c = zeros(numLevels-1,1);
    for r = 1:numLevels-1
        gap_r = qLc(r+1) - qHc(r);           % gap tra zone centrali adiacenti
        z = (margin - gap_r) / tau;          % positivo se gap < margin
        hinge_c(r) = tau * log1p(exp(z));    % >0 anche con gap_r > margin (tende a ~0)
    end
    loss_overlap_central = sum(hinge_c);

    % ====== PATCH 2: 'TAIL' come overlap gaussiano stimato (mai zero) ======
    % σ robusta da q90-q10: σ ≈ (q90-q10)/2.563103
    sigma = (q90 - q10) / 2.563103;
    sigma(~isfinite(sigma)) = 0;
    sigma = max(sigma, 1e-12);  % evita divisioni per zero

    overlap_tail = 0;
    for r = 1:numLevels-1
        dmu  = med_sorted(r+1) - med_sorted(r);
        seff = sqrt( sigma(r)^2 + sigma(r+1)^2 );
        seff = max(seff, 1e-12);
        % Probabilità di errore approssimata per due gaussiane ~ Q(dmu/(2*seff))
        % Q(x) = 0.5*erfc(x/sqrt(2))
        overlap_tail = overlap_tail + 0.5 * erfc( dmu / (2*seff*sqrt(2)) );
    end

    % --- penale di DINAMICA tra mediane (evita collasso di tutte le classi) ---
    dyn_range    = med_sorted(end) - med_sorted(1);
    range_penalty = max(0, range_min - dyn_range);

    % --- loss totale (sempre >0) ---
    loss = loss_overlap_central + lambda_tail * overlap_tail + lambda_range * range_penalty;

    % --- diagnostica ---
    diag = struct();
    diag.counts        = counts;
    diag.med_sorted    = med_sorted;
    diag.order         = ord;           % mapping: rango -> label originale+1
    diag.qL_central    = qLc;
    diag.qH_central    = qHc;
    diag.hinge_central = hinge_c;
    diag.q10           = q10;
    diag.q90           = q90;
    diag.sigma_robust  = sigma;
    diag.overlap_tail  = overlap_tail;  % misura probabilistica
    diag.dyn_range     = dyn_range;
    diag.range_min     = range_min;
    diag.margin        = margin;
    diag.lambda_range  = lambda_range;
    diag.lambda_tail   = lambda_tail;
    diag.fallback      = false;
end

% --- helper: get field or default ---
function v = getf(s, f, d)
    if isstruct(s) && isfield(s,f) && ~isempty(s.(f))
        v = s.(f);
    else
        v = d;
    end
end
