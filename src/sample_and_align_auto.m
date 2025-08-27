function [yk_best, tx_al_best, offset_best, info] = sample_and_align_auto( ...
    P_noisy, tx_symbols, Nsps, ref_power_wave, coarseSpan, K, lossOpts, alignOpts)
% SAMPLE_AND_ALIGN_AUTO
% Ricerca (o applica) l'offset di campionamento e restituisce i campioni simbolo-centrati.
%
% Nuovo argomento opzionale:
%   alignOpts.fixed_offset  -> se presente, usa questo offset e SALTA ogni ricerca
%   alignOpts.do_resample   -> default true; se false, niente ricerca coarse/fine
%
% Retro-compatibile: se alignOpts non è passato, comportamento identico a prima.

    % ---------- default opzionali ----------
    if nargin<5 || isempty(coarseSpan), coarseSpan = 2*Nsps; end
    if nargin<6 || isempty(K), K = 2; end
    if nargin<7 || isempty(lossOpts)
        lossOpts = struct('pLow',0.25,'pHigh',0.75, ...
                          'pTailLow',0.05,'pTailHigh',0.95, ...
                          'lambda_range',1.5,'lambda_tail',0.5, ...
                          'margin_pct',0.10,'range_min_pct',0.90);
    end
    if nargin<8 || isempty(alignOpts), alignOpts = struct(); end
    if ~isfield(alignOpts,'do_resample'), alignOpts.do_resample = true; end

    % Assicura colonne
    P_noisy        = P_noisy(:);
    tx_symbols     = tx_symbols(:);
    ref_power_wave = ref_power_wave(:);

    % ---------- BYPASS: offset fisso ----------
    if isfield(alignOpts,'fixed_offset') && ~isempty(alignOpts.fixed_offset)
        offset_best = round(alignOpts.fixed_offset);
        [yk_best, tx_al_best] = sample_at_offset(P_noisy, tx_symbols, Nsps, offset_best);

        % Diagnostica minima (niente ricerca fatta)
        info = struct();
        info.mode            = 'fixed';
        info.off0            = NaN;
        info.off_grid_coarse = [];
        info.off_grid_fine   = [];
        % Calcola (opzionale) la loss a scopo diagnostico:
        valid = ~isnan(yk_best);
        if any(valid)
            [bestLoss,~] = loss_margin(yk_best(valid), tx_al_best(valid), lossOpts);
        else
            bestLoss = Inf;
        end
        info.bestLoss        = bestLoss;
        info.Nvalid          = sum(valid);
        return;
    end

    % ---------- 1) Stima iniziale offset via cross-correlazione ----------
    p = double(P_noisy) - mean(P_noisy);
    r = double(ref_power_wave) - mean(ref_power_wave);

    [c, lags] = xcorr(p, r);
    [~, idx]  = max(c);
    off0      = round(lags(idx));

    % ---------- Caso: niente resample richiesto ----------
    if ~alignOpts.do_resample
        offset_best = off0;
        [yk_best, tx_al_best] = sample_at_offset(P_noisy, tx_symbols, Nsps, offset_best);

        info = struct();
        info.mode            = 'xcorr_only';
        info.off0            = off0;
        info.off_grid_coarse = [];
        info.off_grid_fine   = [];
        valid = ~isnan(yk_best);
        if any(valid)
            [bestLoss,~] = loss_margin(yk_best(valid), tx_al_best(valid), lossOpts);
        else
            bestLoss = Inf;
        end
        info.bestLoss        = bestLoss;
        info.Nvalid          = sum(valid);
        return;
    end

    % ---------- 2) Ricerca grossolana ±coarseSpan attorno a off0 ----------
    offs_coarse = off0 + (-coarseSpan:coarseSpan);
    bestLoss    = inf;
    offset_best = off0;
    yk_best     = [];
    tx_al_best  = [];

    for o = offs_coarse
        [yk_o, tx_o] = sample_at_offset(P_noisy, tx_symbols, Nsps, o);
        valid = ~isnan(yk_o);
        if ~any(valid), continue; end
        [L,~] = loss_margin(yk_o(valid), tx_o(valid), lossOpts);
        if L < bestLoss
            bestLoss    = L;
            offset_best = o;
            yk_best     = yk_o(valid);
            tx_al_best  = tx_o(valid);
        end
    end

    % ---------- 3) Rifinitura fine ±K attorno a offset_best ----------
    offs_fine = offset_best + (-K:K);
    for o = offs_fine
        [yk_o, tx_o] = sample_at_offset(P_noisy, tx_symbols, Nsps, o);
        valid = ~isnan(yk_o);
        if ~any(valid), continue; end
        [L,~] = loss_margin(yk_o(valid), tx_o(valid), lossOpts);
        if L < bestLoss
            bestLoss    = L;
            offset_best = o;
            yk_best     = yk_o(valid);
            tx_al_best  = tx_o(valid);
        end
    end

    % ---------- Diagnostica ----------
    info = struct();
    info.mode            = 'search';
    info.off0            = off0;
    info.off_grid_coarse = offs_coarse;
    info.off_grid_fine   = offs_fine;
    info.bestLoss        = bestLoss;
    info.Nvalid          = numel(yk_best);
end

% ------ helper locale: campionamento a offset fisso ------
function [yk, tx_al] = sample_at_offset(P_noisy, tx_symbols, Nsps, offset)
    Nsym     = numel(tx_symbols);
    Nsamples = numel(P_noisy);
    center   = round(Nsps/2);
    yk = nan(Nsym,1);
    for k = 1:Nsym
        idx = (k-1)*Nsps + center + offset;
        if idx>=1 && idx<=Nsamples
            yk(k) = P_noisy(idx);
        end
    end
    tx_al = tx_symbols;
end
