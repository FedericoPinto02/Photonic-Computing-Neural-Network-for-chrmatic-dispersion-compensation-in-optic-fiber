function [loss_val, diag] = loss(x, params, tx_symbols, tx_wave, opts)

% LOSS — funzione obiettivo per il training della PNN
% INPUT:
%   x           : vettore parametri (N se PO, 3N se FULL)
%   params      : struct con almeno .N e .Nsps (+ il resto che usi nel forward)
%   tx_symbols  : etichette simboli (0..3) per PAM4
%   tx_wave     : forma d'onda di riferimento TX (come già usi nel forward)
%   opts        : struct opzionale
%       .lossOpts     -> opzioni per loss_margin (lm_opts)
%       .alignOpts    -> fixed_offset / do_resample per sample_and_align_auto
%       .refPowerWave -> traccia di riferimento per la xcorr
%       .noiseStream  -> (opz.) RandStream per riproducibilità
%
% OUTPUT:
%   loss_val    : scalare della loss
%   diag        : diagnostica (facoltativa)

    if nargin < 5 || ~isstruct(opts), opts = struct(); end

    %  1) Modalità e validazione lunghezza x 
    N = params.N;
    mode = 'FULL';
    if isfield(params,'mode') && ~isempty(params.mode)
        mode = upper(params.mode);
    end

    switch mode
        case 'PO'   % Phase-Only: ottimizziamo N fasi (phi_u)
            if numel(x) ~= N
                error('loss(PO): attesi N=%d parametri, ricevuti %d.', N, numel(x));
            end
        case 'FULL' % 3N parametri [theta, phi_u, phi_d]
            if numel(x) ~= 3*N
                error('loss(FULL): attesi 3N=%d parametri, ricevuti %d.', 3*N, numel(x));
            end
        otherwise
            error('loss: params.mode sconosciuto: %s', mode);
    end

    % Matrice Nx3 [theta, phi_u, phi_d] coerente con il resto della pipeline
    param_matrix = expand_params(x, N, mode);

    % 2) Opzioni loss / allineamento
    if isfield(opts,'lossOpts') && ~isempty(opts.lossOpts)
        lm_opts = opts.lossOpts;
    else
        lm_opts = struct('pLow',0.25,'pHigh',0.75, ...
                         'pTailLow',0.05,'pTailHigh',0.95, ...
                         'lambda_range',1.5,'lambda_tail',0.5, ...
                         'margin_pct',0.10,'range_min_pct',0.90);
    end

    alignOpts = struct();
    if isfield(opts,'alignOpts') && ~isempty(opts.alignOpts)
        alignOpts = opts.alignOpts;
    end

    % 3)  Rumore riproducibile
    prevRS = [];
    if isfield(opts,'noiseStream') && ~isempty(opts.noiseStream)
        try
            prevRS = RandStream.getGlobalStream();
            RandStream.setGlobalStream(opts.noiseStream);
        catch
            
        end
    end

P_noisy = forward_rx_chain(tx_wave, param_matrix, params);


    if ~isempty(prevRS)
        try RandStream.setGlobalStream(prevRS); catch, end
    end

   
coarseSpan = 6; 
K = 3;

% 1) Decidi l'offset "base" off_star 
if isfield(alignOpts,'fixed_offset') && ~isempty(alignOpts.fixed_offset)
    % TRAINING: offset fissato dal Main
    off_star = round(alignOpts.fixed_offset);
else
    % VAL/TEST: cerca l'offset con la routine standard
    [~, ~, off_star, ~] = sample_and_align_auto( ...
        P_noisy, tx_symbols, params.Nsps, opts.refPowerWave, ...
        coarseSpan, K, lm_opts, alignOpts);
end

% 2) Campiona SENZA RICERCA 
align_fixed1 = struct('fixed_offset', off_star,   'do_resample', false);
align_fixed2 = struct('fixed_offset', off_star+1, 'do_resample', false);

[yk1, tx1] = sample_and_align_auto( ...
    P_noisy, tx_symbols, params.Nsps, opts.refPowerWave, ...
    0, 0, lm_opts, align_fixed1);   
[yk2, tx2] = sample_and_align_auto( ...
    P_noisy, tx_symbols, params.Nsps, opts.refPowerWave, ...
    0, 0, lm_opts, align_fixed2);   

% 3) Usa solo i campioni validi in comune
valid = isfinite(yk1) & isfinite(yk2);
if ~any(valid)
   
    [loss_val, lm_diag] = loss_margin(yk1(isfinite(yk1)), tx1(isfinite(yk1)), lm_opts);
    if nargout > 1
        diag = struct('mode',mode,'param_matrix',param_matrix,'loss_margin',lm_diag, ...
                      'off_star',off_star,'alignOpts',alignOpts,'note','fallback L2');
    end
    return
end
yk1 = yk1(valid);
yk2 = yk2(valid);
txa = tx1(valid);  

% 4) Calcola le due loss e fai la media (L2)
[L1, d1] = loss_margin(yk1, txa, lm_opts);
[L2, d2] = loss_margin(yk2, txa, lm_opts);
loss_val  = 0.5*(L1 + L2);

end


function PM = expand_params(x, N, mode)
% Restituisce una matrice Nx3 [theta, phi_u, phi_d]
%   mode='PO'   : theta=pi, phi_u = x(:), phi_d=0
%   mode='FULL' : x è lungo 3N e viene reshaped in Nx3

    if nargin<3 || isempty(mode), mode = 'FULL'; end
    mode = upper(mode);

    switch mode
        case 'PO'
            if numel(x) ~= N
                error('expand_params(PO): attesi N=%d parametri, ricevuti %d.', N, numel(x));
            end
            theta = (pi/2)*ones(N,1);
            phi_u = x(:);
            phi_d = zeros(N,1);
            PM = [theta, phi_u, phi_d];

        case 'FULL'
            if numel(x) ~= 3*N
                error('expand_params(FULL): attesi 3N=%d parametri, ricevuti %d.', 3*N, numel(x));
            end
            PM = reshape(x, [3,N])';

        otherwise
            error('expand_params: mode sconosciuta: %s', mode);
    end
end
