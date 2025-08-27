function [tx_symbols, tx_wave, tvec, P_levels] = genPAM4_prbs(Nsym, params, varargin)
% genPAM4_prbs (IM/DD) - genera PAM4 in POTENZA (campo = sqrt(potenza))
% USO:
%   [tx_symbols, tx_wave, tvec] = genPAM4_prbs(Nsym, params)
%   ... opzioni: 'prbsOrder', 11, 'seed', 2025, 'P_levels', [0.1 0.4 0.7 1.0], 'normalizePower', true
%
% OUTPUT:
%   tx_symbols : vettore 0..3
%   tx_wave    : campo complesso (reale) oversampled con ampiezza = sqrt(P_livello)
%   tvec       : time vector
%   P_levels   : livelli di POTENZA usati (dopo eventuale normalizzazione)

p = inputParser;
addOptional(p,'prbsOrder',11);
addOptional(p,'seed',[]);
addOptional(p,'P_levels',[0.1 0.4 0.7 1.0]);    % 4 livelli di potenza (monotoni)
addOptional(p,'normalizePower',true);           % normalizza potenza media a 1
parse(p,varargin{:});
prbsOrder = p.Results.prbsOrder;
seed = p.Results.seed;
P_levels = p.Results.P_levels(:).';
normalizePower = p.Results.normalizePower;

if ~isempty(seed); rng(seed); end
if any(diff(P_levels) <= 0)
    error('P_levels deve essere strettamente crescente, es. [0.1 0.4 0.7 1.0]');
end

Fs   = params.Fs;
Nsps = params.Nsps;
Tb   = 1/params.Baud;
Nsamples = Nsym * Nsps;
tvec = (0:Nsamples-1)/Fs;

% --- PRBS -> simboli PAM4 (0..3)
bits_needed = Nsym * 2;
prbs_bits = prbs_lfsr(prbsOrder, bits_needed);
tx_symbols = zeros(Nsym,1);
for k=1:Nsym
    b1 = prbs_bits(2*k-1); b0 = prbs_bits(2*k);
    tx_symbols(k) = b1*2 + b0; % 0..3
end

% --- Mappa ai 4 LIVELLI DI POTENZA (IM/DD): campo = sqrt(potenza)
if normalizePower
    P_levels = P_levels / mean(P_levels); % potenza media ~1
end
E_levels = sqrt(P_levels);                % campo >=0

% --- Pulse shaping rettangolare (oversampling)
tx_wave = zeros(1, Nsamples);
for k=1:Nsym
    idx = (k-1)*Nsps + (1:Nsps);
    tx_wave(idx) = E_levels(tx_symbols(k)+1);
end
% Campo complesso (reale, fase 0)
tx_wave = tx_wave .* exp(1j*0);

end

% ---------- helper (serve nel path) ----------
function bits = prbs_lfsr(order, nbits, init_state)
if nargin<3 || isempty(init_state)
    init_state = ones(order,1);
end
state = init_state(:);
bits = zeros(nbits,1);
if order==10
    taps = [10 7];
elseif order==11
    taps = [11 9];
else
    error('Aggiungi i taps per ordine %d', order);
end
for i=1:nbits
    newbit = xor(state(taps(1)), state(taps(2)));
    bits(i) = state(end);
    state = [newbit; state(1:end-1)];
end
end
