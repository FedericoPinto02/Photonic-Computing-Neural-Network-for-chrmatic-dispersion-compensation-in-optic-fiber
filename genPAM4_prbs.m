function [tx_symbols, tx_wave, tvec] = genPAM4_prbs(Nsym, params, varargin)
% Usage:
% 1) genPAM4_prbs(Nsym, params) -> genera PRBS di default order=11
% 2) genPAM4_prbs(Nsym, params, 'prbsOrder',11, 'seed',123)
% 3) genPAM4_prbs(Nsym, params, 'externalSymbols', vec)

p = inputParser;
addOptional(p,'prbsOrder',11);
addOptional(p,'seed',[]);
addOptional(p,'externalSymbols',[]);
parse(p,varargin{:});
prbsOrder = p.Results.prbsOrder;
seed = p.Results.seed;
external = p.Results.externalSymbols;

if ~isempty(seed); rng(seed); end

if ~isempty(external)
    tx_symbols = external(:);
    Nsym = numel(tx_symbols);
else
    % generate PRBS bits -> then group log2(M) bits per symbol
    bits_needed = Nsym * 2; % PAM4 -> 2 bit/symbol
    prbs_bits = prbs_lfsr(prbsOrder, bits_needed);
    % map pairs of bits to integers 0..3 (MSB first)
    tx_symbols = zeros(Nsym,1);
    for k=1:Nsym
        b1 = prbs_bits(2*k-1);
        b0 = prbs_bits(2*k);
        tx_symbols(k) = b1*2 + b0;
    end
end

% map to levels -3,-1,1,3 (same as before)
levels = [-3 -1 1 3];
sym_levels = levels(tx_symbols+1)';

% normalization and oversampling as before
Pavg = mean(sym_levels.^2);
scale = 1/sqrt(Pavg);
sym_levels = sym_levels * scale;

Fs = params.Fs; Nsps = params.Nsps;
Nsamples = Nsym*Nsps;
tvec = (0:Nsamples-1)/Fs;
tx_wave = zeros(1,Nsamples);
for k=1:Nsym
    idx = (k-1)*Nsps + (1:Nsps);
    tx_wave(idx) = sym_levels(k);
end
tx_wave = tx_wave .* exp(1j*0); % campo complesso (zero phase iniziale)
end
