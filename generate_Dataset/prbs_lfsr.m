function bits = prbs_lfsr(order, nbits, init_state)
% prbs_lfsr - genera PRBS con LFSR di dato ordine
% order: es. 10 o 11
% nbits: numero bit da generare
% init_state: optional vector binario length=order (default all ones)
if nargin<3 || isempty(init_state)
    init_state = ones(order,1);
end
state = init_state(:);
bits = zeros(nbits,1);
% tap positions for common PRBS (examples):
% PRBS10 taps: [10 7]  (polynomial x^10 + x^7 + 1)
% PRBS11 taps: [11 9]  (x^11 + x^9 + 1)
if order==10
    taps = [10 7];
elseif order==11
    taps = [11 9];
else
    error('Specify taps for order %d', order);
end

for i=1:nbits
    newbit = xor(state(taps(1)), state(taps(2)));
    bits(i) = state(end);       % output bit (can choose msb/lsb convention)
    state = [newbit; state(1:end-1)];
end
end
