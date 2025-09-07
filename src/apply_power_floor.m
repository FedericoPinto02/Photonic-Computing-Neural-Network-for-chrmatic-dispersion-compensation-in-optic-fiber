function P_f = apply_power_floor(P, alpha)
% APPLY_POWER_FLOOR - evita massicci zeri dopo il rumore: clippa a un percentile basso
% USO: P_f = apply_power_floor(P, 0.01);  % floor = quantile 1%
if nargin<2 || isempty(alpha), alpha = 0.01; end
% calcolo del floor sul vettore (robusto a outlier)
flo = quantile(P(:), alpha);
P_f = max(P, flo);
end
