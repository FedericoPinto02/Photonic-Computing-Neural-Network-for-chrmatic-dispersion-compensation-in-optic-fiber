clear; close all; clc;

N = 4;

theta_L1 = [1.05, 0.70];
phi_u_L1 = [0.2, -0.5];
phi_d_L1 = [-0.8, 0.3];


theta_L2 = [0.35, 1.6];
phi_u_L2 = [0.0, 0.9];
phi_d_L2 = [0.5, -0.2];

embed = @(MZI, N, i, j) embed_block(MZI, N, i, j);

% build layer1: M(1,2) and M(3,4)
MZI_12 = MZI(theta_L1(1), phi_u_L1(1), phi_d_L1(1));
MZI_34 = MZI(theta_L1(2), phi_u_L1(2), phi_d_L1(2));
M12 = embed(MZI_12, N, 1, 2);
M34 = embed(MZI_34, N, 3, 4);
Layer1 = M34 * M12; % they act on disjoint subspaces; order not critical

% build layer2: M(1,3) and M(2,4)
MZI_13 = MZI(theta_L2(1), phi_u_L2(1), phi_d_L2(1));
MZI_24 = MZI(theta_L2(2), phi_u_L2(2), phi_d_L2(2));
M13 = embed(MZI_13, N, 1, 3);
M24 = embed(MZI_24, N, 2, 4);
Layer2 = M24 * M13;

% full network (layer order: layer1 then layer2)
U_total = Layer2 * Layer1;

%% --- Verifiche numeriche ---
fprintf('Verifica unitarita'' di U_total (U^H * U):\n');
resU = U_total' * U_total;
disp(resU);

maxResid = max(abs(resU - eye(N)), [], 'all');
fprintf('Max |U^H U - I| = %.3e\n', maxResid);

% verifica conservazione energia (potenza) su vettori
x = randn(N,1) + 1j*(randn(N,1)); % random complex input
yin = x;
yout = U_total * yin;
E_in = sum(abs(yin).^2);
E_out = sum(abs(yout).^2);
fprintf('Energia in = %.6g , Energia out = %.6g , diff = %.3e\n', E_in, E_out, E_out-E_in);

%% --- Test: applicazione a segnali tempo-dominio multi-porta ---
Fs = 80e9; t = (0:999)/Fs; % 1000 samples
% creiamo quattro tones diverse per i 4 ingressi
f0 = [1.0e9, 1.3e9, 2.1e9, 2.5e9];
X = zeros(N, numel(t));
for k=1:N
    X(k,:) = exp(1j*2*pi*f0(k)*t) * (0.6 + 0.4*j*(k-1)); % amplitude+phase differente
end

% applicazione sample-wise (U_total costante)
Y = U_total * X;

% plot snippet: magnitudini degli input e output per i primi 200 campioni
idxPlot = 1:200;
figure('Name','mini-ONN: input vs output magnitudes');
for k=1:N
    subplot(N,2,2*(k-1)+1);
    plot(t(idxPlot)*1e9, abs(X(k,idxPlot)));
    ylabel(sprintf('|Xin(%d)|',k));
    if k==1, title('Inputs (magn)'); end
    subplot(N,2,2*(k-1)+2);
    plot(t(idxPlot)*1e9, abs(Y(k,idxPlot)));
    ylabel(sprintf('|Yout(%d)|',k));
    if k==1, title('Outputs (magn)'); end
end
xlabel('time (ns)');

%% --- Mostro la matrice U_total (ampiezze e fasi) ---
figure('Name','U\_total magnitude');
imagesc(abs(U_total)); colorbar; title('|U\_total|'); axis square;
figure('Name','U\_total phase');
imagesc(angle(U_total)); colorbar; title('angle(U\_total) (rad)'); axis square;

fprintf('Esecuzione demo completata.\n');

%% --- Funzione helper embed_block (locale) ---
function Mbig = embed_block(U2, Nbig, idx1, idx2)
% costrusice NxN che applica U2 sui canali (idx1, idx2) lasciando identita' altrove
% idx1, idx2: scalari indici (1-based)
if idx1 == idx2
    error('idx1 must be different from idx2');
end
% ensure indices in order
inds = [idx1, idx2];
% start with identity
Mbig = eye(Nbig);
% replace submatrix
Mbig(inds, inds) = U2;
end