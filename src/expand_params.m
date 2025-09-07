function PM = expand_params(x, N, mode)
% EXPAND_PARAMS  -> Restituisce una matrice Nx3 [theta, phi_u, phi_d]
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
