function generate_fixed_datasets(params, sizes, seeds, out_prefix)
% GENERATE_FIXED_DATASETS - crea train/val/test PRBS PAM4 IM/DD fissi e li salva su disco.
% USO:
%   generate_fixed_datasets(params, ...
%       struct('train',2000,'val',5000,'test',20000), ...
%       struct('train',2025,'val',2026,'test',2027), ...
%       'dataset');
%
% Salva:
%   dataset_train.mat, dataset_val.mat, dataset_test.mat
% Contenuto .mat:
%   tx_symbols, tx_wave, ref_power_wave (=abs(tx_wave).^2), Fs, Nsps, Baud, meta

if nargin<4, out_prefix = 'dataset'; end

fields = {'train','val','test'};
for f = 1:numel(fields)
    name = fields{f};
    Nsym = sizes.(name);
    seed = seeds.(name);

    [tx_symbols, tx_wave, tvec] = genPAM4_prbs(Nsym, params, ...
        'prbsOrder', 11, 'seed', seed, 'P_levels', [0.1 0.4 0.7 1.0], 'normalizePower', true);

    ref_power_wave = abs(tx_wave).^2;

    meta.created_at = datestr(now);
    meta.prbsOrder  = 11;
    meta.seed       = seed;
    meta.note       = sprintf('%s set: fixed PRBS IM/DD', upper(name));

    Fs   = params.Fs;
    Nsps = params.Nsps;
    Baud = params.Baud;

    fname = sprintf('%s_%s.mat', out_prefix, name);
    save(fname, 'tx_symbols','tx_wave','ref_power_wave','tvec','Fs','Nsps','Baud','meta');
    fprintf('Saved %s (%d symbols)\n', fname, Nsym);
end
end
