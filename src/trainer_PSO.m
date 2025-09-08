function [best_pos, history] = trainer_PSO(objfun, init_pos, opts)
%
% opts richiesti:
%   .numParticles
%   .maxIter
%   .c1, .c2
%
% opts facoltativi (nuovi):
%   .wStart (default 0.9)  - inerzia inizio
%   .wEnd   (default 0.4)  - inerzia fine (lineare)
%   .tol    (default 0.01) - soglia miglioramento per reset stall
%   .patience (default 15) - iter senza miglioramento -> stop
%   .initSpread (default pi) - ampiezza random attorno a init_pos
%
% Backward-compat:
%   Se esiste opts.w (vecchio schema), allora wStart=wEnd=opts.w (niente annealing).

    % dimensioni/problem setup 
    Nvar         = numel(init_pos);
    numParticles = opts.numParticles;
    maxIter      = opts.maxIter;
    c1 = opts.c1;  c2 = opts.c2;

    % default nuovi
    wStart     = 0.9;  if isfield(opts,'wStart'),     wStart = opts.wStart; end
    wEnd       = 0.4;  if isfield(opts,'wEnd'),       wEnd   = opts.wEnd;   end
    tol        = 0.02; if isfield(opts,'tol'),        tol    = opts.tol;    end
    patience   = 15;   if isfield(opts,'patience'),   patience = opts.patience; end
    initSpread = pi;   if isfield(opts,'initSpread'), initSpread = opts.initSpread; end

    % compatibilità: opts.w fisso prevale 
    if isfield(opts,'w')
        wStart = opts.w;
        wEnd   = opts.w;
    end

    % inizializzazione swarm 
    X = repmat(init_pos(:)', numParticles, 1) + (rand(numParticles, Nvar)-0.5)*2*initSpread;
    V = zeros(numParticles, Nvar);
    pbest = X;
    pbest_val = inf(numParticles,1);

    % valutazione iniziale 
    for i=1:numParticles
        val = objfun(X(i,:)');
        pbest_val(i) = val;
    end
    [gbest_val, idx] = min(pbest_val);
    gbest = pbest(idx,:);

    % history 
    history.iter = [];
    history.bestFitness = [];

    % early-stop bookkeeping 
    prev_best = gbest_val;
    stall = 0;

    for iter = 1:maxIter
        % inerzia "annealed"
        if maxIter > 1
            w = wStart + (wEnd - wStart) * (iter-1)/(maxIter-1);
        else
            w = wStart;
        end

        for i = 1:numParticles
            % update velocità & posizione
            V(i,:) = w*V(i,:) + c1*rand*(pbest(i,:) - X(i,:)) + c2*rand*(gbest - X(i,:));
            X(i,:) = X(i,:) + V(i,:);

            % wrap parametri (fasi) in [-pi, pi]
            X(i,:) = mod(X(i,:) + pi, 2*pi) - pi;

            % valuta e aggiorna best locali/globali
            val = objfun(X(i,:)');
            if val < pbest_val(i)
                pbest_val(i) = val;
                pbest(i,:)   = X(i,:);
            end
            if val < gbest_val
                gbest_val = val;
                gbest     = X(i,:);
            end
        end

        % history + log
        history.iter(end+1) = iter;
        history.bestFitness(end+1) = gbest_val;
        if mod(iter,10)==0
            fprintf('PSO iter %d / %d : best loss = %.4g\n', iter, maxIter, gbest_val);
        end

        % early stopping 
        if (prev_best - gbest_val) > tol
            stall = 0;
            prev_best = gbest_val;
        else
            stall = stall + 1;
        end
        if stall >= patience
            if mod(iter,10) ~= 0
                fprintf('PSO iter %d / %d : best loss = %.4g (early stop)\n', iter, maxIter, gbest_val);
            end
            break;
        end
    end

    best_pos = gbest(:);
end
