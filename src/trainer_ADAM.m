function [x_best, hist] = trainer_ADAM(objfun, x0, opts)
%
% opts:
%   .maxIter        (default 60)
%   .alpha          learning rate iniziale (default 0.02)
%   .beta1, .beta2  (default 0.9, 0.999)
%   .eps            (default 1e-8)
%   .grad_estimator 'spsa' (default) oppure 'fd'
%   .c_spsa         ampiezza perturbazione SPSA (default 0.01)
%   .fd_eps         passo per finite-diff central (default 1e-3)
%   .clip_norm      se >0, clip della norma del gradiente (default 0, disattivo)
%   .wrap_2pi       true/false: se true warpa i parametri in [-pi,pi] (default false)
%   .verbose        true/false
%
% Ritorna:
%   x_best : parametri con loss migliore trovati
%   hist   : struct con .iter, .loss, .bestLoss, .alphaTrace

    % defaults
    if ~isfield(opts,'maxIter'), opts.maxIter = 60; end
    if ~isfield(opts,'alpha'), opts.alpha = 0.02; end
    if ~isfield(opts,'beta1'), opts.beta1 = 0.9; end
    if ~isfield(opts,'beta2'), opts.beta2 = 0.999; end
    if ~isfield(opts,'eps'), opts.eps = 1e-8; end
    if ~isfield(opts,'grad_estimator'), opts.grad_estimator = 'spsa'; end
    if ~isfield(opts,'c_spsa'), opts.c_spsa = 0.01; end
    if ~isfield(opts,'fd_eps'), opts.fd_eps = 1e-3; end
    if ~isfield(opts,'clip_norm'), opts.clip_norm = 0; end
    if ~isfield(opts,'wrap_2pi'), opts.wrap_2pi = false; end
    

    x = x0(:);
    D = numel(x);
    m = zeros(D,1); v = zeros(D,1);
    alpha = opts.alpha;
    bestLoss = inf; x_best = x;

    hist.iter = (1:opts.maxIter)';
    hist.loss = nan(opts.maxIter,1);
    hist.bestLoss = nan(opts.maxIter,1);
    hist.alphaTrace = nan(opts.maxIter,1);

    for k = 1:opts.maxIter
        % stima gradiente 
        switch lower(opts.grad_estimator)
            case 'spsa'
                delta = sign(randn(D,1));         
                c = opts.c_spsa;
                fplus  = objfun(wrap_step(x + c*delta, opts.wrap_2pi));
                fminus = objfun(wrap_step(x - c*delta, opts.wrap_2pi));
                g = ((fplus - fminus) / (2*c)) * delta;  
            case 'fd'
                g = zeros(D,1);
                f0 = objfun(x);
                h = opts.fd_eps;
                for i=1:D
                    ei = zeros(D,1); ei(i) = 1;
                    fp = objfun(wrap_step(x + h*ei, opts.wrap_2pi));
                    fm = objfun(wrap_step(x - h*ei, opts.wrap_2pi));
                    g(i) = (fp - fm)/(2*h);
                end
            otherwise
                error('Unknown grad_estimator: %s', opts.grad_estimator);
        end

       
        if opts.clip_norm > 0
            gn = norm(g);
            if gn > opts.clip_norm
                g = g * (opts.clip_norm / gn);
            end
        end

        % update ADAM 
        b1 = opts.beta1; b2 = opts.beta2; eps = opts.eps;
        m = b1*m + (1-b1)*g;
        v = b2*v + (1-b2)*(g.^2);
        mhat = m / (1 - b1^k);
        vhat = v / (1 - b2^k);
        x = x - alpha * mhat ./ (sqrt(vhat) + eps);

       
        x = wrap_step(x, opts.wrap_2pi);

        % eval corrente e best 
        fcur = objfun(x);
        if fcur < bestLoss
            bestLoss = fcur;
            x_best = x;
        end

        
        alpha = 0.98 * alpha;

        
        hist.loss(k) = fcur;
        hist.bestLoss(k) = bestLoss;
        hist.alphaTrace(k) = alpha;


        
    end
end

function xw = wrap_step(x, do_wrap)
    if do_wrap
        xw = mod(x + pi, 2*pi) - pi;
    else
        xw = x;
    end
end
