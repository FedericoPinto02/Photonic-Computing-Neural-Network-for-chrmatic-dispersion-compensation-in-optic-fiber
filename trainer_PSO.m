function [best_pos, history] = trainer_PSO(objfun, init_pos, opts)
% trainer_PSO - minimal PSO optimizer
% init_pos: Nx1 initial vector
% opts: struct with numParticles, maxIter, w, c1, c2

Nvar = numel(init_pos);
numParticles = opts.numParticles;
maxIter = opts.maxIter;
w = opts.w; c1 = opts.c1; c2 = opts.c2;

% initialize swarm
X = repmat(init_pos(:)', numParticles, 1) + (rand(numParticles, Nvar)-0.5)*2*pi; % random spread
V = zeros(numParticles, Nvar);
pbest = X;
pbest_val = inf(numParticles,1);

% evaluate initial
for i=1:numParticles
    val = objfun(X(i,:)');
    pbest_val(i) = val;
end
[gbest_val, idx] = min(pbest_val);
gbest = pbest(idx,:);

history.iter = [];
history.bestFitness = [];

for iter=1:maxIter
    for i=1:numParticles
        % update velocity & position
        V(i,:) = w*V(i,:) + c1*rand*(pbest(i,:) - X(i,:)) + c2*rand*(gbest - X(i,:));
        X(i,:) = X(i,:) + V(i,:);
        % clamp positions to [-pi,pi] for phases
        X(i,:) = mod(X(i,:) + pi, 2*pi) - pi;
        % evaluate
        val = objfun(X(i,:)');
        if val < pbest_val(i)
            pbest_val(i) = val;
            pbest(i,:) = X(i,:);
        end
        if val < gbest_val
            gbest_val = val;
            gbest = X(i,:);
        end
    end
    history.iter(end+1) = iter;
    history.bestFitness(end+1) = gbest_val;
    if mod(iter,10)==0
        fprintf('PSO iter %d / %d : best loss = %.4g\n', iter, maxIter, gbest_val);
    end
end

best_pos = gbest(:);
end
