function seq = RandomGenerator(len, setseed)
    % Bernoulli sequence
    % len : the length of the signal
    if setseed
        rng(10);
    end
    seq = rand(1, len);
    seq(seq >= 0.5) = 1;
    seq(seq < 0.5) = -1;
end