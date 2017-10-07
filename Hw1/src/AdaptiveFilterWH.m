function weight = AdaptiveFilterWH(d,u,M, delay)
    % d     : ideal signal
    % u     : received signal
    % M     : number of w
    % w : weight of filter parameter
    len = size(u,2);
    eff = len - 3 -delay;
    R = zeros(M, M, eff);
    P = zeros(M, eff);
    
    for n = 1+delay : len-3
        mu = transpose(u(1, n+3:(-1):n-M+4 ));
        R(:,:,n) = mu * ctranspose(mu);
        P(:,n) = mu * conj(d(n));
    end

    R_f = mean(R,3);
    P_f = mean(P,2);
    weight = R_f \ P_f;
end