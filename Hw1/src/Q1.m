%% Q1
    clear;
    clc;
    w = [2.9, 3.1, 3.3];
    delay = 7;
    numSeq = 1000;
    noiseCov = 0.001;
    M = 11;
    weight = zeros(M,3);
    error = zeros(1,3);
    error_exact = zeros(1,3);
    for i = 1 : 3     
        s = RandomGenerator(numSeq, true);
        u_tmp = ChannelModel(s, w(i), noiseCov);
        d = [zeros(1,delay),s];
        % preceeding zeros is for d(8), needing u(8) to u(-2)
        u = [zeros(1,3),u_tmp]; 
        %u = [zeros(1,3),u_tmp,zeros(1,3)];  
        weight(:,i) = AdaptiveFilterWH(d, u, M, delay);
        
        y_hat = conv(u(4:1003),transpose(weight(:,i)));
        y_hat1 = y_hat(8:1007);
        error(i) = mean((y_hat1 - s).^2);
        y_hat2 = y_hat(8:1004);
        y_real = s(1:997);
        error_exact(i) = mean((y_hat2 - y_real).^2);    
    end
    error
    error_exact
    weight
