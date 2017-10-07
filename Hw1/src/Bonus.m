%% BONUS
    clear;
    clc;
    w = 2.9;
    delay = 7;
    numSeq = 1000;
    noiseCov = 0.001;
    M = 11;
    mu = 0.09:0.01:0.14;
    
    error_mtx = zeros(size(mu,2),997);
    %weight_mtx = zeros(M,997,size(mu,2));
    
    for i = 1 : size(mu,2) 
        error_iter = zeros(200,997);
        error_L_iter = zeros(200,997);
        weight_iter = zeros(M,997,200);
        for j = 1 : 200
            s = RandomGenerator(numSeq, false);
            u_tmp = ChannelModel(s, w, noiseCov);
            d = [zeros(1,delay),s];
            % preceeding zeros is for d(8), needing u(8) to u(-2)
            u = [zeros(1,3),u_tmp];
        
            %[error_iter(j,:), weight_iter(:,:,j)] = NLMS(d,u,M,delay,mu);
            [error_iter(j,:), ~] = LMS(d,u,M,delay,mu(i));
        end
        error_mtx(i,:) = mean(error_iter,1);
        %weight_mtx(:,:,i) = mean(weight_iter,3);
    end
    
    myfigure(1:997,error_mtx(2:6,:),1:997,error_mtx(1,:),...
             'Number of Iterations', 'Mean-Square Error',...
             0:100:1000, [.001,.1,1,10,1e3,1e5,1e7,1e9], false,true,...
             {'\mu = 0.09', '\mu = 0.10', '\mu = 0.11',...
              '\mu = 0.12', '\mu = 0.13', '\mu = 0.14'});
    myfigure(1:997,error_mtx(2:6,:),1:997,error_mtx(1,:),...
             'Number of Iterations', 'Mean-Square Error',...
             0:100:1000, [.01,.1,1,10,100,1000,10000], false,true,...
             {'\mu = 1.80', '\mu = 1.85', '\mu = 1.90',...
              '\mu = 1.95', '\mu = 2.00', '\mu = 2.05'});