%% Q3--(i)
clear;
    clc;
    w = 2.9;
    delay = 7;
    numSeq = 1000;
    noiseCov = 0.001;
    M = 11;
    mu = [0.04, 0.02, 0.01];
    
    error_mtx = zeros(3,997);
    error_L_mtx = zeros(3,997);
    weight_mtx = zeros(M,997,3);
    
    for i = 1 : 3 
        error_iter = zeros(200,997);
        error_L_iter = zeros(200,997);
        weight_iter = zeros(M,997,200);
        for j = 1 : 200
            s = RandomGenerator(numSeq, false);
            u_tmp = ChannelModel(s, w, noiseCov);
            d = [zeros(1,delay),s];
            % preceeding zeros is for d(8), needing u(8) to u(-2)
            u = [zeros(1,3),u_tmp];
        
            [error_iter(j,:), weight_iter(:,:,j)] = NLMS(d,u,M,delay,mu(i));
            [error_L_iter(j,:), ~] = LMS(d,u,M,delay,mu(i));
        end
        error_mtx(i,:) = mean(error_iter,1);
        error_L_mtx(i,:) = mean(error_L_iter,1);
        weight_mtx(:,:,i) = mean(weight_iter,3);
    end
         
    ydata = zeros(5,997);
    ydata(1:2,:) = error_mtx(2:3,:);
    ydata(3:5,:) = error_L_mtx;
    
    myfigure(1:997,ydata,1:997,error_mtx(1,:),...
             'Number of Iterations', 'Mean-Square Error',...
             0:100:1000, 0:.1:1, false,false,...
             {'\mu = 0.04, NLMS', '\mu = 0.02, NLMS', '\mu = 0.01, NLMS',...
              '\mu = 0.04, LMS',  '\mu = 0.02, LMS',  '\mu = 0.01, LMS'});
    myfigure(1:997,ydata,1:997,error_mtx(1,:),...
             'Number of Iterations', 'Mean-Square Error',...
             0:100:1000, [.001,.1,1], false,true,...
             {'\mu = 0.04, NLMS', '\mu = 0.02, NLMS', '\mu = 0.01, NLMS',...
              '\mu = 0.04, LMS',  '\mu = 0.02, LMS',  '\mu = 0.01, LMS'});     
    error_mtx(:,990:997)
%% Q3--(ii)
    clear;
    clc;
    w = [2.9, 3.1, 3.3];
    delay = 7;
    numSeq = 1000;
    noiseCov = 0.001;
    M = 11;
    mu = 0.02;
    
    error_mtx = zeros(3,997);
    error_L_mtx = zeros(3,997);
    weight_mtx = zeros(M,997,3);
    
    for i = 1 : 3 
        error_iter = zeros(200,997);
        error_L_iter = zeros(200,997);
        weight_iter = zeros(M,997,200);
        for j = 1 : 200
            s = RandomGenerator(numSeq, false);
            %s = s * 3;
            u_tmp = ChannelModel(s, w(i), noiseCov);
            d = [zeros(1,delay),s];
            % preceeding zeros is for d(8), needing u(8) to u(-2)
            u = [zeros(1,3),u_tmp];
        
            [error_iter(j,:), weight_iter(:,:,j)] = NLMS(d,u,M,delay,mu);
            [error_L_iter(j,:), ~] = LMS(d,u,M,delay,mu);
        end
        error_mtx(i,:) = mean(error_iter,1);
        error_L_mtx(i,:) = mean(error_L_iter,1);
        weight_mtx(:,:,i) = mean(weight_iter,3);
    end
    
    ydata = zeros(5,997);
    ydata(1:2,:) = error_mtx(2:3,:);
    ydata(3:5,:) = error_L_mtx;
    
    myfigure(1:997,ydata,1:997,error_mtx(1,:),...
             'Number of Iterations', 'Mean-Square Error',...
             0:100:1000, 0:.1:1, false,false,...
             {'W = 2.9, NLMS', 'W = 3.1, NLMS', 'W = 3.3, NLMS',...
              'W = 2.9, LMS',  'W = 3.1, LMS',  'W = 3.3, LMS'});
    myfigure(1:997,ydata,1:997,error_mtx(1,:),...
             'Number of Iterations', 'Mean-Square Error',...
             0:100:1000, [.001,.01,.1,1], false,true,...
             {'W = 2.9, NLMS', 'W = 3.1, NLMS', 'W = 3.3, NLMS',...
              'W = 2.9, LMS',  'W = 3.1, LMS',  'W = 3.3, LMS'});
          
%% Q3--(iii)
clear;
    clc;
    w = 2.9;
    delay = 7;
    numSeq = 1000;
    noiseCov = 0.001;
    M = 11;
    mu = 0.02;
        
    error_iter = zeros(200,997);
    weight_iter = zeros(M,997,200);
    weight_ref = zeros(M,200);
    for j = 1 : 200
        s = RandomGenerator(numSeq, false);
        u_tmp = ChannelModel(s, w, noiseCov);
        d = [zeros(1,delay),s];
        % preceeding zeros is for d(8), needing u(8) to u(-2)
        u = [zeros(1,3),u_tmp];
        
        [error_iter(j,:), weight_iter(:,:,j)] = NLMS(d,u,M,delay,mu);       
        weight_ref(:,j) = AdaptiveFilterWH(d, u, M, delay);
    end
    error = mean(error_iter,1);
    weight = mean(weight_iter,3);
    weight_ref = mean(weight_ref,2);

    for m = 1 : M
        figure;
        plot([0, 1000], [weight_ref(m), weight_ref(m)],...
            'r', 'LineStyle', ':','linewidth',2);
        hold on;
        plot(1:997, weight(m,:), 'k-','linewidth',2);
        hold off;
    
        axis([0 1000 -0.4 1.4]);
        set(gca,...
        'Units','normalized',...
        'XTick',0:100:1000,...
        'YTick',-0.4:.2:1.4,...
        'Position',[.15 .2 .75 .7],...
        'FontUnits','points',...
        'FontWeight','normal',...
        'FontSize',14,...
        'FontName','Times',...
        'linewidth',1.2,...
        'TickLength', [0.02 0.035]);

        ylabel('Weight',...
        'FontUnits','points',...
        'interpreter','latex',...
        'FontSize',16,...
        'FontName','Times');

        xlabel('Number of Iterations',...
        'FontUnits','points',...
        'interpreter','latex',...
        'FontSize',16,...
        'FontName','Times');
        legend({'ideal value', 'learning curve'});
    end
    
%% Q3--(IV)
    clear;
    clc;
    w = [5,10,15];
    delay = 7;
    numSeq = 1000;
    noiseCov = .001;
    M = 11;
    mu = 0.02;
    
    error_mtx = zeros(3,997);
    error_L_mtx = zeros(3,997);
    weight_mtx = zeros(M,997,3);
    
    for i = 1 : 3 
        error_iter = zeros(200,997);
        error_L_iter = zeros(200,997);
        weight_iter = zeros(M,997,200);
        for j = 1 : 200
            s = RandomGenerator(numSeq, false);
            %s = s * 3;
            u_tmp = ChannelModel(s, w(i), noiseCov);
            d = [zeros(1,delay),s];
            % preceeding zeros is for d(8), needing u(8) to u(-2)
            u = [zeros(1,3),u_tmp];
        
            [error_iter(j,:), weight_iter(:,:,j)] = NLMS(d,u,M,delay,mu);
            [error_L_iter(j,:), ~] = LMS(d,u,M,delay,mu);
        end
        error_mtx(i,:) = mean(error_iter,1);
        error_L_mtx(i,:) = mean(error_L_iter,1);
        weight_mtx(:,:,i) = mean(weight_iter,3);
    end
    
    ydata = zeros(5,997);
    ydata(1:2,:) = error_mtx(2:3,:);
    ydata(3:5,:) = error_L_mtx;
%{    
    myfigure(1:997,ydata,1:997,error_mtx(1,:),...
             'Number of Iterations', 'Mean-Square Error',...
             0:100:1000, 0:1:10, false,false,...
             {'W = 2.9, NLMS', 'W = 3.1, NLMS', 'W = 3.3, NLMS',...
              'W = 2.9, LMS',  'W = 3.1, LMS',  'W = 3.3, LMS'});
%}
    
    myfigure(1:997,ydata,1:997,error_mtx(1,:),...
             'Number of Iterations', 'Mean-Square Error',...
             0:100:1000, [.01,.1,1,10], false,true,...
             {'W = 5.0, NLMS', 'W = 10.0, NLMS', 'W = 15.0, NLMS',...
              'W = 5.0, LMS',  'W = 10.0, LMS',  'W = 15.0, LMS'});
          