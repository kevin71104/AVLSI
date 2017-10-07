function [error, weight] = LMS(d, u, M, delay, mu)
   len = size(u,2);
   eff = len-3-delay;
   
   error = zeros(1,eff);
   temp = zeros(M,1);
   weight = zeros(M,eff);
   for n = 1+delay : len-3
       un = transpose(u(1, n+3:(-1):n-M+4 ));
       R = un * ctranspose(un);
       P = un * conj(d(n));
       temp = temp + mu * (P - R * temp);
       
       y_hat = conv(u(4:1003),transpose(temp));
       y_hat1 = y_hat(8:1007);
       error(n-delay) = mean((y_hat1 - d(8:1007)).^2);
       weight(:,n-delay) = temp;
   end

end