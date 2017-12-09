function g = raisedcosine(t,T,a,type)
len = size(t,2);
halflen = (len - 1 )/ 2;
g = zeros(1,len);
g_half = zeros(1,halflen);
t = t(t>0);
switch type
    case 'time'
        t = t / T;
        g_half = (sin(pi*(1-a)*t) + 4*a*t .* cos(pi*(1+a)*t)) ./ ...
                 (pi*t .* (1-(4*a*t).^2));
    case 'freq'
        if(a<=1)
            W = 1 / (2*T);
            f1 = t(abs(t) <= (1-a)*W);
            len1 = size(f1,2);
            f2 = t(abs(t) > (1-a)*W & t <= (1+a)*W);
            len2 = size(f2,2);
            g_half(1 : len1) = T;
            g_half(len1+1 : len1 + len2) = sqrt(1 - sin(pi*T*(f2 - W)/a)) * T / sqrt(2);
        end
end
g(1 : halflen) = g_half(halflen:-1:1);
g(halflen+1) = (pi * (1-a) + 4* a) / pi;
g(halflen+2 : len) = g_half(1 : halflen);
end
