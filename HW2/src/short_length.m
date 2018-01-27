function outputs = short_length(x,h1,M)

H = reshape(h1(1:12),M,12/M); % [0,3,6,9];[1,4,7,10];[2,5,8,11];
X = reshape(x,M,size(x,2)/M); % X(i,:) = X_{i-1}(z^3)
%{
X_0 = downsample(x,3,0);
X_1 = downsample(x,3,1);
X_2 = downsample(x,3,1);
%}
% after short-length filter
w_0 = conv(X(3,:)+X(2,:),H(2,:)+H(3,:));
w_1 = conv(X(2,:),H(2,:));
w_2 = conv(X(1,:)+X(2,:),H(2,:)+H(1,:));
w_3 = conv(X(3,:),H(3,:));
w_4 = conv(X(1,:),H(1,:));
w_5 = conv(X(1,:)+X(2,:)+X(3,:),H(1,:)+H(2,:)+H(3,:));
% before upsampling
tmp1 = w_1 - w_0;
tmp2 = w_2 - w_1;
tmp3 = w_4 - delay(w_3,1);
clear w_0 w_1 w_2 w_3 w_4;
% after upsampling
up1 = upsample(tmp1,M);
up2 = upsample(tmp2,M);
up3 = upsample(tmp3,M);
up4 = upsample(w_5,M);
clear tmp1 tmp2 tmp3 w_5;
% go through adders
adder1 = delay(up1,1) + up2;
adder2 = delay(adder1,1) + up3;
adder3 = delay(up4,2) + adder2;
adder4 = adder3 - delay(adder2,1);
clear up1 up2 up3 up4 adder1 adder2 adder3;
outputs = adder4;
end