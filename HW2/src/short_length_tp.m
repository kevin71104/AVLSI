function outputs = short_length_tp(x,h1,M)

H = reshape(h1(1:12),M,12/M); % [0,3,6,9];[1,4,7,10];[2,5,8,11];

% before downsample
tmp1 = delay(x,2);
tmp2 = x - delay(x,1);
tmp3 = delay(tmp2,1);
tmp4 = delay(tmp3,1);
% after downsample
down1 = downsample(tmp1,3);
down2 = downsample(tmp2,3);
down3 = downsample(tmp3,3);
down4 = downsample(tmp4,3);
clear tmp1 tmp2 tmp3 tmp4;
% after short-length filter
w_5 = conv(down4, H(2,:)+H(3,:));
w_4 = conv(down4 - down3, H(2,:));
w_3 = conv(down3, H(2,:)+H(1,:));
w_2 = conv(delay(down2,1), H(3,:));
w_1 = conv(down2, H(1,:));
w_0 = conv(down1, H(1,:)+H(2,:)+H(3,:));
clear down1 down2 down3 down4;
% before upsampling
adder1 = w_1 + w_3;
adder2 = w_0 + adder1;
adder5 = w_0 - w_5;
adder4 = w_4 + adder5;
adder3 = w_3 + adder4;
adder6 = adder5 - w_2;
clear w_0 w_1 w_2 w_3 w_4 w_5 adder1 adder4 adder5;
% after upsampling
up1 = upsample(adder2,M);
up2 = upsample(adder3,M);
up3 = upsample(adder6,M);
clear adder2 adder3 adder6;
% go through adders
adder7 = delay(up1,1) + up2;
adder8 = delay(adder7,1) + up3;
clear adder7 up1 up2 up3;
% outputs: don't know why output has two more delays
outputs = [adder8(3:end),zeros(1,2)];
clear adder8;
end