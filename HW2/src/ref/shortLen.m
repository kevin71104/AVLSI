%% shortLen: function description
function [outputs] = shortLen(sig, H)
	% for M = 3, we get H = [H0; H1; H2]
	% H = sum(rawH, 1);
	
	% sig_delay0 = sig;
	% sig_delay1 = delay(sig, 1);
	% sig_delay2 = delay(sig, 2);

	%% down sample by 3
	sig_d0_down3 = downsample(sig, 3, 2);
	sig_d1_down3 = downsample(sig, 3, 1);
	sig_d2_down3 = downsample(sig, 3);

	%% output from middle H like block
	% inter1_1 = conv(sig_d0_down3 + sig_d1_down3, H(2, :) + H(3, :), 'valid');
	% inter1_2 = conv(sig_d1_down3, H(2, :), 'valid');
	% inter1_3 = conv(sig_d1_down3 + sig_d2_down3, H(1, :) + H(2, :), 'valid');
	% inter1_4 = conv(sig_d0_down3, H(3, :), 'valid');
	% inter1_5 = conv(sig_d2_down3, H(1, :), 'valid');
	% inter1_6 = conv(sig_d0_down3 + sig_d1_down3 + sig_d2_down3, H(1, :) + H(2, :) + H(3, :), 'valid');
	inter1_1 = conv(sig_d0_down3 + sig_d1_down3, H(2, :) + H(3, :));
	inter1_2 = conv(sig_d1_down3, H(2, :));
	inter1_3 = conv(sig_d1_down3 + sig_d2_down3, H(1, :) + H(2, :));
	inter1_4 = conv(sig_d0_down3, H(3, :));
	inter1_5 = conv(sig_d2_down3, H(1, :));
	inter1_6 = conv(sig_d0_down3 + sig_d1_down3 + sig_d2_down3, H(1, :) + H(2, :) + H(3, :));

	%% before upsample by 3
	inter2_1 = inter1_2 - inter1_1;
	inter2_2 = inter1_3 - inter1_2;
	inter2_3 = inter1_5 - delay(inter1_4, 1);
	inter2_4 = inter1_6;

	%% up sample by 3
	up3_1 = upsample(inter2_1, 3);
	up3_2 = upsample(inter2_2, 3);
	up3_3 = upsample(inter2_3, 3);
	up3_4 = upsample(inter2_4, 3);

	%% adder after up sample by 3, total as 4
	%% take as the 'L' order
	%% NOTICE: use ahead instead of delay the signal
	adder1 = up3_2 + delay(up3_1, 1);
	adder2 = up3_3 + delay(adder1, 1);
	adder3 = delay(up3_4, 2) + adder2;
	adder4 = adder3 - delay(adder2, 1);

	%% connect to output (y = adder4)
	outputs = adder4;

end