%% shortLen_t: function description
function [outputs] = shortLen_t(sig, H)
	% transpose edition of shortLen.m

	%% befor upsample by 3, from bottom to top
	inter1_1 = delay(sig, 2);
	inter1_2 = sig - delay(sig, 1);
	inter1_3 = delay(inter1_2, 1);
	inter1_4 = delay(inter1_3, 1);

	%% up sample, from bottom to top
	up1 = downsample(inter1_1, 3);
	up2 = downsample(inter1_2, 3);
	up3 = downsample(inter1_3, 3);
	up4 = downsample(inter1_4, 3);

	%% before H block in the medium, from bottom to top
	inter2_1 = conv(up1, H(1, :) + H(2, :) + H(3, :));
	inter2_2 = conv(up2, H(1, :));
	inter2_3 = conv(delay(up2, 1) .* -1, H(3, :));
	inter2_4 = conv(up3, H(1, :) + H(2, :));
	inter2_5 = conv(up4 - up3, H(2, :));
	inter2_6 = conv(up4 .* -1, H(2, :) + H(3, :));

	%% before down sample by 3, from bottom to top
	inter3_1 = inter2_1 + inter2_2 + inter2_4;
	inter3_2 = inter2_1 + inter2_6 + inter2_5 + inter2_4;
	inter3_3 = inter2_1 + inter2_6 + inter2_3;

	%% down sample by 3, from bottom to top
	down1 = upsample(inter3_1, 3);
	down2 = upsample(inter3_2, 3);
	down3 = upsample(inter3_3, 3);

	%% to output, from bottom to top
	y = delay(down1, 1) + down2;
	y = delay(y, 1) + down3;

	%% connect to outputs
	outputs = cat(2, y(:, 3:end), zeros(1, 2));

end

