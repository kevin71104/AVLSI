% H array: parameter from Q2
H = [0.0000 -0.0840 -0.1115; ...
	 0.0000  0.2415  0.4925; ...
	 0.5994  0.4925  0.2415; ...
	 0.0000 -0.1115 -0.0840];
H = H.'
%%
% signal generation
sig = randi([0 1], 1, 90); % length as multiple of 3, easier for reshape
% sig = cat(2, ones(1, 1), zeros(1, 89));

sig_Direct = directForm(sig, H);
sig_Short = shortLen(sig, H);
sig_Short_t  = shortLen_t(sig, H);

t = 0:1:29;

figure
subplot(3,1,1)
stem(t, sig(:, 1:30), 'filled')
title('random-generated signal', 'fontsize', 20)

subplot(3,1,2)
stem(t, sig_Direct(:, 1:30), 'filled')
title('filtered by direct form FIR', 'fontsize', 20)

subplot(3,1,3)
stem(t, sig_Short(:, 1:30), 'filled')
title('filtered by short length FIR', 'fontsize', 20)

figure
subplot(3,1,1)
stem(t, sig(:, 1:30), 'filled')
title('random-generated signal', 'fontsize', 20)

subplot(3,1,2)
stem(t, sig_Direct(:, 1:30), 'filled')
title('filtered by direct form FIR', 'fontsize', 20)

subplot(3,1,3)
stem(t, sig_Short_t(:, 1:30), 'filled')
title('filtered by transposed short length FIR', 'fontsize', 20)
