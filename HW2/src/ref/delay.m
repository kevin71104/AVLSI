%% delay: function description
function [outputs] = delay(sig, delay_step)
	% shift the 'sig' right with 'delay_step' columns
	% and fill the left most columns with zeros
	outputs = cat(2, zeros(1, delay_step), sig(:, 1:end-delay_step));
end