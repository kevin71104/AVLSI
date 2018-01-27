%% directForm: function description
function [outputs] = directForm(sig, rawH)
	H = reshape(rawH, 1, []);
	% outputs = conv(sig, H, 'valid');
	outputs = conv(sig, H);
end