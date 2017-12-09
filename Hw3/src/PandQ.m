function p = PandQ(seq,phase)
% t = multiple T' = multiple T/4
if phase 
    p = (4/pi) * (cos(seq*pi/2)).^2 ./ (1-seq.^2);
    p(seq == 1) = 0.0;
    p(seq == -1) = 0.0;
else
    p = (4/pi) * (cos(seq*pi/2) .* sin(seq*pi/2)) ./ (1-seq.^2);
    p(seq == 1) = 1.0;
    p(seq == -1) = -1.0;
end
end