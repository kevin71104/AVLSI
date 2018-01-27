function outputs = delay(x,units)
length = size(x,2);
outputs = [zeros(1,units), x(1:length-units)];
end