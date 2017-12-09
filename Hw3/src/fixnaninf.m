function o = fixnaninf(i)
i(isinf(i)) = nan;
a = find(isnan(i));
for j = a
    i(a) = (i(a-1) + i(a+1)) / 2;
end
o = i;
end