function Lp = PropaLoss(f,cat)
% f should be in MHz
% Lp is calculated per 100 m
switch cat
    case 3
        Lp = 2.320 * sqrt(f) + 0.238 * f;
    case 4
        Lp = 2.050 * sqrt(f) + 0.043 * f + 0.057 ./ sqrt(f);
    case 5
        Lp = 1.967 * sqrt(f) + 0.023 * f + 0.050 ./ sqrt(f);
end
end