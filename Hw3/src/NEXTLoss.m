function Ln = NEXTLoss(f,cat)
f0 = 0.772;
switch cat
    case 3
        Ln = 43 - 15 * log10(f/f0);
    case 4
        Ln = 58 - 15 * log10(f/f0);
    case 5
        Ln = 64 - 15 * log10(f/f0);
end
end