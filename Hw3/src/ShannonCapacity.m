function c = ShannonCapacity(d,W,cat)
% calculate per 100 m (propaLoss)
d = d/100;
Ln = NEXTLoss(W,cat);
Lp = PropaLoss(W/2,cat);
c = (W/3) * (Ln + 15 / log(10) - (2*sqrt(2)/3)*d*Lp);
end