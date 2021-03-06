%% Q3
clear;
clc;
T = 1/(12.96*1e6);
fc = 12.96*1e6;
t = (-2048:2048) * T/512;
tmin = min(t)/T;
tmax = max(t)/T;
g = raisedcosine(t,T,0.5,'time');
p = g .* cos(2*pi*fc*t);
q = g .* sin(2*pi*fc*t);

figure;
hold on;
%plot(t,g,'r','linewidth',2, 'DisplayName', 'baseband');
plot(t/T,p,'k','linewidth',2, 'DisplayName', 'In-Phase');
plot(t/T,q,'b','linewidth',2, 'DisplayName', 'Quadrature');

ax1 =gca;
axis([tmin tmax -1.2 1.6])
set(ax1,...
'Units','normalized',...
'XTick',-4:2:4,...
'XTickLabel',{'-4T','-2T','0','2T','4T'},...
'YTick',-1.2:.4:1.6,...
'Position',[.15 .2 .75 .7],...
'FontUnits','points',...
'FontWeight','normal',...
'FontSize',14,...
'FontName','Times',...
'linewidth',1.2,...
'TickLength', [0.02 0.035]);

ylabel('Amplitude',...
'FontUnits','points',...
'interpreter','latex',...
'FontSize',16,...
'FontName','Times');

xlabel('Time (normalized to T)',...
'FontUnits','points',...
'interpreter','latex',...
'FontSize',16,...
'FontName','Times');

legend('show');

%% Q3-2
clear;
clc;
T = 1/(12.96*1e6);
fc = 12.96*1e6;
f = (-1 : 0.0001 : 1) * fc;
f_all = (-2 : 0.0001 : 2) * fc;
G = raisedcosine(f,T,0.5,'freq');
len = (size(f,2)-1)/2;

P = zeros(1,4*len+1);
P(1:2*len+1) = G;
P(2*len+1:4*len+1) = G;
Q = zeros(1,4*len+1);
Q(1:2*len+1) = -1 * G;
Q(2*len+1:4*len+1) = G;
H = zeros(1,4*len+1);
H(len+1:3*len+1) = G;

figure;
hold on;
plot(f_all,H,'k-','linewidth',2, 'DisplayName', 'Baseband');
plot(f_all,P,'b-','linewidth',2, 'DisplayName', 'In-Phase');
plot(f_all,Q,'r--','linewidth',2, 'DisplayName', 'Quarature');

ax1 =gca;
axis([-2*fc 2*fc -1.1*T 1.1*T])
set(ax1,...
'Units','normalized',...
'XTick',(-2:2)*fc,...
'YTick',(-1:.5:1)*T,...
'XTickLabel',{'-2f_c','-f_c','0','f_c','2f_c'},...
'YTickLabel',{'-T','-0.5T','0','0.5T','T'},...
'Position',[.15 .2 .75 .7],...
'FontUnits','points',...
'FontWeight','normal',...
'FontSize',14,...
'FontName','Times',...
'linewidth',1.2,...
'TickLength', [0.02 0.035]);

ylabel('Amplitude',...
'FontUnits','points',...
'interpreter','latex',...
'FontSize',16,...
'FontName','Times');

xlabel('Frequency (Normalized to $$f_c$$)',...
'FontUnits','points',...
'interpreter','latex',...
'FontSize',16,...
'FontName','Times');

legend('show');

%% Q3-3
clear;
clc;
T = 1/(12.96*1e6);
fc = 12.96*1e6;
t = (-2048:2048) * T/512;
tmin = min(t)/T;
tmax = max(t)/T;
g = raisedcosine(t,T,0.5,'time');
p = g .* cos(2*pi*fc*t);
q = g .* sin(2*pi*fc*t);

g = fixnaninf(g(1:4096));
p = fixnaninf(p(1:4096));
q = fixnaninf(q(1:4096));

%n = 2^nextpow2(length(g));
G = fft(g);
P = fft(p);
Q = fft(q);
f = (1:4096) / (4096*T);

%plot(abs(G),'r');
%plot(abs(fftshift(G)),'b');
%plot(abs(fftshift(P)),'k');
%plot(abs(fftshift(Q)),'r');
%scatter(angle(fftshift(P)),angle(fftshift(Q)));
%scatter(angle(P),angle(Q));
%scatter(abs(fftshift(P)),abs(fftshift(Q)));
idxP = find(abs(P) > 5);
idxQ = find(abs(Q) > 5);
%scatter(abs(P),abs(Q));
tmp = P(idxP) ./ Q(idxQ);
len = size(tmp,2);

figure;
ax1 = subplot(1,1,1);
stem(f(idxP)/fc,abs(P(idxP)),'filled','b-',...
     'LineWidth',1, 'DisplayName', 'In-Phase');
hold on;
stem(f(idxP)/fc,abs(Q(idxP)),'r--',...
     'LineWidth',1, 'DisplayName', 'Quarature');

axis([0 1 0 300])
set(ax1,...
'Units','normalized',...
'XTick',0:.5:1,...
'YTick',0:100:300,...
'XTickLabel',{'0','0.5f_c','f_c'},...
'FontUnits','points',...
'FontWeight','normal',...
'FontSize',14,...
'FontName','Times',...
'linewidth',1.2,...
'TickLength', [0.02 0.035]);

ylabel('Amplitude',...
'FontUnits','points',...
'interpreter','latex',...
'FontSize',16,...
'FontName','Times');

xlabel('Frequency (Normalized to $$f_c$$)',...
'FontUnits','points',...
'interpreter','latex',...
'FontSize',16,...
'FontName','Times');
legend('show');

figure;
ax2 = subplot(1,1,1);
scatter(f(idxP)/fc,abs(tmp),'b');
axis([0 1 0.8 1.2])
set(ax2,...
'Units','normalized',...
'XTick',0:.5:1,...
'YTick',0.8:.2:1.2,...
'XTickLabel',{'0','0.5f_c','f_c'},...
'FontUnits','points',...
'FontWeight','normal',...
'FontSize',14,...
'FontName','Times',...
'linewidth',1.2,...
'TickLength', [0.02 0.035]);

ylabel('Ratio',...
'FontUnits','points',...
'interpreter','latex',...
'FontSize',16,...
'FontName','Times');

xlabel('Frequency (Normalized to $$f_c$$)',...
'FontUnits','points',...
'interpreter','latex',...
'FontSize',16,...
'FontName','Times');

figure;
ax3 = subplot(1,1,1);
scatter(f(idxP)/fc,angle(P(idxP))/pi,'b','DisplayName', 'In-Phase');
hold on;
scatter(f(idxP)/fc,angle(Q(idxP))/pi,'r','DisplayName', 'Quarature');

axis([0 1 -1 1])
set(ax3,...
'Units','normalized',...
'XTick',0:.5:1,...
'YTick',-1:.5:1,...
'XTickLabel',{'0','0.5f_c','f_c'},...
'FontUnits','points',...
'FontWeight','normal',...
'FontSize',14,...
'FontName','Times',...
'linewidth',1.2,...
'TickLength', [0.02 0.035]);

ylabel('Phase (Normalized to $$\pi$$)',...
'FontUnits','points',...
'interpreter','latex',...
'FontSize',16,...
'FontName','Times');

xlabel('Frequency (Normalized to $$f_c$$)',...
'FontUnits','points',...
'interpreter','latex',...
'FontSize',16,...
'FontName','Times');
legend('show');

figure;
ax4 = subplot(1,1,1);
scatter(f(idxP)/fc,abs(angle(tmp))/pi,'b');
axis([0 1 0 1])
set(ax4,...
'Units','normalized',...
'XTick',0:.5:1,...
'YTick',0:.5:1,...
'XTickLabel',{'0','0.5f_c','f_c'},...
'FontUnits','points',...
'FontWeight','normal',...
'FontSize',14,...
'FontName','Times',...
'linewidth',1.2,...
'TickLength', [0.02 0.035]);

ylabel('Angular difference (Normalized to $$\pi$$)',...
'FontUnits','points',...
'interpreter','latex',...
'FontSize',16,...
'FontName','Times');

xlabel('Frequency (Normalized to $$f_c$$)',...
'FontUnits','points',...
'interpreter','latex',...
'FontSize',16,...
'FontName','Times');