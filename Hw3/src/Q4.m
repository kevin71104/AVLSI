%% Q4
clear;
clc;

Fs = 51.84 * 1e6;
Ts = 1 / Fs;
fc = 12.96*1e6;
T = 1/fc;

N = 2^14;
n = -N/2 : N/2-1;
len = N;
t = -len/2 : len/2-1;
A = [-3,-1,1,3];
seq1 = randi([1,4],1,N);
seq2 = randi([1,4],1,N);
an = A(seq1);
bn = A(seq2);

s = zeros(1, len);
for i = 1:len
    idx = ((i - 1 - len/2) -4*n );
    pseq = PandQ(idx,true);
    qseq = PandQ(idx,false);

    temp = an .* pseq - bn .* qseq;
    s(i) = sum(temp);
end

S = fft(s);
spectrum = abs(S);
spectrum = 20*log10(spectrum);
f = (0:N/2-1) / (N*Ts*1e6);

figure;
plot(f,spectrum(len/2+1:len),'b-');
ax1 =gca;
axis([0 30 -20 60])
set(ax1,...
'Units','normalized',...
'XTick',0:10:30,...
'YTick',-20:20:60,...
'Position',[.15 .2 .75 .7],...
'FontUnits','points',...
'FontWeight','normal',...
'FontSize',14,...
'FontName','Times',...
'linewidth',1.2,...
'TickLength', [0.02 0.035]);

ylabel('Amplitude (dB)',...
'FontUnits','points',...
'interpreter','latex',...
'FontSize',16,...
'FontName','Times');

xlabel('Frequency (MHz)',...
'FontUnits','points',...
'interpreter','latex',...
'FontSize',16,...
'FontName','Times');

%legend('show');
