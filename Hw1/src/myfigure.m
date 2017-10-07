function graph = myfigure(xdata,ydaya,xref,yref,xName,yName,xtick,ytick,logx,logy,leg)

graph = figure;
ax1 = gca;

number = size(ydaya,1);

if logy && ~logx
    %semilogy(xdata,data1(1,:),'k-','linewidth',2);
    semilogy(xref,yref,'k-','linewidth',2);
    if number == 2
        hold on ;
        semilogy(xdata,ydaya(1,:),'b--','linewidth',2);
        semilogy(xdata,ydaya(2,:),'g-.','linewidth',2);
        hold off ;
    elseif number == 5
        hold on ;
        semilogy(xdata,ydaya(1,:),'b-','linewidth',2);
        semilogy(xdata,ydaya(2,:),'g-','linewidth',2);
        semilogy(xdata,ydaya(3,:),'y','linewidth',2);
        semilogy(xdata,ydaya(4,:),'r','linewidth',2);
        semilogy(xdata,ydaya(5,:),'m','linewidth',2);
        hold off ;    
    end
elseif ~logy && logx
    %semilogy(xdata,data1(1,:),'k-','linewidth',2);
    semilogx(xref,yref,'k-','linewidth',2);
    if number == 2
        hold on ;
        semilogx(xdata,ydaya(1,:),'k--o','linewidth',2,'MarkerFaceColor','k');
        semilogx(xdata,ydaya(2,:),'k-d','linewidth',2);
        hold off ;
    elseif number == 3
        hold on ;
        semilogx(xdata,ydaya(2,:),'k--o','linewidth',2);
        semilogx(xdata,ydaya(3,:),'k--d','linewidth',2);
        hold off ;
    end
else
    plot(xref,yref,'k-','linewidth',2);
    if number == 2
        hold on ;
        plot(xdata,ydaya(1,:),'b--','linewidth',2);
        plot(xdata,ydaya(2,:),'g-.','linewidth',2);
        hold off;
    elseif number == 5
        hold on ;
        plot(xdata,ydaya(1,:),'b-','linewidth',2);
        plot(xdata,ydaya(2,:),'g-','linewidth',2);
        plot(xdata,ydaya(3,:),'k--','linewidth',2);
        plot(xdata,ydaya(4,:),'b--','linewidth',2);
        plot(xdata,ydaya(5,:),'g--','linewidth',2);
        hold off ;       
    end
end

%axis([min(xdata) max(xdata) min(ytick) max(ytick)])
axis([min(xtick) max(xtick) min(ytick) max(ytick)])
set(ax1,...
'Units','normalized',...
'XTick',xtick,...
'YTick',ytick,...
'Position',[.15 .2 .75 .7],...
'FontUnits','points',...
'FontWeight','normal',...
'FontSize',14,...
'FontName','Times',...
'linewidth',1.2,...
'TickLength', [0.02 0.035]);

ylabel({yName},...
'FontUnits','points',...
'interpreter','latex',...
'FontSize',16,...
'FontName','Times');

xlabel({xName},...
'FontUnits','points',...
'interpreter','latex',...
'FontSize',16,...
'FontName','Times');

legend(leg);
end