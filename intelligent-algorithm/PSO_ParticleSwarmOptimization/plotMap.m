%% 作图
function plotMap(n,cityCoor,tourGbest,N)

clf
hold on
plot([cityCoor(tourGbest(1),1),cityCoor(tourGbest(n),1)],[cityCoor(tourGbest(1),2),...
    cityCoor(tourGbest(n),2)],'ms-','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','g')
hold on
for i=2:n
    plot([cityCoor(tourGbest(i-1),1),cityCoor(tourGbest(i),1)],[cityCoor(tourGbest(i-1),2),...
        cityCoor(tourGbest(i),2)],'ms-','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','g')
    hold on
end

legend('规划路径')
scatter(cityCoor(:,1),cityCoor(:,2));
theTitle = ['规划路径',num2str(N)];
title(theTitle,'fontsize',10)
xlabel('km','fontsize',10)
ylabel('km','fontsize',10)

grid on
ylim([4 80])
drawnow
end
