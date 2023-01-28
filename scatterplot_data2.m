%%Load the files
list_lr = [1e-4,1e-3,1e-2];
bayes_acc = load("./Datasets/Data2/Clean_Metrics/bayes_fbeta_cleandata2.mat").bayes_fbeta;
br_acc=zeros(length(list_lr),50,13);
zhang_acc=zeros(length(list_lr),50,13);
sener_acc = zeros(3,length(list_lr),50,13);
for i = 1:length(list_lr)
    lr = list_lr(i);
    br_acc(i,:,:) = load(['./Datasets/Data1/Clean_Metrics/br_lr_' num2str(lr)... 
                     '_metrics_data2.mat']).br;

    zhang_acc(i,:,:) = load(['./Datasets/Data1/Clean_Metrics/zhang_lr_' num2str(lr)... 
                     '_metrics_data2.mat']).zhang;
    for j =1:3
        sener_acc(j,i,:,:) = load(['./Datasets/Data1/Clean_Metrics/sener_dnn' num2str(j) '_lr_' num2str(lr)... 
                         '_metrics_data2.mat']).sener;
    end
end

bayes_acc = repmat(bayes_acc,[length(zhang_acc(1,:,1)),1]);
%% Uncomment for finding the max lr
br_max = zeros(size(br_acc,2),length(list_lr));
for i =1:length(list_lr)
    br_max(:,i) = br_acc(i,:,2);
end
br_max = max(br_max,[],2);

zhang_max = zeros(size(zhang_acc,2),length(list_lr));
for i =1:length(list_lr)
    zhang_max(:,i) = zhang_acc(i,:,2);
end
zhang_max = max(zhang_max,[],2);

sener_max_dnn1 = zeros(size(sener_acc,3),length(list_lr));
for i =1:length(list_lr)
    sener_max_dnn1(:,i) = sener_acc(1,i,:,2);
end
sener_max_dnn1 = max(sener_max_dnn1,[],2);

sener_max_dnn2 = zeros(size(sener_acc,3),length(list_lr));
for i =1:length(list_lr)
    sener_max_dnn2(:,i) = sener_acc(2,i,:,2);
end
sener_max_dnn2 = max(sener_max_dnn2,[],2);

sener_max_dnn3 = zeros(size(sener_acc,3),length(list_lr));
for i =1:length(list_lr)
    sener_max_dnn3(:,i) = sener_acc(3,i,:,2);
end
sener_max_dnn3 = max(sener_max_dnn3,[],2);

% Interpolation of data for smoothness.
inp_data = 50;
x = zhang_acc(1,:,1);
xi = linspace(min(x), max(x), inp_data);

y = zhang_max;
zhang_max_i = interp1(x, y, xi, 'spline', 'extrap');
y = sener_max_dnn1;
sener_max_dnn1_i = interp1(x, y, xi, 'spline', 'extrap');
y = sener_max_dnn2;
sener_max_dnn2_i = interp1(x, y, xi, 'spline', 'extrap');
y = sener_max_dnn3;
sener_max_dnn3_i = interp1(x, y, xi, 'spline', 'extrap');
y = br_max;
br_max_i = interp1(x, y, xi, 'spline', 'extrap');

figure();
semilogx(xi,zhang_max_i)
xlim([10 5000]);
ylim([0.40 0.7]);
legend('Zhang','Location','southeast')
hold on;
semilogx(xi,sener_max_dnn1_i,'DisplayName','Sener 1')
semilogx(xi,sener_max_dnn2_i,'DisplayName','Sener 2')
semilogx(xi,sener_max_dnn3_i,'DisplayName','Sener 3')
semilogx(xi,br_max_i,'DisplayName','BR')
semilogx(zhang_acc(1,:,1),bayes_acc,'--','DisplayName','Bayes optimal')
savefig('data1_lr_max.fig');

%% plot everything in one figure
% Interpolation of data for smoothness.
x = zhang_acc(1,:,1);
xi = linspace(min(x), max(x), inp_data);

figure();
xlim([10 5000]);
ylim([0.40 0.7]);
legend('Location','southeast')
hold on;
for i=1:length(list_lr)
    st_z = ['Zhang_' num2str(list_lr(i))];
    st_br = ['BR_' num2str(list_lr(i))];
    semilogx(xi,...
        interp1(x, zhang_acc(i,:,2), xi, 'spline', 'extrap'),'DisplayName',st_z)
    semilogx(xi,...
        interp1(x, br_acc(i,:,2), xi, 'spline', 'extrap'),'DisplayName',st_br)
    for j=1:3
        st_s = ['Sener_' num2str(j) '_' num2str(list_lr(i))];
        semilogx(xi,...
            interp1(x, squeeze(sener_acc(j,i,:,2)), xi, 'spline', 'extrap'),...
            'DisplayName',string(st_s))
    end
    
end
semilogx(zhang_acc(1,:,1),bayes_acc,'--','DisplayName','Bayes optimal')
savefig('data1_lr_all.fig');

%% plot one figure for each lr
% x = zhang_acc(1,:,1);
% xi = linspace(min(x), max(x), 150);
% for i=1:length(list_lr)
%     figure();
%     xlim([10 5000]);
%     ylim([0.40 0.7]);
%     legend('Location','southeast')
%     hold on;
%     semilogx(zhang_acc(1,:,1),bayes_acc,'--','DisplayName','Bayes optimal')
%     st_z = ['Zhang_' num2str(list_lr(i))];
%     semilogx(xi,...
%         interp1(x, zhang_acc(i,:,2), xi, 'linear', 'extrap'),'DisplayName',st_z)
%     for j=1:3
%         st_s = ['Sener_' num2str(j) '_lr_' num2str(list_lr(i))];
%         semilogx(xi,...
%             interp1(x, squeeze(sener_acc(j,i,:,2)), xi, 'linear', 'extrap'),...
%             'DisplayName',string(st_s))
%     end
%     semilogx(xi,...
%         interp1(x, br_acc(i,:,2), xi, 'linear', 'extrap'),'DisplayName',st_br)
%     title(['Algorithm comparison for lr=' num2str(list_lr(i))])
%     savefig(['data1_lr_' num2str(list_lr(i)) '.fig'])
% end

x = zhang_acc(1,:,1);
% xi = linspace(min(x), max(x), 150);
st_z='ZA';
st_s='SA';
st_br='BR';
for i=1:length(list_lr)
    figure();
    xlim([10 312]);
    ylim([0.2 0.7]);
    legend('Location','southeast')
    hold on;
    
    %Bayes Optimal Curve/Line
    plot(zhang_acc(1,:,1),bayes_acc,...
        'color','k',...
        'LineWidth', 3.5,...
        'DisplayName','BOC');
%         'filled',...
%         'MarkerFaceColor', '#77AC30',...
%         'DisplayName','Bayes-Optimal Classifier');
    
    %Bayes Optimal Curve/Line
%     st_z = ['Zhang_' num2str(list_lr(i))];    
%     semilogx(xi,...
%         interp1(x, zhang_acc(i,:,2), xi, 'linear', 'extrap'),'DisplayName',st_z)
    s=scatter(x, zhang_acc(i,:,2), 'filled',...
        'MarkerFaceColor', 	'#77AC30' ,...
        'SizeData',80,...
        'DisplayName',st_z);
    
    %Sener DNN1, DNN2, and DNN3 curves
    for j=1:3
%         st_s = ['Sener_' num2str(j) '_lr_' num2str(list_lr(i))];
        if j==1
            scatter(x, squeeze(sener_acc(j,i,:,2)),'filled', ...
            'MarkerFaceColor', '#4DBEEE',...
            'SizeData',80,...
            'DisplayName', string(st_s)+"-1"+string(j) );
        end
        if j==2
            scatter(x, squeeze(sener_acc(j,i,:,2)),'filled', ...
            'MarkerFaceColor', '#0072BD',...
            'SizeData',80,...
            'DisplayName', string(st_s)+"-1"+string(j));
        end
        if j==3
            scatter(x, squeeze(sener_acc(j,i,:,2)),'filled', ...
            'MarkerFaceColor', 'b',...
            'SizeData',80,...
            'DisplayName', string(st_s)+"-1"+string(j));
%         semilogx(xi,...
%             interp1(x, squeeze(sener_acc(j,i,:,2)), xi, 'linear', 'extrap'),...
%             'DisplayName',string(st_s))
        end
    end
    %Binary Relevance Curve
    scatter(x, br_acc(i,:,2), 'filled', ...
        'MarkerFaceColor', 'r',...
        'SizeData',80,...
        'DisplayName',st_br);
%     semilogx(xi,...
%         interp1(x, br_acc(i,:,2), xi, 'linear', 'extrap'),'DisplayName',st_br)
    title(['Rough Test-$F_{\beta}$ accuracies with LR=' num2str(list_lr(i))],...
    'interpreter','latex');
    grid on
    hold on
    savefig(['./LearningCurves/Dataset1/dataset1_lr_' num2str(list_lr(i)) '.fig'])
%     set(gca,'xscale','log')
    
end
