% 输入数据
data = textread('Datasets\DS2.txt');

% 获得数据的尺寸
[n, d] = size(data);

% 获得数据矩阵和真实标签
X = data(:, 1:d-1);
ref = data(:, d);

% 设定CDC的两个参数，k表示最邻近对象数量，ratio表示内部点数量占比
% 推荐参数 DS1和DS2: [30, 0.8], DS3: [11, 0.92]
knum = 30;
ratio = 0.8;

% 设定K-means和Ncut的参数，cnum表示簇的数量
cnum = max(ref);

% 执行CDC算法、K-means算法、Ncut算法
cdc_clus = CDC(X, knum, ratio);
km_clus = kmeans(X, cnum);
sc_clus = spectralcluster(X, cnum);

% 可视化聚类结果
if (d > 3)
    X = tsne(X);
end
subplot(2,2,1);
plotcluster2(X, ref);
title('Ground Truth');
subplot(2,2,2);
plotcluster2(X, cdc_clus);
title('CDC');
subplot(2,2,3);
plotcluster2(X, km_clus);
title('K-means');
subplot(2,2,4);
plotcluster2(X, sc_clus);
title('Ncut');

% 外部评价聚类精度
[cdc_ACC, cdc_NMI, cdc_ARI] = ClustEval(ref, cdc_clus);
[km_ACC, km_NMI, km_ARI] = ClustEval(ref, km_clus);
[sc_ACC, sc_NMI, sc_ARI] = ClustEval(ref, sc_clus);
figure(2);
bar([cdc_ACC, km_ACC, sc_ACC;cdc_NMI, km_NMI, sc_NMI;cdc_ARI, km_ARI, sc_ARI]);
set(gca,'xticklabel',{'ACC','NMI','ARI'})
legend('CDC','K-means','Ncut')