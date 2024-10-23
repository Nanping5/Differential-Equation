% 清空工作区
clear;
clc;

% 设置随机种子（可选）
rng(1);

% 1. 数据准备
numSamples1 = 500; % 数据集 I：多边形目标
numSamples2 = 500; % 数据集 II：光滑形状
numSamples3 = 500; % 数据集 III：混合形状

% 数据集 I：多边形目标（正方形、三角形等）
X1 = rand(numSamples1, 2); % 随机生成输入特征
Y1 = zeros(numSamples1, 1); % 初始化 RCS 输出
for i = 1:numSamples1
    Y1(i) = calculateRCSPolygon(X1(i, :)); % 计算RCS
end

% 数据集 II：光滑形状（圆形、椭圆形等）
X2 = rand(numSamples2, 2); % 随机生成输入特征
Y2 = zeros(numSamples2, 1); % 初始化 RCS 输出
for i = 1:numSamples2
    Y2(i) = calculateRCSEllipse(X2(i, :)); % 计算RCS
end

% 数据集 III：混合形状
X3 = rand(numSamples3, 2); % 随机生成输入特征
Y3 = zeros(numSamples3, 1); % 初始化 RCS 输出
for i = 1:numSamples3
    Y3(i) = calculateRCSMixed(X3(i, :)); % 计算RCS
end

% 划分训练集与验证集
trainRatio = 0.8;
trainIdx1 = 1:round(trainRatio * numSamples1);
valIdx1 = round(trainRatio * numSamples1) + 1:numSamples1;

trainIdx2 = 1:round(trainRatio * numSamples2);
valIdx2 = round(trainRatio * numSamples2) + 1:numSamples2;

trainIdx3 = 1:round(trainRatio * numSamples3);
valIdx3 = round(trainRatio * numSamples3) + 1:numSamples3;

XTrain1 = X1(trainIdx1, :);
YTrain1 = Y1(trainIdx1); % 注意这里不需要冒号
XVal1 = X1(valIdx1, :);
YVal1 = Y1(valIdx1); % 注意这里不需要冒号

XTrain2 = X2(trainIdx2, :);
YTrain2 = Y2(trainIdx2); % 注意这里不需要冒号
XVal2 = X2(valIdx2, :);
YVal2 = Y2(valIdx2); % 注意这里不需要冒号

XTrain3 = X3(trainIdx3, :);
YTrain3 = Y3(trainIdx3); % 注意这里不需要冒号
XVal3 = X3(valIdx3, :);
YVal3 = Y3(valIdx3); % 注意这里不需要冒号

% 2. 模型构建
layers = [
    featureInputLayer(2) % 输入层
    fullyConnectedLayer(64) % 第一隐藏层
    reluLayer % 激活函数
    fullyConnectedLayer(32) % 第二隐藏层
    reluLayer % 激活函数
    fullyConnectedLayer(1) % 输出层
    regressionLayer]; % 回归层

% 3. 训练选项
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ... % 最大训练周期数
    'MiniBatchSize', 32, ... % 批量大小
    'InitialLearnRate', 0.001, ... % 学习率
    'ValidationData', {XVal1, YVal1}, ... % 验证数据
    'Verbose', false, ... % 不显示详细输出
    'Plots', 'training-progress'); % 绘制训练过程图

% 4. 训练网络（使用数据集 I 进行训练）
net1 = trainNetwork(XTrain1, YTrain1, layers, options);

% 5. 进行预测与评估（数据集 I）
YPred1 = predict(net1, XVal1);
mse1 = mean((YPred1 - YVal1).^2);
disp(['数据集 I 的均方误差: ', num2str(mse1)]);

% 6. 计算预测时间
tic; % 启动计时
YPred1 = predict(net1, XVal1); % 针对数据集 I 进行预测
predictionTime1 = toc;
disp(['数据集 I 预测时间: ', num2str(predictionTime1), '秒']);

% 7. 可视化结果
figure;
subplot(1,2,1);
scatter(1:length(YVal1), YVal1, 'b');
title('数据集 I 实际结果');
xlabel('样本');
ylabel('RCS');

subplot(1,2,2);
scatter(1:length(YPred1), YPred1, 'r');
title('数据集 I 预测结果');
xlabel('样本');
ylabel('RCS');

% 类似的步骤可以用于数据集 II 和 III
% 这里为数据集 II
options.ValidationData = {XVal2, YVal2}; % 更新验证数据
net2 = trainNetwork(XTrain2, YTrain2, layers, options); % 训练网络

YPred2 = predict(net2, XVal2); % 进行预测
mse2 = mean((YPred2 - YVal2).^2);
disp(['数据集 II 的均方误差: ', num2str(mse2)]);

% 可视化数据集 II 的结果
figure;
subplot(1,2,1);
scatter(1:length(YVal2), YVal2, 'b');
title('数据集 II 实际结果');
xlabel('样本');
ylabel('RCS');

subplot(1,2,2);
scatter(1:length(YPred2), YPred2, 'r');
title('数据集 II 预测结果');
xlabel('样本');
ylabel('RCS');

% 数据集 III
options.ValidationData = {XVal3, YVal3}; % 更新验证数据
net3 = trainNetwork(XTrain3, YTrain3, layers, options); % 训练网络

YPred3 = predict(net3, XVal3); % 进行预测
mse3 = mean((YPred3 - YVal3).^2);
disp(['数据集 III 的均方误差: ', num2str(mse3)]);

% 可视化数据集 III 的结果
figure;
subplot(1,2,1);
scatter(1:length(YVal3), YVal3, 'b');
title('数据集 III 实际结果');
xlabel('样本');
ylabel('RCS');

subplot(1,2,2);
scatter(1:length(YPred3), YPred3, 'r');
title('数据集 III 预测结果');
xlabel('样本');
ylabel('RCS');

% 8. RCS 计算函数示例
function RCS = calculateRCSPolygon(params)
    % 假设边长为 params(1)，角度为 params(2)
    % 计算RCS（示例）
    sideLength = params(1);
    angle = params(2);
    RCS = sideLength^2 * cosd(angle); % 示例计算
end

function RCS = calculateRCSEllipse(params)
    % 假设长轴为 params(1)，短轴为 params(2)
    % 计算RCS（示例）
    majorAxis = params(1);
    minorAxis = params(2);
    RCS = majorAxis * minorAxis; % 示例计算
end

function RCS = calculateRCSMixed(params)
    % 计算混合形状的RCS（示例）
    RCS = calculateRCSPolygon(params) + calculateRCSEllipse(params); % 示例计算
end