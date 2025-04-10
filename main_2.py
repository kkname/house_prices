import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import copy

# 数据加载
train_data = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')

# 合并数据集以进行特征处理
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# ========== 特征工程 ==========
# 1. 添加组合特征
if 'TotalBsmtSF' in all_features.columns and '1stFlrSF' in all_features.columns and '2ndFlrSF' in all_features.columns:
    all_features['TotalSF'] = all_features['TotalBsmtSF'] + all_features['1stFlrSF'] + all_features['2ndFlrSF']

if 'FullBath' in all_features.columns and 'HalfBath' in all_features.columns and 'BsmtFullBath' in all_features.columns and 'BsmtHalfBath' in all_features.columns:
    all_features['TotalBathrooms'] = all_features['FullBath'] + 0.5 * all_features['HalfBath'] + \
                                     all_features['BsmtFullBath'] + 0.5 * all_features['BsmtHalfBath']

if 'PoolArea' in all_features.columns:
    all_features['HasPool'] = (all_features['PoolArea'] > 0).astype(int)

if 'GarageArea' in all_features.columns:
    all_features['HasGarage'] = (all_features['GarageArea'] > 0).astype(int)

if 'Fireplaces' in all_features.columns:
    all_features['HasFireplace'] = (all_features['Fireplaces'] > 0).astype(int)

if 'YearRemodAdd' in all_features.columns and 'YearBuilt' in all_features.columns:
    all_features['Remodeled'] = (all_features['YearRemodAdd'] > all_features['YearBuilt']).astype(int)
    all_features['HouseAge'] = 2020 - all_features['YearBuilt']  # 假设当前年份为2020
    all_features['RemodAge'] = 2020 - all_features['YearRemodAdd']

# 2. 对高度偏斜的数字特征进行对数变换
skewed_features = ['LotArea', 'GrLivArea', 'GarageArea']
for feature in skewed_features:
    if feature in all_features.columns:
        # 确保值为正（为了对数变换）
        if (all_features[feature] <= 0).any():
            all_features[feature] = all_features[feature] + 1  # 避免log(0)
        all_features[feature] = np.log1p(all_features[feature])

# 处理数值特征
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 处理类别特征
all_features = pd.get_dummies(all_features, dummy_na=True)

# 准备训练和测试数据
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

# 对标签进行对数变换，使其更接近正态分布
train_labels = torch.log1p(train_labels)

# 定义损失函数
loss = nn.MSELoss()
in_features = train_features.shape[1]


# ========== 优化的神经网络模型 ==========
def get_net():
    net = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1)
    )
    # 使用He初始化
    for layer in net.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    return net


def log_rmse(net, features, labels):
    # 预测值取指数，因为我们对标签进行了对数变换
    preds = net(features)
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(preds, -float('inf'), float('inf'))
    rmse = torch.sqrt(loss(clipped_preds, labels))
    return rmse.item()


# ========== 优化的训练函数 ==========
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = DataLoader(dataset, batch_size, shuffle=True)

    # 优化器
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # 早停机制
    best_test_loss = float('inf')
    patience = 20
    patience_counter = 0
    best_model = None

    for epoch in range(num_epochs):
        # 训练模式
        net.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            # 梯度裁剪，防止梯度爆炸
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
            optimizer.step()

        # 评估模式
        net.eval()
        with torch.no_grad():
            train_loss = log_rmse(net, train_features, train_labels)
            train_ls.append(train_loss)

            if test_labels is not None:
                test_loss = log_rmse(net, test_features, test_labels)
                test_ls.append(test_loss)

                # 更新学习率调度器
                scheduler.step(test_loss)

                # 早停检查
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    patience_counter = 0
                    # 保存最佳模型
                    best_model = copy.deepcopy(net.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= patience and epoch > 50:  # 至少训练50轮
                        print(f'Early stopping at epoch {epoch}')
                        # 恢复到最佳模型
                        net.load_state_dict(best_model)
                        return train_ls[:epoch - patience + 1], test_ls[:epoch - patience + 1]

    # 如果没有早停，恢复到最佳模型
    if best_model is not None:
        net.load_state_dict(best_model)
    return train_ls, test_ls


# ========== K折交叉验证 ==========
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    models = []

    for i in range(k):
        print(f'开始第 {i + 1} 折训练')
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        models.append(net)

        if i == 0:
            plt.figure(figsize=(10, 6))
            epochs = list(range(1, len(train_ls) + 1))
            plt.plot(epochs, train_ls, label='train')
            plt.plot(epochs, valid_ls, label='valid')
            plt.xlabel('epoch')
            plt.ylabel('log rmse')
            plt.xlim([1, len(train_ls)])
            plt.grid(True)
            plt.legend()
            plt.show()
        print(f'折 {i + 1}，训练 log rmse {float(train_ls[-1]):f}, '
              f'验证 log rmse {float(valid_ls[-1]):f}')

    print(f'{k}-折交叉验证: 平均训练 log rmse: {float(train_l_sum / k):f}, '
          f'平均验证 log rmse: {float(valid_l_sum / k):f}')

    return models


# ========== 集成预测 ==========
def ensemble_predict(models, X_test):
    # 创建与测试样本相同大小的零张量
    pred_sum = torch.zeros(X_test.shape[0], 1, dtype=torch.float32)

    # 对每个模型进行评估
    for model in models:
        model.eval()  # 设置为评估模式
        with torch.no_grad():
            pred = model(X_test)
            pred_sum += pred

    # 平均所有模型的预测
    return pred_sum / len(models)


# ========== 主函数 ==========
def main():
    # 设置随机种子以便复现结果
    torch.manual_seed(42)
    np.random.seed(42)

    # 超参数设置
    k = 5  # k折交叉验证
    num_epochs = 300
    learning_rate = 0.001
    weight_decay = 0.01
    batch_size = 32

    # 使用k折交叉验证训练多个模型
    print("开始训练集成模型...")
    models = k_fold(k, train_features, train_labels, num_epochs, learning_rate,
                    weight_decay, batch_size)

    # 使用集成模型进行预测
    print("生成测试集预测...")
    test_pred = ensemble_predict(models, test_features)

    # 将对数预测转换回原始尺度
    test_pred = torch.expm1(test_pred)

    # 将预测结果保存到CSV文件
    test_pred_np = test_pred.numpy()
    submission = pd.DataFrame()
    submission['Id'] = test_data.Id
    submission['SalePrice'] = test_pred_np
    submission.to_csv('submission.csv', index=False)
    print("预测完成，结果已保存到 submission.csv")

    # 绘制每个特征的重要性（仅针对第一个模型的第一层权重）
    try:
        feature_importance(models[0], all_features.columns[:train_features.shape[1]])
    except:
        print("无法绘制特征重要性")


# 可视化特征重要性
def feature_importance(model, feature_names):
    # 获取第一层的权重
    weights = model[0].weight.data

    # 计算每个特征的重要性（权重的绝对值平均）
    importance = weights.abs().mean(dim=0)

    # 转换为numpy数组
    importance = importance.numpy()

    # 创建特征名称和重要性的数据框
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })

    # 按重要性排序
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

    # 绘制前20个最重要的特征
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance_df['Feature'][:20], feature_importance_df['Importance'][:20])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()