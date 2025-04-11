import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 数据加载
train_data = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')

# 分离ID
train_ids = train_data['Id']
test_ids = test_data['Id']

# 合并数据集以进行特征处理
all_data = pd.concat([train_data.drop(['Id', 'SalePrice'], axis=1), 
                      test_data.drop(['Id'], axis=1)])

# ========== 特征工程 ==========
# 处理缺失值
all_data = all_data.fillna({
    'LotFrontage': all_data['LotFrontage'].median(),
    'MasVnrArea': 0,
    'GarageYrBlt': all_data['YearBuilt'],  # 假设车库与房屋同年建造
})

# 其余缺失值按类型填充
for col in all_data.columns:
    if all_data[col].dtype == 'object':
        all_data[col] = all_data[col].fillna('None')
    else:
        all_data[col] = all_data[col].fillna(0)

# 添加一些新特征
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBathrooms'] = all_data['FullBath'] + 0.5 * all_data['HalfBath'] + \
                           all_data['BsmtFullBath'] + 0.5 * all_data['BsmtHalfBath']
all_data['HasPool'] = (all_data['PoolArea'] > 0).astype(int)
all_data['HasGarage'] = (all_data['GarageArea'] > 0).astype(int)
all_data['HasFireplace'] = (all_data['Fireplaces'] > 0).astype(int)
all_data['Remodeled'] = (all_data['YearRemodAdd'] > all_data['YearBuilt']).astype(int)
all_data['HouseAge'] = 2020 - all_data['YearBuilt']
all_data['RemodAge'] = 2020 - all_data['YearRemodAdd']

# 类别特征编码 - 改为one-hot编码而不是category类型
categorical_cols = all_data.select_dtypes(include=['object']).columns
all_data = pd.get_dummies(all_data, columns=categorical_cols, drop_first=True)

# 分割回训练集和测试集
n_train = train_data.shape[0]
train_features = all_data[:n_train]
test_features = all_data[n_train:]

# 对目标变量进行对数转换
train_labels = np.log1p(train_data['SalePrice'])

# ========== XGBoost模型 ==========
def train_xgb_model(X, y, test_X, k=5, params=None):
    # 设置默认XGBoost参数
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'alpha': 0.1,  # L1正则化
            'lambda': 1.0,  # L2正则化
            'gamma': 0,     # 分裂所需的最小损失减少
            'seed': 42
        }
    
    # K折交叉验证
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # 存储每折的模型和评分
    models = []
    train_rmse_scores = []
    valid_rmse_scores = []
    feature_importance_df = pd.DataFrame()
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
        print(f"\n训练第 {fold+1} 折...")
        
        # 分割数据
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        # 创建XGBoost数据结构
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        
        # 训练模型
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=3000,
            evals=[(dtrain, 'train'), (dvalid, 'valid')],
            early_stopping_rounds=100,
            verbose_eval=100
        )
        
        # 记录模型和分数
        models.append(model)
        
        # 训练集和验证集预测
        train_preds = model.predict(dtrain)
        valid_preds = model.predict(dvalid)
        
        # 计算RMSE
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_preds))
        
        print(f"第 {fold+1} 折 - 训练 RMSE: {train_rmse:.6f}, 验证 RMSE: {valid_rmse:.6f}")
        train_rmse_scores.append(train_rmse)
        valid_rmse_scores.append(valid_rmse)
        
        # 特征重要性
        fold_importance = pd.DataFrame()
        fold_importance['Feature'] = model.get_score(importance_type='gain').keys()
        fold_importance['Importance'] = model.get_score(importance_type='gain').values()
        fold_importance['Fold'] = fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance], axis=0)
    
    # 输出平均分数
    print(f"\n{k}折交叉验证结果:")
    print(f"平均训练 RMSE: {np.mean(train_rmse_scores):.6f} (±{np.std(train_rmse_scores):.6f})")
    print(f"平均验证 RMSE: {np.mean(valid_rmse_scores):.6f} (±{np.std(valid_rmse_scores):.6f})")
    
    # 可视化特征重要性
    try:
        plot_feature_importance(feature_importance_df)
    except Exception as e:
        print(f"绘制特征重要性时出错: {e}")
    
    # 使用所有模型进行测试集预测
    dtest = xgb.DMatrix(test_X)
    test_preds = np.zeros(test_X.shape[0])
    for model in models:
        test_preds += model.predict(dtest) / k
    
    return test_preds, models, feature_importance_df

def plot_feature_importance(feature_importance_df):
    # 计算每个特征的平均重要性
    mean_importance = feature_importance_df.groupby('Feature')['Importance'].mean().reset_index()
    mean_importance = mean_importance.sort_values('Importance', ascending=False).head(20)
    
    # 绘制特征重要性
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(mean_importance)), mean_importance['Importance'])
    plt.yticks(range(len(mean_importance)), mean_importance['Feature'])
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.show()

# 模型训练和预测
print("开始XGBoost模型训练...")
test_preds, models, feature_importance = train_xgb_model(
    train_features, train_labels, test_features
)

# 将预测转回原始尺度并创建提交文件
test_preds_original = np.expm1(test_preds)
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': test_preds_original
})
submission.to_csv('xgboost_submission.csv', index=False)
print("预测已保存到 xgboost_submission.csv")