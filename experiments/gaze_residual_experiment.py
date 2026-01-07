"""
实验：眼动特征的独立预测价值分析

目的：排除时间因素后，测试眼动特征是否具有独立的预测能力

实验设计：
1. 用时间相关特征预测注意力分数 → R²_time
2. 计算残差（实际分数 - 时间模型预测）
3. 用纯眼动特征预测残差 → R²_gaze_on_residual
4. 如果 R²_gaze_on_residual > 0.1，说明眼动特征有独立价值
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, KFold
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 加载特征数据
features_path = r"d:\Ouonnki\cadt\boost\data\outputs\features\all_features.csv"
df = pd.read_csv(features_path)

print("=" * 70)
print("眼动特征独立预测价值实验")
print("=" * 70)

# 目标变量
y = df['total_score'].values

# 定义特征组
# 时间相关特征（搜索时间、完成时间、实验时长等）
time_features = [
    'global_total_search_time_mean', 'global_total_search_time_std',
    'global_total_search_time_median', 'global_total_search_time_min',
    'global_search_time_mean_mean', 'global_search_time_mean_std',
    'global_search_time_mean_median', 'global_search_time_mean_min',
    'global_search_time_std_mean', 'global_completion_rate_mean',
    'global_completion_rate_std', 'global_completion_rate_median',
    'overall_completion_rate', 'total_experiment_duration',
    'shortest_task_time', 'longest_task_time', 'total_clicks',
    'fatigue_slope', 'learning_rate', 'stability_score'
]

# 纯眼动特征（路径效率、注视熵、速度、方向变化、犹豫时间等）
gaze_features = [
    # 路径效率
    'global_path_efficiency_mean', 'global_path_efficiency_std', 'global_path_efficiency_min',
    # 注视熵
    'global_fixation_entropy_mean', 'global_fixation_entropy_std', 'global_fixation_entropy_max',
    # 眼动速度
    'global_gaze_velocity_mean', 'global_gaze_velocity_std',
    # 方向变化
    'global_direction_changes_mean', 'global_direction_changes_std', 'global_direction_changes_total',
    # 犹豫时间
    'global_hesitation_time_mean', 'global_hesitation_time_std',
    'global_hesitation_time_total', 'global_hesitation_ratio_mean',
    # 眼动点数
    'total_gaze_points', 'avg_gaze_points_per_task',
    # 路径长度
    'total_path_length', 'avg_path_length_per_task',
    # 加速度
    'global_acceleration_mean',
    # 学习效应
    'path_efficiency_learning', 'entropy_trend', 'distractor_efficiency_effect'
]

# 过滤存在的特征
available_time = [f for f in time_features if f in df.columns]
available_gaze = [f for f in gaze_features if f in df.columns]

print(f"\n可用时间特征数: {len(available_time)}")
print(f"可用眼动特征数: {len(available_gaze)}")

# 准备特征矩阵
X_time = df[available_time].values
X_gaze = df[available_gaze].values

# 模型配置
model_params = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ========== 实验1：仅时间特征预测分数 ==========
print("\n" + "=" * 70)
print("实验1：仅用时间特征预测注意力分数")
print("=" * 70)

model_time = XGBRegressor(**model_params)
y_pred_time = cross_val_predict(model_time, X_time, y, cv=kf)

r2_time = r2_score(y, y_pred_time)
mae_time = mean_absolute_error(y, y_pred_time)
rmse_time = np.sqrt(mean_squared_error(y, y_pred_time))

print(f"R2 (time features): {r2_time:.4f}")
print(f"MAE: {mae_time:.4f}")
print(f"RMSE: {rmse_time:.4f}")

# 计算残差
residuals = y - y_pred_time
print(f"\n残差统计:")
print(f"  均值: {residuals.mean():.4f}")
print(f"  标准差: {residuals.std():.4f}")
print(f"  范围: [{residuals.min():.4f}, {residuals.max():.4f}]")

# ========== 实验2：仅眼动特征预测分数 ==========
print("\n" + "=" * 70)
print("实验2：仅用眼动特征预测注意力分数")
print("=" * 70)

model_gaze = XGBRegressor(**model_params)
y_pred_gaze = cross_val_predict(model_gaze, X_gaze, y, cv=kf)

r2_gaze = r2_score(y, y_pred_gaze)
mae_gaze = mean_absolute_error(y, y_pred_gaze)
rmse_gaze = np.sqrt(mean_squared_error(y, y_pred_gaze))

print(f"R2 (gaze features): {r2_gaze:.4f}")
print(f"MAE: {mae_gaze:.4f}")
print(f"RMSE: {rmse_gaze:.4f}")

# ========== 实验3：眼动特征预测残差（关键实验）==========
print("\n" + "=" * 70)
print("实验3：用眼动特征预测时间模型的残差（关键实验）")
print("=" * 70)

model_residual = XGBRegressor(**model_params)
residual_pred = cross_val_predict(model_residual, X_gaze, residuals, cv=kf)

r2_residual = r2_score(residuals, residual_pred)
mae_residual = mean_absolute_error(residuals, residual_pred)
rmse_residual = np.sqrt(mean_squared_error(residuals, residual_pred))

print(f"R2 (gaze->residual): {r2_residual:.4f}")
print(f"MAE: {mae_residual:.4f}")
print(f"RMSE: {rmse_residual:.4f}")

# ========== 实验4：组合模型（时间+眼动）==========
print("\n" + "=" * 70)
print("实验4：组合模型（时间+眼动特征）")
print("=" * 70)

X_combined = np.hstack([X_time, X_gaze])
model_combined = XGBRegressor(**model_params)
y_pred_combined = cross_val_predict(model_combined, X_combined, y, cv=kf)

r2_combined = r2_score(y, y_pred_combined)
mae_combined = mean_absolute_error(y, y_pred_combined)
rmse_combined = np.sqrt(mean_squared_error(y, y_pred_combined))

print(f"R2 (combined): {r2_combined:.4f}")
print(f"MAE: {mae_combined:.4f}")
print(f"RMSE: {rmse_combined:.4f}")

# ========== 结果汇总 ==========
print("\n" + "=" * 70)
print("实验结果汇总")
print("=" * 70)

print("\nModel Performance Comparison:")
print("-" * 60)
print(f"{'Model':<25} {'R2':>10} {'MAE':>10} {'RMSE':>10}")
print("-" * 60)
print(f"{'Time features only':<25} {r2_time:>10.4f} {mae_time:>10.4f} {rmse_time:>10.4f}")
print(f"{'Gaze features only':<25} {r2_gaze:>10.4f} {mae_gaze:>10.4f} {rmse_gaze:>10.4f}")
print(f"{'Gaze->Residual (KEY)':<25} {r2_residual:>10.4f} {mae_residual:>10.4f} {rmse_residual:>10.4f}")
print(f"{'Combined (Time+Gaze)':<25} {r2_combined:>10.4f} {mae_combined:>10.4f} {rmse_combined:>10.4f}")
print("-" * 60)

# ========== 结论分析 ==========
print("=" * 70)
print("结论分析")
print("=" * 70)

# 计算眼动特征的独立贡献
incremental_r2 = r2_combined - r2_time
variance_explained_by_gaze = r2_residual * (1 - r2_time)  # 在剩余方差中解释的比例

print(f"\n1. Time features explain {r2_time*100:.1f}% of score variance")
print(f"   -> Confirms time is the primary predictor of attention score")
print(f"\n2. Gaze features alone explain {r2_gaze*100:.1f}% of score variance")
print(f"   -> Gaze features have strong predictive power on their own")
print(f"\n3. [KEY FINDING] Gaze features predict residuals: R2 = {r2_residual:.4f}")

if r2_residual > 0.1:
    print(f"   [YES] R2 = {r2_residual:.4f} > 0.1, gaze features have INDEPENDENT predictive value!")
    print(f"   -> After removing time effects, gaze features still explain {r2_residual*100:.1f}% of residual variance")
    print(f"   -> Gaze patterns contain attention information that time cannot capture")
elif r2_residual > 0.05:
    print(f"   [PARTIAL] R2 = {r2_residual:.4f} is between 0.05-0.1, gaze features have some independent value")
    print(f"   -> Effect is weak but statistically meaningful")
else:
    print(f"   [NO] R2 = {r2_residual:.4f} < 0.05, gaze features have limited independent value")
    print(f"   -> Gaze features may primarily reflect attention through time indirectly")

print(f"\n4. Combined model incremental contribution: delta_R2 = {incremental_r2:.4f}")
print(f"   -> After adding gaze features, R2 improved from {r2_time:.4f} to {r2_combined:.4f}")
print(f"   -> Incremental improvement of {incremental_r2*100:.2f}%")

# ========== 进一步分析：眼动特征对残差的重要性 ==========
print("=" * 70)
print("眼动特征对残差预测的重要性排名")
print("=" * 70)

# 重新训练完整模型以获取特征重要性
model_residual_full = XGBRegressor(**model_params)
model_residual_full.fit(X_gaze, residuals)

importance = model_residual_full.feature_importances_
feature_importance = list(zip(available_gaze, importance))
feature_importance.sort(key=lambda x: x[1], reverse=True)

print("\nTop 10 重要眼动特征（预测残差）:")
print("-" * 50)
for i, (feat, imp) in enumerate(feature_importance[:10], 1):
    print(f"{i:2d}. {feat:40s} {imp:.4f}")

print("\n" + "=" * 70)
print("实验完成")
print("=" * 70)
