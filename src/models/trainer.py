# -*- coding: utf-8 -*-
"""
模型训练模块

使用 XGBoost 进行回归预测，支持交叉验证
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    XGBoost 模型训练器

    支持交叉验证、超参数配置和模型保存
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_splits: int = 5,
        random_state: int = 42,
        use_scaling: bool = True,
    ):
        """
        初始化训练器

        Args:
            n_estimators: 树的数量
            max_depth: 树的最大深度
            learning_rate: 学习率
            n_splits: 交叉验证折数
            random_state: 随机种子
            use_scaling: 是否使用特征缩放
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_splits = n_splits
        self.random_state = random_state
        self.use_scaling = use_scaling

        self.model: Optional[XGBRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[List[str]] = None

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        训练模型（带交叉验证）

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标值 (n_samples,)
            feature_names: 特征名称列表

        Returns:
            交叉验证评估指标
        """
        self.feature_names = feature_names

        # 处理 NaN 和 Inf
        X = self._clean_features(X)

        # 特征缩放
        if self.use_scaling:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X

        # 创建分层标签用于交叉验证
        try:
            y_bins = pd.qcut(y, q=min(5, len(y) // 3), labels=False, duplicates='drop')
        except ValueError:
            # 如果无法分箱，使用普通 KFold
            y_bins = np.zeros(len(y))

        # 交叉验证
        if len(np.unique(y_bins)) > 1:
            kf = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            splits = kf.split(X_scaled, y_bins)
        else:
            kf = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            splits = kf.split(X_scaled)

        metrics = {
            'r2_scores': [],
            'mae_scores': [],
            'rmse_scores': [],
            'fold_predictions': [],
        }

        fold_models = []

        for fold, (train_idx, val_idx) in enumerate(splits):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0,
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            y_pred = model.predict(X_val)

            # 计算指标
            r2 = r2_score(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))

            metrics['r2_scores'].append(r2)
            metrics['mae_scores'].append(mae)
            metrics['rmse_scores'].append(rmse)
            metrics['fold_predictions'].append({
                'indices': val_idx,
                'y_true': y_val,
                'y_pred': y_pred,
            })

            fold_models.append(model)

            logger.info(f"Fold {fold + 1}: R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")

        # 在全数据上训练最终模型
        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
        )
        self.model.fit(X_scaled, y)

        # 汇总结果
        result = {
            'r2_mean': float(np.mean(metrics['r2_scores'])),
            'r2_std': float(np.std(metrics['r2_scores'])),
            'mae_mean': float(np.mean(metrics['mae_scores'])),
            'mae_std': float(np.std(metrics['mae_scores'])),
            'rmse_mean': float(np.mean(metrics['rmse_scores'])),
            'rmse_std': float(np.std(metrics['rmse_scores'])),
            'fold_metrics': metrics,
            'n_samples': len(y),
            'n_features': X.shape[1],
        }

        logger.info(
            f"交叉验证结果: R2={result['r2_mean']:.4f}±{result['r2_std']:.4f}, "
            f"MAE={result['mae_mean']:.4f}±{result['mae_std']:.4f}, "
            f"RMSE={result['rmse_mean']:.4f}±{result['rmse_std']:.4f}"
        )

        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测

        Args:
            X: 特征矩阵

        Returns:
            预测值
        """
        if self.model is None:
            raise ValueError("模型尚未训练")

        X = self._clean_features(X)

        if self.use_scaling and self.scaler is not None:
            X = self.scaler.transform(X)

        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性

        Returns:
            特征名称到重要性的映射
        """
        if self.model is None:
            raise ValueError("模型尚未训练")

        importance = self.model.feature_importances_

        if self.feature_names is not None:
            return dict(zip(self.feature_names, importance))
        else:
            return {f'feature_{i}': imp for i, imp in enumerate(importance)}

    def save_model(self, path: Union[str, Path]) -> None:
        """
        保存模型

        Args:
            path: 模型保存路径
        """
        if self.model is None:
            raise ValueError("模型尚未训练")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save_model(str(path))
        logger.info(f"模型已保存到: {path}")

    def load_model(self, path: Union[str, Path]) -> None:
        """
        加载模型

        Args:
            path: 模型路径
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")

        self.model = XGBRegressor()
        self.model.load_model(str(path))
        logger.info(f"模型已加载: {path}")

    @staticmethod
    def _clean_features(X: np.ndarray) -> np.ndarray:
        """
        清洗特征矩阵

        - 替换 NaN 为 0
        - 替换 Inf 为 0
        """
        X = np.array(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X


class HyperparameterTuner:
    """
    超参数调优器

    使用网格搜索进行超参数优化
    """

    def __init__(
        self,
        param_grid: Optional[Dict[str, List]] = None,
        n_splits: int = 5,
        random_state: int = 42,
    ):
        """
        初始化调优器

        Args:
            param_grid: 参数网格
            n_splits: 交叉验证折数
            random_state: 随机种子
        """
        self.param_grid = param_grid or {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
        }
        self.n_splits = n_splits
        self.random_state = random_state

        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = -np.inf
        self.results: List[Dict] = []

    def search(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """
        执行网格搜索

        Args:
            X: 特征矩阵
            y: 目标值

        Returns:
            最佳参数和分数
        """
        from itertools import product

        # 生成所有参数组合
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        combinations = list(product(*param_values))

        logger.info(f"开始网格搜索: {len(combinations)} 个参数组合")

        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))

            trainer = ModelTrainer(
                n_splits=self.n_splits,
                random_state=self.random_state,
                **params,
            )

            metrics = trainer.train(X, y)
            r2_mean = metrics['r2_mean']

            self.results.append({
                'params': params,
                'r2_mean': r2_mean,
                'r2_std': metrics['r2_std'],
                'mae_mean': metrics['mae_mean'],
                'rmse_mean': metrics['rmse_mean'],
            })

            if r2_mean > self.best_score:
                self.best_score = r2_mean
                self.best_params = params

            logger.info(
                f"[{i + 1}/{len(combinations)}] "
                f"params={params}, R2={r2_mean:.4f}"
            )

        logger.info(f"最佳参数: {self.best_params}, R2={self.best_score:.4f}")

        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.results,
        }
