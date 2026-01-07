# -*- coding: utf-8 -*-
"""
SHAP 可解释性模块

提供模型的可解释性分析功能
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP 可解释性分析器

    使用 SHAP 值解释 XGBoost 模型预测
    """

    def __init__(
        self,
        model,
        feature_names: Optional[List[str]] = None,
    ):
        """
        初始化分析器

        Args:
            model: 训练好的 XGBoost 模型
            feature_names: 特征名称列表
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self.expected_value = None

    def explain(self, X: np.ndarray) -> Dict[str, Any]:
        """
        计算 SHAP 值

        Args:
            X: 特征矩阵

        Returns:
            包含 SHAP 值和特征重要性的字典
        """
        try:
            import shap
        except ImportError:
            logger.error("请安装 shap 库: pip install shap")
            return {}

        # 创建解释器
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(X)
        self.expected_value = self.explainer.expected_value

        # 计算特征重要性（平均绝对 SHAP 值）
        feature_importance = np.abs(self.shap_values).mean(axis=0)

        # 创建排名
        if self.feature_names is not None:
            importance_ranking = sorted(
                zip(self.feature_names, feature_importance),
                key=lambda x: x[1],
                reverse=True,
            )
        else:
            importance_ranking = sorted(
                [(f'feature_{i}', imp) for i, imp in enumerate(feature_importance)],
                key=lambda x: x[1],
                reverse=True,
            )

        logger.info("SHAP 分析完成")

        return {
            'shap_values': self.shap_values,
            'expected_value': self.expected_value,
            'feature_importance': importance_ranking,
            'importance_dict': dict(importance_ranking),
        }

    def plot_summary(
        self,
        X: np.ndarray,
        save_path: Optional[Union[str, Path]] = None,
        max_display: int = 20,
        figsize: Tuple[int, int] = (12, 8),
    ) -> None:
        """
        生成 SHAP 摘要图（beeswarm plot）

        展示：
        - 特征重要性排序
        - 特征值与 SHAP 值的关系

        Args:
            X: 特征矩阵
            save_path: 保存路径
            max_display: 最多显示的特征数
            figsize: 图像大小
        """
        try:
            import shap
        except ImportError:
            logger.error("请安装 shap 库: pip install shap")
            return

        if self.shap_values is None:
            self.explain(X)

        plt.figure(figsize=figsize)
        shap.summary_plot(
            self.shap_values,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
            logger.info(f"SHAP 摘要图已保存: {save_path}")

        plt.close()

    def plot_bar(
        self,
        save_path: Optional[Union[str, Path]] = None,
        max_display: int = 20,
        figsize: Tuple[int, int] = (10, 8),
    ) -> None:
        """
        生成特征重要性条形图

        Args:
            save_path: 保存路径
            max_display: 最多显示的特征数
            figsize: 图像大小
        """
        try:
            import shap
        except ImportError:
            logger.error("请安装 shap 库: pip install shap")
            return

        if self.shap_values is None:
            logger.error("请先调用 explain() 方法")
            return

        plt.figure(figsize=figsize)
        shap.summary_plot(
            self.shap_values,
            feature_names=self.feature_names,
            max_display=max_display,
            plot_type='bar',
            show=False,
        )

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
            logger.info(f"特征重要性图已保存: {save_path}")

        plt.close()

    def plot_dependence(
        self,
        feature_name: str,
        X: np.ndarray,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (10, 6),
    ) -> None:
        """
        生成特征依赖图

        展示单个特征与 SHAP 值的关系

        Args:
            feature_name: 特征名称
            X: 特征矩阵
            save_path: 保存路径
            figsize: 图像大小
        """
        try:
            import shap
        except ImportError:
            logger.error("请安装 shap 库: pip install shap")
            return

        if self.shap_values is None:
            self.explain(X)

        if self.feature_names is None:
            logger.error("需要提供特征名称")
            return

        try:
            feature_idx = self.feature_names.index(feature_name)
        except ValueError:
            logger.error(f"特征 {feature_name} 不存在")
            return

        plt.figure(figsize=figsize)
        shap.dependence_plot(
            feature_idx,
            self.shap_values,
            X,
            feature_names=self.feature_names,
            show=False,
        )

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
            logger.info(f"特征依赖图已保存: {save_path}")

        plt.close()

    def plot_waterfall(
        self,
        X: np.ndarray,
        sample_idx: int = 0,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (12, 8),
    ) -> None:
        """
        生成瀑布图

        展示单个样本的特征贡献

        Args:
            X: 特征矩阵
            sample_idx: 样本索引
            save_path: 保存路径
            figsize: 图像大小
        """
        try:
            import shap
        except ImportError:
            logger.error("请安装 shap 库: pip install shap")
            return

        if self.explainer is None:
            self.explain(X)

        # 获取单个样本的解释
        explanation = shap.Explanation(
            values=self.shap_values[sample_idx],
            base_values=self.expected_value,
            data=X[sample_idx],
            feature_names=self.feature_names,
        )

        plt.figure(figsize=figsize)
        shap.waterfall_plot(explanation, show=False)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
            logger.info(f"瀑布图已保存: {save_path}")

        plt.close()

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        获取前 N 个重要特征

        Args:
            n: 返回的特征数量

        Returns:
            (特征名, 重要性) 元组列表
        """
        if self.shap_values is None:
            logger.error("请先调用 explain() 方法")
            return []

        feature_importance = np.abs(self.shap_values).mean(axis=0)

        if self.feature_names is not None:
            ranking = sorted(
                zip(self.feature_names, feature_importance),
                key=lambda x: x[1],
                reverse=True,
            )
        else:
            ranking = sorted(
                [(f'feature_{i}', imp) for i, imp in enumerate(feature_importance)],
                key=lambda x: x[1],
                reverse=True,
            )

        return ranking[:n]

    def get_feature_interactions(
        self,
        X: np.ndarray,
        feature1: str,
        feature2: str,
    ) -> Dict[str, Any]:
        """
        分析两个特征之间的交互作用

        Args:
            X: 特征矩阵
            feature1: 第一个特征名
            feature2: 第二个特征名

        Returns:
            交互分析结果
        """
        try:
            import shap
        except ImportError:
            logger.error("请安装 shap 库: pip install shap")
            return {}

        if self.feature_names is None:
            logger.error("需要提供特征名称")
            return {}

        try:
            idx1 = self.feature_names.index(feature1)
            idx2 = self.feature_names.index(feature2)
        except ValueError as e:
            logger.error(f"特征不存在: {e}")
            return {}

        if self.shap_values is None:
            self.explain(X)

        # 计算交互强度（简化版本）
        correlation = np.corrcoef(
            self.shap_values[:, idx1],
            self.shap_values[:, idx2],
        )[0, 1]

        return {
            'feature1': feature1,
            'feature2': feature2,
            'shap_correlation': float(correlation),
        }
