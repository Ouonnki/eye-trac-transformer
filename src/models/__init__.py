# -*- coding: utf-8 -*-
"""模型训练和解释模块"""

# 延迟导入，避免依赖问题
def __getattr__(name):
    if name == 'ModelTrainer':
        from .trainer import ModelTrainer
        return ModelTrainer
    elif name == 'SHAPExplainer':
        from .explainer import SHAPExplainer
        return SHAPExplainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['ModelTrainer', 'SHAPExplainer']
