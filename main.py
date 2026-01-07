# -*- coding: utf-8 -*-
"""
眼动追踪注意力预测流水线 - 主程序入口

基于舒尔特方格任务的眼动轨迹数据，通过三层特征工程预测注意力评分
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DATA_DIR,
    OUTPUT_DIR,
    FEATURES_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    TARGET_TOLERANCE,
    XGBOOST_PARAMS,
    CV_SPLITS,
)
from src.data.loader import DataLoader
from src.data.preprocessor import GazePreprocessor
from src.segmentation.event_segmenter import AdaptiveSegmenter
from src.features.micro_features import MicroFeatureExtractor
from src.features.meso_features import MesoFeatureExtractor
from src.features.macro_features import MacroFeatureExtractor
from src.models.trainer import ModelTrainer
from src.models.explainer import SHAPExplainer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / 'pipeline.log', encoding='utf-8'),
    ],
)
logger = logging.getLogger(__name__)


def ensure_directories():
    """确保输出目录存在"""
    for dir_path in [OUTPUT_DIR, FEATURES_DIR, MODELS_DIR, FIGURES_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


def process_subject(
    subject_id: str,
    loader: DataLoader,
    preprocessor: GazePreprocessor,
    micro_extractor: MicroFeatureExtractor,
    meso_extractor: MesoFeatureExtractor,
    macro_extractor: MacroFeatureExtractor,
) -> dict:
    """
    处理单个被试的完整流水线

    Args:
        subject_id: 被试编号
        loader: 数据加载器
        preprocessor: 预处理器
        micro_extractor: 微观特征提取器
        meso_extractor: 中观特征提取器
        macro_extractor: 宏观特征提取器

    Returns:
        包含被试信息和宏观特征的字典
    """
    # 加载被试数据
    subject = loader.load_subject(subject_id)

    # 存储所有任务的中观特征
    meso_features_list = []

    for trial in subject.trials:
        # 预处理
        trial = preprocessor.preprocess_trial(trial)

        if not trial.raw_gaze_points:
            continue

        # 事件分割（使用自适应分割器从点击坐标学习布局）
        segmenter = AdaptiveSegmenter(
            task_config=trial.config,
            screen_width=SCREEN_WIDTH,
            screen_height=SCREEN_HEIGHT,
        )
        # 传递点击数据和眼动轨迹
        gaze_trajectory = getattr(trial, 'gaze_trajectory', None)
        segments = segmenter.segment(trial.raw_gaze_points, gaze_trajectory)
        trial.segments = segments

        if not segments:
            continue

        # 提取微观特征
        micro_features = [
            micro_extractor.extract(segment)
            for segment in segments
        ]

        # 提取中观特征
        meso_features = meso_extractor.extract(trial, micro_features)
        meso_features_list.append(meso_features)

    if not meso_features_list:
        return None

    # 提取宏观特征
    macro_features = macro_extractor.extract(subject, meso_features_list)

    return {
        'subject_id': subject_id,
        'total_score': subject.total_score,
        'category': subject.category,
        'features': macro_features,
        'trial_count': len(meso_features_list),
    }


def run_pipeline():
    """
    运行完整的分析流水线
    """
    logger.info("=" * 60)
    logger.info("眼动追踪注意力预测流水线")
    logger.info("=" * 60)

    # 1. 确保目录存在
    ensure_directories()

    # 2. 初始化组件
    logger.info("初始化组件...")
    loader = DataLoader(DATA_DIR)
    preprocessor = GazePreprocessor(
        screen_width=SCREEN_WIDTH,
        screen_height=SCREEN_HEIGHT,
    )
    micro_extractor = MicroFeatureExtractor(
        screen_width=SCREEN_WIDTH,
        screen_height=SCREEN_HEIGHT,
    )
    meso_extractor = MesoFeatureExtractor(
        target_tolerance=TARGET_TOLERANCE,
    )
    macro_extractor = MacroFeatureExtractor()

    # 3. 加载元数据
    logger.info("加载元数据...")
    try:
        loader.load_labels()
        loader.load_tasks()
    except FileNotFoundError as e:
        logger.error(f"加载元数据失败: {e}")
        return

    # 4. 获取被试列表
    subject_ids = loader.get_all_subject_ids()
    logger.info(f"找到 {len(subject_ids)} 个被试")

    # 5. 处理每个被试
    logger.info("处理被试数据...")
    all_results = []
    failed_subjects = []

    for subject_id in tqdm(subject_ids, desc="处理被试"):
        try:
            result = process_subject(
                subject_id,
                loader,
                preprocessor,
                micro_extractor,
                meso_extractor,
                macro_extractor,
            )
            if result is not None:
                all_results.append(result)
            else:
                failed_subjects.append(subject_id)
        except Exception as e:
            logger.warning(f"处理被试 {subject_id} 时出错: {e}")
            failed_subjects.append(subject_id)

    logger.info(f"成功处理 {len(all_results)} 个被试，失败 {len(failed_subjects)} 个")

    if not all_results:
        logger.error("没有成功处理任何被试，退出")
        return

    # 6. 构建特征矩阵
    logger.info("构建特征矩阵...")

    # 获取特征名称
    feature_names = list(all_results[0]['features'].keys())
    logger.info(f"特征数量: {len(feature_names)}")

    # 构建特征矩阵和标签
    X = np.array([
        [result['features'].get(name, 0.0) for name in feature_names]
        for result in all_results
    ])
    y = np.array([result['total_score'] for result in all_results])
    subject_ids_processed = [result['subject_id'] for result in all_results]

    # 保存特征
    features_df = pd.DataFrame(X, columns=feature_names)
    features_df['subject_id'] = subject_ids_processed
    features_df['total_score'] = y
    features_df['category'] = [result['category'] for result in all_results]
    features_df['trial_count'] = [result['trial_count'] for result in all_results]

    features_path = FEATURES_DIR / 'all_features.csv'
    features_df.to_csv(features_path, index=False, encoding='utf-8-sig')
    logger.info(f"特征已保存到: {features_path}")

    # 7. 模型训练
    logger.info("训练模型...")
    trainer = ModelTrainer(
        n_estimators=XGBOOST_PARAMS['n_estimators'],
        max_depth=XGBOOST_PARAMS['max_depth'],
        learning_rate=XGBOOST_PARAMS['learning_rate'],
        n_splits=CV_SPLITS,
        random_state=XGBOOST_PARAMS['random_state'],
    )

    metrics = trainer.train(X, y, feature_names)

    # 打印结果
    print("\n" + "=" * 60)
    print("模型评估结果")
    print("=" * 60)
    print(f"样本数量: {metrics['n_samples']}")
    print(f"特征数量: {metrics['n_features']}")
    print(f"R2 分数:  {metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}")
    print(f"MAE:      {metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f}")
    print(f"RMSE:     {metrics['rmse_mean']:.4f} ± {metrics['rmse_std']:.4f}")
    print("=" * 60)

    # 保存模型
    model_path = MODELS_DIR / 'xgboost_model.json'
    trainer.save_model(model_path)

    # 保存评估指标
    metrics_df = pd.DataFrame({
        'metric': ['r2_mean', 'r2_std', 'mae_mean', 'mae_std', 'rmse_mean', 'rmse_std'],
        'value': [
            metrics['r2_mean'], metrics['r2_std'],
            metrics['mae_mean'], metrics['mae_std'],
            metrics['rmse_mean'], metrics['rmse_std'],
        ],
    })
    metrics_df.to_csv(FEATURES_DIR / 'model_metrics.csv', index=False)

    # 8. SHAP 分析
    logger.info("进行 SHAP 可解释性分析...")
    try:
        explainer = SHAPExplainer(trainer.model, feature_names)
        explanation = explainer.explain(X)

        # 生成图表
        explainer.plot_summary(X, save_path=FIGURES_DIR / 'shap_summary.png')
        explainer.plot_bar(save_path=FIGURES_DIR / 'feature_importance.png')

        # 输出重要特征
        print("\n" + "=" * 60)
        print("特征重要性排名（Top 15）")
        print("=" * 60)
        for i, (name, importance) in enumerate(explainer.get_top_features(15)):
            print(f"{i + 1:2d}. {name:40s}: {importance:.4f}")
        print("=" * 60)

        # 保存特征重要性
        importance_df = pd.DataFrame(
            explanation['feature_importance'],
            columns=['feature', 'importance'],
        )
        importance_df.to_csv(
            FEATURES_DIR / 'feature_importance.csv',
            index=False,
            encoding='utf-8-sig',
        )

        # 为关键特征生成依赖图
        key_features = [
            'distraction_sensitivity',
            'fatigue_slope',
            'stability_score',
            'learning_rate',
        ]
        for feat in key_features:
            if feat in feature_names:
                explainer.plot_dependence(
                    feat, X,
                    save_path=FIGURES_DIR / f'dependence_{feat}.png',
                )

    except Exception as e:
        logger.warning(f"SHAP 分析失败: {e}")
        logger.info("继续保存其他结果...")

    # 9. 保存失败列表
    if failed_subjects:
        failed_df = pd.DataFrame({'subject_id': failed_subjects})
        failed_df.to_csv(FEATURES_DIR / 'failed_subjects.csv', index=False)

    logger.info("流水线执行完成！")
    print("\n输出文件：")
    print(f"  - 特征: {FEATURES_DIR / 'all_features.csv'}")
    print(f"  - 模型: {MODELS_DIR / 'xgboost_model.json'}")
    print(f"  - 图表: {FIGURES_DIR}")


def quick_test():
    """
    快速测试：只处理前 5 个被试
    """
    logger.info("执行快速测试...")

    ensure_directories()

    loader = DataLoader(DATA_DIR)
    preprocessor = GazePreprocessor()
    micro_extractor = MicroFeatureExtractor()
    meso_extractor = MesoFeatureExtractor()
    macro_extractor = MacroFeatureExtractor()

    try:
        loader.load_labels()
        loader.load_tasks()
    except FileNotFoundError as e:
        logger.error(f"加载元数据失败: {e}")
        return

    subject_ids = loader.get_all_subject_ids()[:5]
    logger.info(f"测试 {len(subject_ids)} 个被试")

    for subject_id in subject_ids:
        try:
            result = process_subject(
                subject_id,
                loader,
                preprocessor,
                micro_extractor,
                meso_extractor,
                macro_extractor,
            )
            if result:
                logger.info(
                    f"被试 {subject_id}: "
                    f"分数={result['total_score']:.2f}, "
                    f"类别={result['category']}, "
                    f"试次={result['trial_count']}, "
                    f"特征数={len(result['features'])}"
                )
        except Exception as e:
            logger.error(f"处理 {subject_id} 失败: {e}")

    logger.info("快速测试完成！")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='眼动追踪注意力预测流水线')
    parser.add_argument(
        '--test',
        action='store_true',
        help='执行快速测试（只处理前 5 个被试）',
    )
    args = parser.parse_args()

    if args.test:
        quick_test()
    else:
        run_pipeline()
