# -*- coding: utf-8 -*-
"""
数据诊断脚本

追踪数据流向，定位片段数量膨胀的具体位置。
不预设是重复点击，而是追踪每个环节的数量变化。

使用方法：
    python scripts/diagnose_clicks.py --data_dir data/gaze_trajectory_data
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import DataLoader as GazeDataLoader
from src.data.schemas import ClickPoint
from src.data.preprocessor import GazePreprocessor
from src.segmentation.event_segmenter import AdaptiveSegmenter


def analyze_raw_excel(data_path: Path, subject_id: str, task_id: int) -> Dict:
    """分析原始 Excel 文件的 sheet3 行数"""
    file_path = data_path / str(subject_id) / f'{task_id}.xlsx'
    if not file_path.exists():
        return None

    try:
        click_df = pd.read_excel(file_path, sheet_name='sheet3')
        gaze_df = pd.read_excel(file_path, sheet_name='sheet4')

        return {
            'sheet3_rows': len(click_df),
            'sheet4_rows': len(gaze_df),
            'sheet3_columns': list(click_df.columns),
        }
    except Exception as e:
        return {'error': str(e)}


def diagnose_single_subject(
    subject_id: str,
    loader: GazeDataLoader,
    preprocessor: GazePreprocessor,
    data_path: Path,
    screen_width: int = 1920,
    screen_height: int = 1080,
) -> Dict:
    """
    诊断单个被试的数据流向

    追踪每个环节的数量：
    1. 原始 Excel sheet3 行数
    2. 预处理后点击点数量
    3. 分割后片段数量
    """
    result = {
        'subject_id': subject_id,
        'tasks': [],
        'total_sheet3_rows': 0,
        'total_click_points': 0,
        'total_segments': 0,
        'errors': [],
    }

    try:
        subject = loader.load_subject(subject_id)
    except Exception as e:
        result['errors'].append(f'加载被试失败: {e}')
        return result

    for trial in subject.trials:
        task_info = {
            'task_id': trial.task_id,
            'sheet3_rows': 0,
            'click_points_before_preprocess': 0,
            'click_points_after_preprocess': 0,
            'segments': 0,
            'avg_points_per_segment': 0,
        }

        # 1. 分析原始 Excel
        raw_info = analyze_raw_excel(data_path, subject_id, trial.task_id)
        if raw_info and 'error' not in raw_info:
            task_info['sheet3_rows'] = raw_info['sheet3_rows']
            task_info['sheet4_rows'] = raw_info.get('sheet4_rows', 0)
            result['total_sheet3_rows'] += raw_info['sheet3_rows']

        # 2. 预处理
        preprocessor.preprocess_trial(trial)

        # 获取点击点数量（点击和眼动轨迹现在分离存储）
        click_points = trial.clicks
        if click_points:
            task_info['click_points_after_preprocess'] = len(click_points)
            result['total_click_points'] += len(click_points)

        # 获取眼动轨迹
        gaze_trajectory = trial.gaze_points

        if not click_points:
            result['tasks'].append(task_info)
            continue

        # 3. 分割：第一个参数是点击事件，第二个参数是眼动轨迹
        segmenter = AdaptiveSegmenter(
            task_config=trial.config,
            screen_width=screen_width,
            screen_height=screen_height,
        )
        segments = segmenter.segment(click_points, gaze_trajectory)

        task_info['segments'] = len(segments)
        result['total_segments'] += len(segments)

        if segments:
            task_info['avg_points_per_segment'] = sum(
                len(s.gaze_points) for s in segments
            ) / len(segments)

        result['tasks'].append(task_info)

    return result


def print_diagnosis_report(results: List[Dict], output_path: str = None):
    """打印诊断报告"""
    total_subjects = len(results)
    total_sheet3 = sum(r['total_sheet3_rows'] for r in results)
    total_clicks = sum(r['total_click_points'] for r in results)
    total_segments = sum(r['total_segments'] for r in results)

    report = []
    report.append('=' * 70)
    report.append('数据流向诊断报告')
    report.append('=' * 70)

    report.append(f'\n总体统计:')
    report.append(f'  被试数量: {total_subjects}')
    report.append(f'  原始 sheet3 总行数: {total_sheet3}')
    report.append(f'  预处理后点击点总数: {total_clicks}')
    report.append(f'  分割后片段总数: {total_segments}')

    # 计算膨胀比例
    if total_sheet3 > 0:
        click_ratio = total_clicks / total_sheet3
        segment_ratio = total_segments / total_sheet3
        report.append(f'\n膨胀比例分析:')
        report.append(f'  点击点/sheet3行数: {click_ratio:.2f}x')
        report.append(f'  片段数/sheet3行数: {segment_ratio:.2f}x')

    # 预期值对比
    expected_clicks_per_subject = 30 * 25  # 30个任务，每个25次点击
    expected_segments_per_subject = 30 * 24  # 30个任务，每个24个片段
    expected_total_clicks = expected_clicks_per_subject * total_subjects
    expected_total_segments = expected_segments_per_subject * total_subjects

    report.append(f'\n与预期值对比:')
    report.append(f'  预期点击总数 (30任务×25点击): {expected_total_clicks}')
    report.append(f'  实际点击总数: {total_clicks} ({total_clicks/expected_total_clicks:.1f}x)')
    report.append(f'  预期片段总数 (30任务×24片段): {expected_total_segments}')
    report.append(f'  实际片段总数: {total_segments} ({total_segments/expected_total_segments:.1f}x)')

    # 分析每个被试的情况
    report.append(f'\n按被试统计 (显示异常情况):')
    report.append('-' * 70)

    anomalies = []
    for r in results:
        avg_segments = r['total_segments'] / max(len(r['tasks']), 1)
        expected_segments = 24  # 每个任务预期24个片段

        if avg_segments > expected_segments * 2:  # 超过预期2倍
            anomalies.append({
                'subject_id': r['subject_id'],
                'tasks': len(r['tasks']),
                'total_sheet3': r['total_sheet3_rows'],
                'total_clicks': r['total_click_points'],
                'total_segments': r['total_segments'],
                'avg_segments_per_task': avg_segments,
            })

    if anomalies:
        for a in anomalies[:10]:  # 只显示前10个
            report.append(
                f"  {a['subject_id']}: sheet3={a['total_sheet3']}, "
                f"clicks={a['total_clicks']}, segments={a['total_segments']}, "
                f"avg/task={a['avg_segments_per_task']:.1f}"
            )
        if len(anomalies) > 10:
            report.append(f'  ... 还有 {len(anomalies) - 10} 个异常被试')
    else:
        report.append('  未发现明显异常')

    # 分析任务级别的模式
    report.append(f'\n按任务统计:')
    report.append('-' * 70)

    task_stats = defaultdict(lambda: {'sheet3': [], 'clicks': [], 'segments': []})
    for r in results:
        for t in r['tasks']:
            task_id = t['task_id']
            task_stats[task_id]['sheet3'].append(t['sheet3_rows'])
            task_stats[task_id]['clicks'].append(t['click_points_after_preprocess'])
            task_stats[task_id]['segments'].append(t['segments'])

    # 找出片段数异常的任务
    abnormal_tasks = []
    for task_id, stats in sorted(task_stats.items()):
        avg_segments = sum(stats['segments']) / max(len(stats['segments']), 1)
        avg_sheet3 = sum(stats['sheet3']) / max(len(stats['sheet3']), 1)

        if avg_segments > 50:  # 每个任务预期最多24个片段
            abnormal_tasks.append({
                'task_id': task_id,
                'avg_sheet3': avg_sheet3,
                'avg_segments': avg_segments,
            })

    if abnormal_tasks:
        report.append('  片段数异常的任务:')
        for t in abnormal_tasks[:5]:
            report.append(
                f"    任务{t['task_id']}: avg_sheet3={t['avg_sheet3']:.1f}, "
                f"avg_segments={t['avg_segments']:.1f}"
            )
    else:
        report.append('  所有任务的片段数正常')

    # 定位问题环节
    report.append(f'\n问题定位:')
    report.append('-' * 70)

    if total_clicks > total_sheet3 * 1.1:
        report.append('  [!] 预处理阶段: 点击点数量 > sheet3行数，可能存在重复标记')
    elif total_clicks < total_sheet3 * 0.9:
        report.append('  [!] 预处理阶段: 点击点数量 < sheet3行数，可能有数据丢失')
    else:
        report.append('  [OK] 预处理阶段: 点击点数量与sheet3行数基本一致')

    if total_segments > total_clicks:
        report.append('  [!] 分割阶段: 片段数 > 点击数，这在理论上不应该发生')
    elif total_segments > total_clicks * 0.9:
        report.append('  [OK] 分割阶段: 片段数约等于点击数-1，符合预期')
    else:
        report.append('  [!] 分割阶段: 片段数明显少于点击数，可能有过滤逻辑')

    # 推荐的下一步
    report.append(f'\n建议:')
    report.append('-' * 70)

    if total_sheet3 > expected_total_clicks * 10:
        report.append('  1. 检查原始Excel的sheet3格式，确认是否只包含点击事件')
        report.append('  2. 可能sheet3包含了连续眼动数据而非仅点击事件')
    elif total_segments > expected_total_segments * 10:
        report.append('  1. 检查分割逻辑是否正确')
        report.append('  2. 考虑添加最小片段长度过滤')
    else:
        report.append('  数据流向基本正常')

    report.append('\n' + '=' * 70)

    # 输出
    report_text = '\n'.join(report)
    print(report_text)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f'\n报告已保存到: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='数据流向诊断')
    parser.add_argument('--data_dir', type=str, default='data/gaze_trajectory_data',
                        help='原始数据目录')
    parser.add_argument('--output', type=str, default='outputs/diagnosis_report.txt',
                        help='输出报告路径')
    parser.add_argument('--max_subjects', type=int, default=0,
                        help='最大诊断被试数（0表示全部）')
    parser.add_argument('--screen_width', type=int, default=1920)
    parser.add_argument('--screen_height', type=int, default=1080)
    args = parser.parse_args()

    print('=' * 60)
    print('数据流向诊断')
    print('=' * 60)
    print(f'数据目录: {args.data_dir}')

    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f'错误: 数据目录不存在: {data_path}')
        return

    # 创建输出目录
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 初始化
    loader = GazeDataLoader(args.data_dir)
    loader.load_labels()
    loader.load_tasks()
    preprocessor = GazePreprocessor(
        screen_width=args.screen_width,
        screen_height=args.screen_height
    )

    subject_ids = loader.get_all_subject_ids()
    if args.max_subjects > 0:
        subject_ids = subject_ids[:args.max_subjects]

    print(f'将诊断 {len(subject_ids)} 个被试\n')

    # 诊断
    results = []
    for subject_id in tqdm(subject_ids, desc='诊断进度'):
        result = diagnose_single_subject(
            subject_id, loader, preprocessor, data_path,
            args.screen_width, args.screen_height
        )
        results.append(result)

    # 输出报告
    print_diagnosis_report(results, str(output_path))


if __name__ == '__main__':
    main()
