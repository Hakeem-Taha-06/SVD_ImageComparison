"""
Enhanced results analysis script.
Re-analyzes per-frame data with ground truth to compute complete metrics
and generate clear performance comparison.

Usage:
    python analyze_results.py --results results/dataset2
"""

import argparse
import csv
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_ground_truth(labels_path):
    """Load ground truth labels with flexible column names."""
    labels = {}
    
    with open(labels_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Find column names
        frame_col = None
        novel_col = None
        
        for field in reader.fieldnames:
            if field.lower() in ['frame', 'frame_index', 'frame_idx']:
                frame_col = field
            if field.lower() in ['isnoveltruth', 'is_novel', 'novel', 'is_novel_truth']:
                novel_col = field
        
        if frame_col is None or novel_col is None:
            raise ValueError(f"CSV must have frame and novelty columns. Found: {reader.fieldnames}")
        
        for row in reader:
            frame_idx = int(row[frame_col])
            is_novel = int(row[novel_col])
            labels[frame_idx] = bool(is_novel)
    
    return labels


def load_per_frame_data(csv_path):
    """Load per-frame log data."""
    data = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'frame_index': int(row['frame_index']),
                'compression_rank': int(row['compression_rank']),
                'threshold': float(row['threshold']),
                'is_novel_predicted': bool(int(row['is_novel_predicted'])),
                'ai_called': bool(int(row['ai_called'])),
                'ai_e2e_ms': float(row['ai_e2e_ms']),
                'cpu_percent': float(row['cpu_percent']),
                'memory_mb': float(row['memory_mb'])
            })
    return data


def load_baseline(csv_path):
    """Load baseline metrics."""
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        baseline = next(reader)
        return {
            'total_frames': int(baseline['total_frames']),
            'baseline_fps': float(baseline['baseline_fps']),
            'baseline_ms_per_frame': float(baseline['baseline_ms_per_frame']),
            'baseline_cpu_percent': float(baseline['baseline_cpu_percent']),
            'baseline_memory_mb': float(baseline['baseline_memory_mb']),
            'baseline_total_ai_calls': int(baseline['baseline_total_ai_calls']),
            'baseline_total_time_ms': float(baseline['baseline_total_time_ms'])
        }


def calculate_metrics(predictions, ground_truth):
    """Calculate TP, TN, FP, FN, precision, recall, F1, accuracy."""
    tp = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
    tn = sum(1 for p, g in zip(predictions, ground_truth) if not p and not g)
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)
    fn = sum(1 for p, g in zip(predictions, ground_truth) if not p and g)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0.0
    
    return tp, tn, fp, fn, precision, recall, f1, accuracy


def analyze_configuration(config_data, ground_truth):
    """Analyze a single configuration."""
    predictions = []
    truths = []
    
    for frame_data in config_data:
        frame_idx = frame_data['frame_index']
        if frame_idx in ground_truth:
            predictions.append(frame_data['is_novel_predicted'])
            truths.append(ground_truth[frame_idx])
    
    if not predictions:
        return None
    
    tp, tn, fp, fn, precision, recall, f1, accuracy = calculate_metrics(predictions, truths)
    
    # Calculate aggregate metrics
    total_frames = len(config_data)
    novel_frames = sum(1 for d in config_data if d['is_novel_predicted'])
    ai_calls = sum(1 for d in config_data if d['ai_called'])
    
    avg_time_ms = sum(d['ai_e2e_ms'] for d in config_data) / total_frames
    avg_cpu = sum(d['cpu_percent'] for d in config_data) / total_frames
    avg_memory = sum(d['memory_mb'] for d in config_data) / total_frames
    
    return {
        'total_frames': total_frames,
        'novel_frames': novel_frames,
        'ai_calls': ai_calls,
        'forwarding_ratio': ai_calls / total_frames,
        'reduction_pct': 100 * (1 - ai_calls / total_frames),
        'avg_time_ms': avg_time_ms,
        'avg_fps': 1000 / avg_time_ms if avg_time_ms > 0 else 0,
        'avg_cpu': avg_cpu,
        'avg_memory': avg_memory,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }


def print_comparison(baseline, config, rank, threshold):
    """Print detailed comparison between filter and baseline."""
    print(f"\n{'=' * 80}")
    print(f"CONFIGURATION: Rank={rank}, Threshold={threshold:.4f}")
    print(f"{'=' * 80}")
    
    # Performance metrics
    print(f"\n{'PERFORMANCE METRICS':-^80}")
    print(f"{'Metric':<30} {'Baseline':<20} {'With Filter':<20} {'Improvement':<10}")
    print(f"{'-' * 80}")
    
    # FPS
    fps_improvement = ((config['avg_fps'] - baseline['baseline_fps']) / baseline['baseline_fps']) * 100
    print(f"{'FPS':<30} {baseline['baseline_fps']:<20.2f} {config['avg_fps']:<20.2f} {fps_improvement:>+9.2f}%")
    
    # ms per frame
    time_improvement = ((baseline['baseline_ms_per_frame'] - config['avg_time_ms']) / baseline['baseline_ms_per_frame']) * 100
    print(f"{'ms/frame':<30} {baseline['baseline_ms_per_frame']:<20.2f} {config['avg_time_ms']:<20.2f} {time_improvement:>+9.2f}%")
    
    # CPU
    cpu_saving = baseline['baseline_cpu_percent'] - config['avg_cpu']
    cpu_improvement = (cpu_saving / baseline['baseline_cpu_percent']) * 100
    print(f"{'CPU usage (%)':<30} {baseline['baseline_cpu_percent']:<20.2f} {config['avg_cpu']:<20.2f} {cpu_improvement:>+9.2f}%")
    
    # Memory
    memory_saving = baseline['baseline_memory_mb'] - config['avg_memory']
    memory_improvement = (memory_saving / baseline['baseline_memory_mb']) * 100
    print(f"{'Memory (MB)':<30} {baseline['baseline_memory_mb']:<20.2f} {config['avg_memory']:<20.2f} {memory_improvement:>+9.2f}%")
    
    # AI calls
    print(f"\n{'WORKLOAD REDUCTION':-^80}")
    print(f"{'Total AI Calls (Baseline):':<50} {baseline['baseline_total_ai_calls']}")
    print(f"{'Total AI Calls (With Filter):':<50} {config['ai_calls']}")
    print(f"{'AI Calls Avoided:':<50} {baseline['baseline_total_ai_calls'] - config['ai_calls']}")
    print(f"{'Reduction:':<50} {config['reduction_pct']:.2f}%")
    
    # Quality metrics
    print(f"\n{'DETECTION QUALITY METRICS':-^80}")
    print(f"{'True Positives (TP):':<50} {config['tp']} (Correctly detected novel frames)")
    print(f"{'True Negatives (TN):':<50} {config['tn']} (Correctly ignored non-novel frames)")
    print(f"{'False Positives (FP):':<50} {config['fp']} (Wrongly detected as novel)")
    print(f"{'False Negatives (FN):':<50} {config['fn']} (Missed novel frames)")
    print(f"{'-' * 80}")
    print(f"{'Precision:':<50} {config['precision']:.4f} (TP / (TP + FP))")
    print(f"{'Recall:':<50} {config['recall']:.4f} (TP / (TP + FN))")
    print(f"{'F1 Score:':<50} {config['f1']:.4f} (Harmonic mean of P and R)")
    print(f"{'Filter Accuracy:':<50} {config['accuracy']:.4f} ((TP + TN) / Total)")
    print(f"\n  ℹ Filter Accuracy measures how well the filter identifies novel vs non-novel frames.")
    print(f"    High accuracy = filter correctly decides which frames need AI processing.")
    
    # Overall savings
    total_time_baseline = baseline['baseline_total_time_ms']
    total_time_filter = config['avg_time_ms'] * config['total_frames']
    time_saved_ms = total_time_baseline - total_time_filter
    time_saved_pct = (time_saved_ms / total_time_baseline) * 100
    
    print(f"\n{'OVERALL SAVINGS':-^80}")
    print(f"{'Total Time (Baseline):':<50} {total_time_baseline:.2f} ms ({total_time_baseline/1000:.2f}s)")
    print(f"{'Total Time (With Filter):':<50} {total_time_filter:.2f} ms ({total_time_filter/1000:.2f}s)")
    print(f"{'Time Saved:':<50} {time_saved_ms:.2f} ms ({time_saved_ms/1000:.2f}s)")
    print(f"{'Time Saved (%):':<50} {time_saved_pct:.2f}%")
    
    print(f"\n{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results with ground truth')
    parser.add_argument('--results', type=str, required=True, help='Results directory')
    parser.add_argument('--dataset', type=str, help='Dataset path (to find labels)')
    parser.add_argument('--labels', type=str, help='Direct path to labels CSV')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results)
    
    # Find CSV files
    per_frame_files = list(results_dir.glob('per_frame_log_*.csv'))
    baseline_files = list(results_dir.glob('baseline_metrics_*.csv'))
    
    if not per_frame_files or not baseline_files:
        logger.error(f"Could not find required CSV files in {results_dir}")
        sys.exit(1)
    
    # Use most recent files
    per_frame_csv = sorted(per_frame_files)[-1]
    baseline_csv = sorted(baseline_files)[-1]
    
    logger.info(f"Using per-frame data: {per_frame_csv.name}")
    logger.info(f"Using baseline data: {baseline_csv.name}")
    
    # Load ground truth
    if args.labels:
        labels_path = Path(args.labels)
    elif args.dataset:
        dataset_path = Path(args.dataset)
        dataset_name = dataset_path.name
        # Look in data/labels/ folder
        labels_path = Path('data/labels') / f"{dataset_name}_labels.csv"
    else:
        # Try to infer from results directory name
        dataset_name = results_dir.name
        # Look in data/labels/ folder
        labels_path = Path('data/labels') / f"{dataset_name}_labels.csv"
    
    if not labels_path.exists():
        logger.error(f"Ground truth labels not found: {labels_path}")
        logger.info("Please specify --labels or --dataset argument")
        logger.info(f"Expected location: data/labels/{dataset_name}_labels.csv")
        sys.exit(1)
    
    logger.info(f"Loading ground truth: {labels_path}")
    ground_truth = load_ground_truth(labels_path)
    logger.info(f"Loaded {len(ground_truth)} ground truth labels")
    
    # Load data
    per_frame_data = load_per_frame_data(per_frame_csv)
    baseline = load_baseline(baseline_csv)
    
    logger.info(f"Loaded {len(per_frame_data)} per-frame records")
    
    # Group by configuration
    configs = {}
    for frame_data in per_frame_data:
        key = (frame_data['compression_rank'], frame_data['threshold'])
        if key not in configs:
            configs[key] = []
        configs[key].append(frame_data)
    
    logger.info(f"Found {len(configs)} configurations\n")
    
    # Analyze each configuration
    results = []
    for (rank, threshold), config_data in sorted(configs.items()):
        analysis = analyze_configuration(config_data, ground_truth)
        if analysis:
            analysis['rank'] = rank
            analysis['threshold'] = threshold
            results.append(analysis)
            print_comparison(baseline, analysis, rank, threshold)
    
    # Find best configurations
    if results:
        print(f"\n{'=' * 80}")
        print(f"{'BEST CONFIGURATIONS':^80}")
        print(f"{'=' * 80}\n")
        
        # Best F1 score
        best_f1 = max(results, key=lambda x: x['f1'])
        print(f"Best F1 Score: {best_f1['f1']:.4f}")
        print(f"  Rank: {best_f1['rank']}, Threshold: {best_f1['threshold']:.4f}")
        print(f"  Precision: {best_f1['precision']:.4f}, Recall: {best_f1['recall']:.4f}")
        print(f"  Reduction: {best_f1['reduction_pct']:.2f}%, FPS: {best_f1['avg_fps']:.2f}\n")
        
        # Best reduction with acceptable F1 (>0.7)
        good_results = [r for r in results if r['f1'] >= 0.7]
        if good_results:
            best_reduction = max(good_results, key=lambda x: x['reduction_pct'])
            print(f"Best Reduction (F1 >= 0.7): {best_reduction['reduction_pct']:.2f}%")
            print(f"  Rank: {best_reduction['rank']}, Threshold: {best_reduction['threshold']:.4f}")
            print(f"  F1 Score: {best_reduction['f1']:.4f}")
            print(f"  FPS: {best_reduction['avg_fps']:.2f}\n")
        
        # Best FPS
        best_fps = max(results, key=lambda x: x['avg_fps'])
        print(f"Best FPS: {best_fps['avg_fps']:.2f}")
        print(f"  Rank: {best_fps['rank']}, Threshold: {best_fps['threshold']:.4f}")
        print(f"  F1 Score: {best_fps['f1']:.4f}, Reduction: {best_fps['reduction_pct']:.2f}%\n")
        
        print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
