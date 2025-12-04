"""
Simple unified testing pipeline for SVD novelty detection.

Usage:
    # Quick test with defaults
    python test_pipeline.py --dataset data/raw_data/dataset2_Faculty_entering

    # Full parameter sweep
    python test_pipeline.py --dataset data/raw_data/dataset2_Faculty_entering --full-sweep
    
    # Custom parameters
    python test_pipeline.py --dataset data/raw_data/dataset2_Faculty_entering --ranks 10,20 --thresholds 0.1,0.2

    # With mock AI (faster testing)
    python test_pipeline.py --dataset data/raw_data/dataset2_Faculty_entering --mock-ai
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Simple colored logging
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for better readability."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        record.levelname = f"{log_color}{record.levelname:8s}{reset}"
        return super().format(record)


def setup_logging():
    """Setup clean logging output."""
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter('%(levelname)s %(message)s'))
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler]
    )


def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_results_summary(output_dir):
    """Print summary of generated files."""
    output_path = Path(output_dir)
    
    print_header("RESULTS GENERATED")
    
    # Find all CSV files
    csv_files = {
        'Per-Frame Log': list(output_path.glob('per_frame_log_*.csv')),
        'Summary Metrics': list(output_path.glob('summary_metrics_*.csv')),
        'Baseline Metrics': list(output_path.glob('baseline_metrics_*.csv')),
        'Comparison Metrics': list(output_path.glob('comparison_metrics_*.csv')),
        'Parameter Sweep': list(output_path.glob('parameter_sweep_*.csv'))
    }
    
    for file_type, files in csv_files.items():
        if files:
            latest = sorted(files)[-1]
            print(f"  ✓ {file_type:<20} → {latest.name}")
    
    # Check for analysis reports
    if (output_path / 'PERFORMANCE_REPORT.md').exists():
        print(f"  ✓ {'Performance Report':<20} → PERFORMANCE_REPORT.md")
    
    print("\n" + "=" * 80)
    print(f"\n📁 All results saved to: {output_dir}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Simple testing pipeline for SVD novelty detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with defaults
  python test_pipeline.py --dataset data/raw_data/dataset2_Faculty_entering
  
  # Full parameter sweep (5 ranks × 5 thresholds = 25 configs)
  python test_pipeline.py --dataset data/raw_data/dataset2_Faculty_entering --full-sweep
  
  # Custom parameters
  python test_pipeline.py --dataset data/raw_data/dataset2_Faculty_entering --ranks 10,20,30 --thresholds 0.05,0.1,0.15
  
  # Fast testing with mock AI
  python test_pipeline.py --dataset data/raw_data/dataset2_Faculty_entering --mock-ai
        """
    )
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset folder (e.g., data/raw_data/dataset2_Faculty_entering)')
    
    # Optional arguments
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: results/<dataset_name>)')
    
    parser.add_argument('--ranks', type=str, default='10,30',
                        help='Comma-separated compression ranks (default: 10,30)')
    
    parser.add_argument('--thresholds', type=str, default='0.05,0.15',
                        help='Comma-separated thresholds (default: 0.05,0.15)')
    
    parser.add_argument('--full-sweep', action='store_true',
                        help='Run full parameter sweep (ranks: 10,20,30,40,50, thresholds: 0.01,0.05,0.10,0.15,0.20)')
    
    parser.add_argument('--mock-ai', action='store_true',
                        help='Use mock AI for faster testing (no actual YOLO inference)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Determine output directory
    if args.output is None:
        dataset_name = Path(args.dataset).name
        args.output = f"results/{dataset_name}"
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Parse parameters
    if args.full_sweep:
        ranks = [10, 20, 30, 40, 50]
        thresholds = [0.01, 0.05, 0.10, 0.15, 0.20]
    else:
        ranks = [int(r.strip()) for r in args.ranks.split(',')]
        thresholds = [float(t.strip()) for t in args.thresholds.split(',')]
    
    # Print configuration
    print_header("PIPELINE CONFIGURATION")
    print(f"  Dataset:         {args.dataset}")
    print(f"  Output:          {args.output}")
    print(f"  Ranks:           {ranks}")
    print(f"  Thresholds:      {thresholds}")
    print(f"  Total configs:   {len(ranks) * len(thresholds)}")
    print(f"  Mock AI:         {args.mock_ai}")
    print("=" * 80)
    
    # Import test suite
    from run_tests import TestSuite
    
    # Initialize test suite
    test_suite = TestSuite(
        dataset_path=args.dataset,
        output_dir=args.output,
        use_mock_ai=args.mock_ai
    )
    
    # Load ground truth
    print_header("LOADING GROUND TRUTH")
    ground_truth_loaded = test_suite.load_ground_truth()
    
    if not ground_truth_loaded:
        logger.warning("No ground truth found - quality metrics (precision/recall/F1) will not be available")
        logger.info(f"To enable quality metrics, create: {Path(args.dataset).name}_labels.csv")
    
    # Run tests
    print_header("RUNNING TESTS")
    logger.info(f"Starting test suite at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_suite.run_full_test_suite(
        compression_ranks=ranks,
        thresholds=thresholds
    )
    
    # Print results
    print_results_summary(args.output)
    
    # Generate analysis report if ground truth exists
    if ground_truth_loaded:
        print_header("GENERATING ANALYSIS REPORT")
        
        try:
            from analyze_results import load_ground_truth, load_per_frame_data, load_baseline
            from analyze_results import analyze_configuration, print_comparison
            
            # Find latest CSVs
            output_path = Path(args.output)
            per_frame_csv = sorted(output_path.glob('per_frame_log_*.csv'))[-1]
            baseline_csv = sorted(output_path.glob('baseline_metrics_*.csv'))[-1]
            
            # Load data
            dataset_name = Path(args.dataset).name
            labels_path = Path(args.dataset).parent / f"{dataset_name}_labels.csv"
            
            ground_truth = load_ground_truth(labels_path)
            per_frame_data = load_per_frame_data(per_frame_csv)
            baseline = load_baseline(baseline_csv)
            
            # Group by configuration
            configs = {}
            for frame_data in per_frame_data:
                key = (frame_data['compression_rank'], frame_data['threshold'])
                if key not in configs:
                    configs[key] = []
                configs[key].append(frame_data)
            
            # Analyze and find best
            results = []
            for (rank, threshold), config_data in sorted(configs.items()):
                analysis = analyze_configuration(config_data, ground_truth)
                if analysis:
                    analysis['rank'] = rank
                    analysis['threshold'] = threshold
                    results.append(analysis)
            
            # Print best configuration
            if results:
                best_f1 = max(results, key=lambda x: x['f1'])
                best_fps = max(results, key=lambda x: x['avg_fps'])
                
                print("\n" + "=" * 80)
                print("  BEST CONFIGURATIONS")
                print("=" * 80)
                print(f"\n  Best F1 Score: {best_f1['f1']:.4f}")
                print(f"    → Rank: {best_f1['rank']}, Threshold: {best_f1['threshold']:.4f}")
                print(f"    → Precision: {best_f1['precision']:.4f}, Recall: {best_f1['recall']:.4f}, Accuracy: {best_f1['accuracy']:.4f}")
                print(f"    → Reduction: {best_f1['reduction_pct']:.2f}%, FPS: {best_f1['avg_fps']:.2f}")
                
                print(f"\n  Best FPS: {best_fps['avg_fps']:.2f}")
                print(f"    → Rank: {best_fps['rank']}, Threshold: {best_fps['threshold']:.4f}")
                print(f"    → F1 Score: {best_fps['f1']:.4f}, Accuracy: {best_fps['accuracy']:.4f}")
                print(f"    → Reduction: {best_fps['reduction_pct']:.2f}%")
                print("\n" + "=" * 80 + "\n")
                
        except Exception as e:
            logger.error(f"Could not generate analysis report: {e}")
    
    # Final message
    print_header("PIPELINE COMPLETE")
    print(f"  ✓ Tests completed successfully")
    print(f"  ✓ Results saved to: {args.output}")
    print(f"  ✓ Total configurations tested: {len(ranks) * len(thresholds)}")
    
    if ground_truth_loaded:
        print(f"\n  📊 To view detailed analysis:")
        print(f"     python analyze_results.py --results {args.output}")
    else:
        print(f"\n  ⚠ To enable quality metrics:")
        print(f"     1. Create ground truth labels CSV")
        print(f"     2. Run: python create_ground_truth_template.py --dataset {args.dataset}")
        print(f"     3. Edit the CSV to mark novel frames")
        print(f"     4. Re-run this pipeline")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
