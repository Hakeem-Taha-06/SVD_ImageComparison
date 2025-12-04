"""
Main testing script for SVD novelty detection evaluation.
Runs comprehensive experiments and generates all required metrics and CSV outputs.

Usage:
    python run_tests.py --dataset data/dataset1_test --output results/dataset1
    python run_tests.py --dataset data/dataset2_Faculty_entering --ranks 10,20,30 --thresholds 0.01,0.05,0.1
    python run_tests.py --dataset data/dataset3_evening_university_walk --grid-search --mock-ai
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from testing.baseline_runner import BaselineRunner
from testing.experiment_runner import ExperimentRunner
from testing.ground_truth_loader import GroundTruthLoader
from testing.metrics_calculator import MetricsCalculator, BaselineMetrics
from testing.csv_logger import CSVLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('testing.log')
    ]
)
logger = logging.getLogger(__name__)


class TestSuite:
    """Main test suite orchestrator."""
    
    def __init__(
        self,
        dataset_path: str,
        output_dir: str = "results",
        target_size: tuple = (240, 320),
        use_mock_ai: bool = False
    ):
        """
        Initialize test suite.
        
        Args:
            dataset_path: Path to dataset folder
            output_dir: Directory for output CSV files
            target_size: (height, width) for frame resizing
            use_mock_ai: Use mock AI instead of real YOLO
        """
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.target_size = target_size
        self.use_mock_ai = use_mock_ai
        
        # Initialize components
        self.baseline_runner = BaselineRunner(
            target_size=target_size,
            use_mock_ai=use_mock_ai
        )
        self.experiment_runner = ExperimentRunner(
            target_size=target_size,
            use_mock_ai=use_mock_ai
        )
        self.ground_truth_loader = GroundTruthLoader()
        self.metrics_calculator = MetricsCalculator()
        self.csv_logger = CSVLogger(output_dir=output_dir)
        
        logger.info("=" * 80)
        logger.info("TEST SUITE INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Dataset: {dataset_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Target size: {target_size}")
        logger.info(f"Using mock AI: {use_mock_ai}")
        logger.info("=" * 80)
    
    def load_ground_truth(self) -> bool:
        """
        Load ground truth labels if available.
        
        Returns:
            True if ground truth loaded successfully, False otherwise
        """
        try:
            labels = self.ground_truth_loader.load_from_dataset_folder(self.dataset_path)
            self.metrics_calculator.set_ground_truth(labels)
            self.experiment_runner.set_ground_truth(labels)
            
            stats = self.ground_truth_loader.get_statistics()
            logger.info(f"\nGround truth loaded successfully:")
            logger.info(f"  Dataset: {stats['dataset_name']}")
            logger.info(f"  Total frames: {stats['total_frames']}")
            logger.info(f"  Novel frames: {stats['novel_frames']}")
            logger.info(f"  Non-novel frames: {stats['non_novel_frames']}")
            logger.info(f"  Novel ratio: {stats['novel_ratio']:.2%}")
            
            return True
            
        except FileNotFoundError as e:
            logger.warning(f"Ground truth not found: {e}")
            logger.warning("Continuing without ground truth (quality metrics will be unavailable)")
            return False
        except Exception as e:
            logger.error(f"Error loading ground truth: {e}")
            return False
    
    def run_baseline(self) -> BaselineMetrics:
        """
        Run baseline measurements.
        
        Returns:
            BaselineMetrics instance
        """
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: BASELINE MEASUREMENT")
        logger.info("=" * 80)
        
        # Load dataset
        self.baseline_runner.load_dataset(self.dataset_path)
        
        # Run baseline
        baseline_data = self.baseline_runner.run_baseline()
        
        # Print summary
        self.baseline_runner.print_baseline_summary()
        
        # Calculate baseline metrics
        baseline_metrics = self.metrics_calculator.calculate_baseline_metrics(baseline_data)
        
        return baseline_metrics
    
    def run_experiments(
        self,
        compression_ranks: List[int],
        thresholds: List[float]
    ) -> List:
        """
        Run parameter sweep experiments.
        
        Args:
            compression_ranks: List of ranks to test
            thresholds: List of thresholds to test
            
        Returns:
            List of all experiment data
        """
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: PARAMETER SWEEP EXPERIMENTS")
        logger.info("=" * 80)
        
        # Load dataset
        self.experiment_runner.load_dataset(self.dataset_path)
        
        # Run parameter sweep
        experiment_data = self.experiment_runner.run_parameter_sweep(
            compression_ranks=compression_ranks,
            thresholds=thresholds
        )
        
        return experiment_data
    
    def analyze_results(
        self,
        experiment_data: List,
        baseline_metrics: BaselineMetrics
    ) -> dict:
        """
        Analyze experiment results and generate all metrics.
        
        Args:
            experiment_data: List of PerFrameMetrics from experiments
            baseline_metrics: BaselineMetrics instance
            
        Returns:
            Dictionary with all analysis results
        """
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 3: ANALYSIS")
        logger.info("=" * 80)
        
        # Set experiment data in calculator
        self.metrics_calculator.per_frame_data = experiment_data
        
        # Get unique configurations
        configs = set((m.compression_rank, m.threshold) for m in experiment_data)
        logger.info(f"Analyzing {len(configs)} configurations...")
        
        # Calculate summary metrics for each configuration
        summaries = []
        for rank, threshold in sorted(configs):
            summary = self.metrics_calculator.calculate_summary_metrics(rank, threshold)
            summaries.append(summary)
        
        # Calculate comparison metrics
        all_comparisons = []
        for summary in summaries:
            comparisons = self.metrics_calculator.calculate_comparison_metrics(
                summary, baseline_metrics
            )
            all_comparisons.extend(comparisons)
        
        logger.info(f"Generated {len(summaries)} summary metrics")
        logger.info(f"Generated {len(all_comparisons)} comparison metrics")
        
        return {
            'summaries': summaries,
            'comparisons': all_comparisons
        }
    
    def write_results(
        self,
        experiment_data: List,
        summaries: List,
        baseline_metrics: BaselineMetrics,
        comparisons: List
    ) -> dict:
        """
        Write all results to CSV files.
        
        Args:
            experiment_data: List of PerFrameMetrics
            summaries: List of SummaryMetrics
            baseline_metrics: BaselineMetrics
            comparisons: List of ComparisonMetrics
            
        Returns:
            Dictionary mapping CSV type to file path
        """
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 4: WRITING RESULTS")
        logger.info("=" * 80)
        
        output_files = self.csv_logger.write_all(
            per_frame_data=experiment_data,
            summaries=summaries,
            baseline=baseline_metrics,
            comparisons=comparisons
        )
        
        return output_files
    
    def print_summary(
        self,
        summaries: List,
        baseline_metrics: BaselineMetrics
    ) -> None:
        """
        Print comprehensive summary of results.
        
        Args:
            summaries: List of SummaryMetrics
            baseline_metrics: BaselineMetrics
        """
        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("=" * 80)
        
        # Baseline
        logger.info("\nBaseline Performance (No Filter):")
        logger.info(f"  FPS: {baseline_metrics.baseline_fps:.2f}")
        logger.info(f"  ms/frame: {baseline_metrics.baseline_ms_per_frame:.2f}")
        logger.info(f"  CPU: {baseline_metrics.baseline_cpu_percent:.1f}%")
        logger.info(f"  Memory: {baseline_metrics.baseline_memory_mb:.1f} MB")
        logger.info(f"  Total AI calls: {baseline_metrics.baseline_total_ai_calls}")
        
        # Find best configurations
        if summaries:
            # Best F1 score
            summaries_with_f1 = [s for s in summaries if s.f1_score is not None]
            if summaries_with_f1:
                best_f1 = max(summaries_with_f1, key=lambda s: s.f1_score)
                logger.info("\nBest Configuration by F1 Score:")
                logger.info(f"  Rank: {best_f1.compression_rank}, Threshold: {best_f1.threshold:.4f}")
                logger.info(f"  F1 Score: {best_f1.f1_score:.4f}")
                logger.info(f"  Precision: {best_f1.precision:.4f}")
                logger.info(f"  Recall: {best_f1.recall:.4f}")
                logger.info(f"  Accuracy: {best_f1.accuracy:.4f}")
                logger.info(f"  Reduction: {best_f1.reduction_percentage:.2f}%")
                logger.info(f"  FPS: {best_f1.avg_fps:.2f}")
            
            # Best reduction
            best_reduction = max(summaries, key=lambda s: s.reduction_percentage)
            logger.info("\nBest Configuration by Reduction:")
            logger.info(f"  Rank: {best_reduction.compression_rank}, Threshold: {best_reduction.threshold:.4f}")
            logger.info(f"  Reduction: {best_reduction.reduction_percentage:.2f}%")
            logger.info(f"  Novel frames: {best_reduction.novel_frames}/{best_reduction.total_frames}")
            if best_reduction.f1_score is not None:
                logger.info(f"  F1 Score: {best_reduction.f1_score:.4f}")
            logger.info(f"  FPS: {best_reduction.avg_fps:.2f}")
            
            # Best FPS
            best_fps = max(summaries, key=lambda s: s.avg_fps)
            logger.info("\nBest Configuration by FPS:")
            logger.info(f"  Rank: {best_fps.compression_rank}, Threshold: {best_fps.threshold:.4f}")
            logger.info(f"  FPS: {best_fps.avg_fps:.2f}")
            logger.info(f"  Reduction: {best_fps.reduction_percentage:.2f}%")
            if best_fps.f1_score is not None:
                logger.info(f"  F1 Score: {best_fps.f1_score:.4f}")
        
        logger.info("\n" + "=" * 80)
    
    def run_full_test_suite(
        self,
        compression_ranks: List[int],
        thresholds: List[float]
    ) -> dict:
        """
        Run complete test suite: baseline, experiments, analysis, output.
        
        Args:
            compression_ranks: List of ranks to test
            thresholds: List of thresholds to test
            
        Returns:
            Dictionary with all results and output file paths
        """
        # Load ground truth
        has_ground_truth = self.load_ground_truth()
        
        # Phase 1: Baseline
        baseline_metrics = self.run_baseline()
        
        # Phase 2: Experiments
        experiment_data = self.run_experiments(compression_ranks, thresholds)
        
        # Phase 3: Analysis
        analysis = self.analyze_results(experiment_data, baseline_metrics)
        
        # Phase 4: Write results
        output_files = self.write_results(
            experiment_data=experiment_data,
            summaries=analysis['summaries'],
            baseline_metrics=baseline_metrics,
            comparisons=analysis['comparisons']
        )
        
        # Print summary
        self.print_summary(analysis['summaries'], baseline_metrics)
        
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUITE COMPLETE")
        logger.info("=" * 80)
        
        return {
            'has_ground_truth': has_ground_truth,
            'baseline': baseline_metrics,
            'summaries': analysis['summaries'],
            'comparisons': analysis['comparisons'],
            'output_files': output_files
        }


def parse_list_arg(arg_str: str, parse_type=float) -> List:
    """Parse comma-separated list argument."""
    return [parse_type(x.strip()) for x in arg_str.split(',')]


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive SVD novelty detection tests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single dataset with default parameters
  python run_tests.py --dataset data/dataset1_test
  
  # Custom ranks and thresholds
  python run_tests.py --dataset data/dataset2_Faculty_entering --ranks 10,20,30 --thresholds 0.01,0.05,0.1
  
  # Grid search
  python run_tests.py --dataset data/dataset3 --grid-search --rank-min 5 --rank-max 50 --rank-step 5
  
  # With mock AI (no YOLO required)
  python run_tests.py --dataset data/dataset1_test --mock-ai
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to dataset folder (e.g., data/dataset1_test)'
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory for CSV files (default: results)'
    )
    
    # Parameter options
    parser.add_argument(
        '--ranks',
        type=str,
        default='10,20,30,40,50',
        help='Comma-separated list of compression ranks (default: 10,20,30,40,50)'
    )
    
    parser.add_argument(
        '--thresholds',
        type=str,
        default='0.01,0.05,0.1,0.15,0.2',
        help='Comma-separated list of thresholds (default: 0.01,0.05,0.1,0.15,0.2)'
    )
    
    # Grid search options
    parser.add_argument(
        '--grid-search',
        action='store_true',
        help='Use grid search instead of explicit lists'
    )
    
    parser.add_argument('--rank-min', type=int, default=5, help='Minimum rank for grid search')
    parser.add_argument('--rank-max', type=int, default=50, help='Maximum rank for grid search')
    parser.add_argument('--rank-step', type=int, default=10, help='Rank step for grid search')
    
    parser.add_argument('--threshold-min', type=float, default=0.01, help='Minimum threshold for grid search')
    parser.add_argument('--threshold-max', type=float, default=0.2, help='Maximum threshold for grid search')
    parser.add_argument('--threshold-step', type=float, default=0.05, help='Threshold step for grid search')
    
    # Other options
    parser.add_argument(
        '--size',
        type=str,
        default='240,320',
        help='Target frame size as height,width (default: 240,320)'
    )
    
    parser.add_argument(
        '--mock-ai',
        action='store_true',
        help='Use mock AI detection (no YOLO required)'
    )
    
    args = parser.parse_args()
    
    # Parse target size
    target_size = tuple(int(x.strip()) for x in args.size.split(','))
    
    # Initialize test suite
    test_suite = TestSuite(
        dataset_path=args.dataset,
        output_dir=args.output,
        target_size=target_size,
        use_mock_ai=args.mock_ai
    )
    
    # Determine parameters
    if args.grid_search:
        logger.info("Using grid search mode")
        # Generate parameter lists
        compression_ranks = list(range(args.rank_min, args.rank_max + 1, args.rank_step))
        
        thresholds = []
        threshold = args.threshold_min
        while threshold <= args.threshold_max:
            thresholds.append(round(threshold, 4))
            threshold += args.threshold_step
    else:
        compression_ranks = parse_list_arg(args.ranks, int)
        thresholds = parse_list_arg(args.thresholds, float)
    
    # Run full test suite
    try:
        results = test_suite.run_full_test_suite(
            compression_ranks=compression_ranks,
            thresholds=thresholds
        )
        
        logger.info("\n✓ All tests completed successfully")
        logger.info(f"✓ Results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
