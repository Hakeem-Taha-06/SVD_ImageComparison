"""
Experiment runner for parameter sweep testing.
Tests multiple (rank, threshold) combinations and collects comprehensive metrics.
"""

import time
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict

from ai.detection_ai_test1 import AIDetectionModule
from preprocessing.image_preprocessing_test1 import FramePreprocessor
from svd.svd_test2 import SVDNoveltyDetector
from testing.metrics_calculator import PerFrameMetrics, MetricsCalculator
from testing.system_monitor import SystemMonitor

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Run SVD novelty detection experiments with parameter sweeps."""
    
    def __init__(
        self,
        target_size: tuple = (240, 320),
        use_mock_ai: bool = False
    ):
        """
        Initialize experiment runner.
        
        Args:
            target_size: (height, width) for frame resizing
            use_mock_ai: Use mock AI detection instead of real YOLO
        """
        self.target_size = target_size
        self.use_mock_ai = use_mock_ai
        
        self.preprocessor = FramePreprocessor(target_size=target_size)
        self.ai_module = AIDetectionModule(use_mock=use_mock_ai)
        self.monitor = SystemMonitor()
        
        self.experiment_data: List[PerFrameMetrics] = []
        self.ground_truth: Optional[Dict[int, bool]] = None
        
        logger.info("=" * 70)
        logger.info("Experiment Runner Initialized")
        logger.info(f"  Target size: {target_size}")
        logger.info(f"  Using mock AI: {use_mock_ai}")
        logger.info("=" * 70)
    
    def load_dataset(self, dataset_path: str) -> None:
        """
        Load dataset for experiments.
        
        Args:
            dataset_path: Path to folder or .mp4 file
        """
        if dataset_path.endswith('.mp4'):
            logger.info(f"Loading video: {dataset_path}")
            self.preprocessor.load_video(dataset_path, difference_window=1)
        else:
            logger.info(f"Loading images from: {dataset_path}")
            self.preprocessor.load_images(dataset_path)
        
        logger.info(f"Loaded {len(self.preprocessor.frames)} frames for experiments")
    
    def set_ground_truth(self, ground_truth: Dict[int, bool]) -> None:
        """
        Set ground truth labels for quality metrics.
        
        Args:
            ground_truth: Dictionary mapping frame_index -> is_novel
        """
        self.ground_truth = ground_truth
        logger.info(f"Ground truth set: {len(ground_truth)} labels")
    
    def run_single_configuration(
        self,
        compression_rank: int,
        threshold: float
    ) -> List[PerFrameMetrics]:
        """
        Run experiment for a single (rank, threshold) configuration.
        
        Args:
            compression_rank: SVD compression rank
            threshold: Novelty detection threshold
            
        Returns:
            List of PerFrameMetrics for all frames
        """
        logger.info("\n" + "-" * 70)
        logger.info(f"Running Configuration: rank={compression_rank}, threshold={threshold:.4f}")
        logger.info("-" * 70)
        
        # Initialize SVD detector for this configuration
        svd_detector = SVDNoveltyDetector(compression_rank=compression_rank)
        
        config_data = []
        total_frames = len(self.preprocessor.frames)
        
        # Reset preprocessor
        self.preprocessor.frame_index = 0
        
        # Initialize reference error
        reference_error = None
        
        for frame_idx, frame in self.preprocessor.get_next_frame():
            timestamp_start = time.time()
            
            # Gate: SVD reconstruction and novelty decision
            gate_start = time.time()
            error, frame_reconstructed = svd_detector.compare_frames(frame)
            gate_end = time.time()
            gate_ms = (gate_end - gate_start) * 1000
            
            # Initialize reference error on first frame
            if reference_error is None:
                reference_error = error
                # Always process first frame through AI
                is_novel = True
            else:
                # Novelty decision
                is_novel = abs(error - reference_error) > threshold
                if is_novel:
                    reference_error = error  # Update reference
            
            # AI processing if novel
            ai_called = False
            ai_infer_ms = 0.0
            
            if is_novel:
                ai_start = time.time()
                detections = self.ai_module.run_detection_ai(frame)
                ai_end = time.time()
                ai_infer_ms = (ai_end - ai_start) * 1000
                ai_called = True
            
            # Total end-to-end time
            timestamp_end = time.time()
            e2e_ms = (timestamp_end - timestamp_start) * 1000
            
            # Get resource usage
            cpu_percent = self.monitor.get_cpu_percent(interval=0.01)
            memory_mb = self.monitor.get_memory_mb()
            
            # Get ground truth if available
            is_novel_truth = None
            if self.ground_truth is not None and frame_idx in self.ground_truth:
                is_novel_truth = self.ground_truth[frame_idx]
            
            # Create metrics record
            metrics = PerFrameMetrics(
                frame_index=frame_idx,
                compression_rank=compression_rank,
                threshold=threshold,
                reconstruction_error=error,
                is_novel_predicted=is_novel,
                is_novel_truth=is_novel_truth,
                gate_ms=gate_ms,
                ai_called=ai_called,
                ai_infer_ms=ai_infer_ms,
                ai_e2e_ms=e2e_ms,
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                timestamp=timestamp_start
            )
            
            config_data.append(metrics)
            
            # Log progress
            if frame_idx % 10 == 0 or frame_idx == total_frames - 1:
                status = "NOVEL" if is_novel else "SKIP"
                logger.info(
                    f"[{status}] Frame {frame_idx} ({frame_idx+1}/{total_frames}) | "
                    f"Error: {error:.6f} | Gate: {gate_ms:.2f}ms | "
                    f"AI: {ai_infer_ms:.2f}ms | E2E: {e2e_ms:.2f}ms"
                )
        
        # Calculate summary for this config
        novel_count = sum(1 for m in config_data if m.is_novel_predicted)
        reduction_pct = 100.0 * (1.0 - novel_count / total_frames) if total_frames > 0 else 0.0
        
        logger.info("-" * 70)
        logger.info(f"Configuration Complete: rank={compression_rank}, threshold={threshold:.4f}")
        logger.info(f"  Novel frames: {novel_count}/{total_frames}")
        logger.info(f"  Reduction: {reduction_pct:.2f}%")
        logger.info("-" * 70)
        
        return config_data
    
    def run_parameter_sweep(
        self,
        compression_ranks: List[int],
        thresholds: List[float]
    ) -> List[PerFrameMetrics]:
        """
        Run parameter sweep across multiple (rank, threshold) combinations.
        
        Args:
            compression_ranks: List of compression ranks to test
            thresholds: List of thresholds to test
            
        Returns:
            List of all PerFrameMetrics from all configurations
        """
        logger.info("\n" + "=" * 70)
        logger.info("PARAMETER SWEEP")
        logger.info("=" * 70)
        logger.info(f"Compression ranks: {compression_ranks}")
        logger.info(f"Thresholds: {thresholds}")
        logger.info(f"Total configurations: {len(compression_ranks) * len(thresholds)}")
        logger.info("=" * 70)
        
        all_data = []
        total_configs = len(compression_ranks) * len(thresholds)
        config_num = 0
        
        for rank in compression_ranks:
            for threshold in thresholds:
                config_num += 1
                logger.info(f"\n[Configuration {config_num}/{total_configs}]")
                
                # Run single configuration
                config_data = self.run_single_configuration(rank, threshold)
                all_data.extend(config_data)
        
        logger.info("\n" + "=" * 70)
        logger.info("PARAMETER SWEEP COMPLETE")
        logger.info(f"  Total configurations tested: {total_configs}")
        logger.info(f"  Total measurements: {len(all_data)}")
        logger.info("=" * 70)
        
        self.experiment_data = all_data
        return all_data
    
    def run_grid_search(
        self,
        rank_min: int,
        rank_max: int,
        rank_step: int,
        threshold_min: float,
        threshold_max: float,
        threshold_step: float
    ) -> List[PerFrameMetrics]:
        """
        Run grid search over rank and threshold ranges.
        
        Args:
            rank_min: Minimum compression rank
            rank_max: Maximum compression rank
            rank_step: Step size for ranks
            threshold_min: Minimum threshold
            threshold_max: Maximum threshold
            threshold_step: Step size for thresholds
            
        Returns:
            List of all PerFrameMetrics from all configurations
        """
        # Generate parameter lists
        compression_ranks = list(range(rank_min, rank_max + 1, rank_step))
        
        thresholds = []
        threshold = threshold_min
        while threshold <= threshold_max:
            thresholds.append(round(threshold, 4))
            threshold += threshold_step
        
        return self.run_parameter_sweep(compression_ranks, thresholds)
    
    def get_best_configuration(
        self,
        metric: str = 'f1_score',
        min_reduction: float = 0.0
    ) -> Optional[Tuple[int, float, dict]]:
        """
        Find best configuration based on a specific metric.
        
        Args:
            metric: Metric to optimize ('f1_score', 'accuracy', 'reduction_percentage', etc.)
            min_reduction: Minimum required reduction percentage
            
        Returns:
            Tuple of (best_rank, best_threshold, metrics_dict) or None
        """
        if not self.experiment_data:
            logger.warning("No experiment data available")
            return None
        
        # Calculate metrics for all configurations
        calculator = MetricsCalculator()
        calculator.per_frame_data = self.experiment_data
        if self.ground_truth:
            calculator.set_ground_truth(self.ground_truth)
        
        # Get unique configurations
        configs = set((m.compression_rank, m.threshold) for m in self.experiment_data)
        
        best_config = None
        best_value = -float('inf')
        
        for rank, threshold in configs:
            summary = calculator.calculate_summary_metrics(rank, threshold)
            
            # Check minimum reduction requirement
            if summary.reduction_percentage < min_reduction:
                continue
            
            # Get metric value
            value = getattr(summary, metric, None)
            if value is None:
                continue
            
            if value > best_value:
                best_value = value
                best_config = (rank, threshold, {
                    'rank': rank,
                    'threshold': threshold,
                    metric: value,
                    'reduction_percentage': summary.reduction_percentage,
                    'f1_score': summary.f1_score,
                    'accuracy': summary.accuracy,
                    'avg_fps': summary.avg_fps
                })
        
        return best_config
    
    def reset(self) -> None:
        """Reset experiment data."""
        self.experiment_data.clear()
        logger.info("Experiment runner reset")
