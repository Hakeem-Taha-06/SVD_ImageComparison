"""
Comprehensive metrics calculation module for SVD novelty detection evaluation.
Computes performance, behavioral, and quality metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerFrameMetrics:
    """Metrics for a single frame processing."""
    frame_index: int
    compression_rank: int
    threshold: float
    reconstruction_error: float
    is_novel_predicted: bool
    is_novel_truth: Optional[bool] = None
    gate_ms: float = 0.0  # Time for SVD gating decision
    ai_called: bool = False
    ai_infer_ms: float = 0.0  # AI inference time
    ai_e2e_ms: float = 0.0  # End-to-end time (gate + AI if called)
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    timestamp: float = 0.0


@dataclass
class SummaryMetrics:
    """Aggregated metrics for a (rank, threshold) configuration."""
    compression_rank: int
    threshold: float
    
    # Filter behavior
    total_frames: int = 0
    novel_frames: int = 0
    ignored_frames: int = 0
    forwarding_ratio: float = 0.0  # novel_frames / total_frames
    reduction_percentage: float = 0.0  # 100 * (1 - forwarding_ratio)
    
    # Performance metrics
    avg_gate_ms: float = 0.0
    avg_ai_infer_ms: float = 0.0
    avg_ai_e2e_ms: float = 0.0
    avg_fps: float = 0.0
    avg_ms_per_frame: float = 0.0
    
    # Resource usage
    avg_cpu_percent: float = 0.0
    avg_memory_mb: float = 0.0
    
    # Quality metrics (if ground truth available)
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    accuracy: Optional[float] = None
    
    # Confusion matrix elements
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


@dataclass
class BaselineMetrics:
    """Metrics for baseline (no filter) run."""
    total_frames: int = 0
    baseline_fps: float = 0.0
    baseline_ms_per_frame: float = 0.0
    baseline_cpu_percent: float = 0.0
    baseline_memory_mb: float = 0.0
    baseline_total_ai_calls: int = 0
    baseline_total_time_ms: float = 0.0


@dataclass
class ComparisonMetrics:
    """Comparison between filter and baseline."""
    compression_rank: int
    threshold: float
    metric: str
    baseline_value: float
    filter_value: float
    improvement_percent: float


class MetricsCalculator:
    """Calculate all required metrics for SVD novelty detection evaluation."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.per_frame_data: List[PerFrameMetrics] = []
        self.ground_truth: Optional[Dict[int, bool]] = None
    
    def set_ground_truth(self, ground_truth: Dict[int, bool]) -> None:
        """
        Set ground truth labels for novelty detection quality metrics.
        
        Args:
            ground_truth: Dictionary mapping frame_index -> is_novel_truth
        """
        self.ground_truth = ground_truth
        logger.info(f"Ground truth loaded: {len(ground_truth)} labels")
    
    def add_frame_metrics(self, metrics: PerFrameMetrics) -> None:
        """
        Add metrics for a single frame.
        
        Args:
            metrics: PerFrameMetrics instance
        """
        self.per_frame_data.append(metrics)
    
    def calculate_classification_metrics(
        self, 
        predictions: List[bool], 
        ground_truth: List[bool]
    ) -> Tuple[float, float, float, float, int, int, int, int]:
        """
        Calculate precision, recall, F1-score, and accuracy.
        
        Args:
            predictions: List of predicted novelty labels
            ground_truth: List of true novelty labels
            
        Returns:
            Tuple of (precision, recall, f1, accuracy, tp, tn, fp, fn)
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        if len(predictions) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0
        
        # Calculate confusion matrix
        tp = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
        tn = sum(1 for p, g in zip(predictions, ground_truth) if not p and not g)
        fp = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)
        fn = sum(1 for p, g in zip(predictions, ground_truth) if not p and g)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0.0
        
        return precision, recall, f1, accuracy, tp, tn, fp, fn
    
    def calculate_summary_metrics(
        self, 
        compression_rank: int, 
        threshold: float
    ) -> SummaryMetrics:
        """
        Calculate summary metrics for a specific (rank, threshold) configuration.
        
        Args:
            compression_rank: SVD compression rank
            threshold: Novelty detection threshold
            
        Returns:
            SummaryMetrics instance
        """
        # Filter data for this configuration
        config_data = [
            m for m in self.per_frame_data 
            if m.compression_rank == compression_rank and m.threshold == threshold
        ]
        
        if not config_data:
            logger.warning(f"No data for rank={compression_rank}, threshold={threshold}")
            return SummaryMetrics(compression_rank=compression_rank, threshold=threshold)
        
        summary = SummaryMetrics(
            compression_rank=compression_rank,
            threshold=threshold
        )
        
        # Basic counts
        summary.total_frames = len(config_data)
        summary.novel_frames = sum(1 for m in config_data if m.is_novel_predicted)
        summary.ignored_frames = summary.total_frames - summary.novel_frames
        
        # Filter behavior
        summary.forwarding_ratio = summary.novel_frames / summary.total_frames if summary.total_frames > 0 else 0.0
        summary.reduction_percentage = 100.0 * (1.0 - summary.forwarding_ratio)
        
        # Performance metrics
        gate_times = [m.gate_ms for m in config_data]
        ai_times = [m.ai_infer_ms for m in config_data if m.ai_called]
        e2e_times = [m.ai_e2e_ms for m in config_data]
        
        summary.avg_gate_ms = np.mean(gate_times) if gate_times else 0.0
        summary.avg_ai_infer_ms = np.mean(ai_times) if ai_times else 0.0
        summary.avg_ai_e2e_ms = np.mean(e2e_times) if e2e_times else 0.0
        summary.avg_ms_per_frame = summary.avg_ai_e2e_ms
        summary.avg_fps = 1000.0 / summary.avg_ms_per_frame if summary.avg_ms_per_frame > 0 else 0.0
        
        # Resource usage
        summary.avg_cpu_percent = np.mean([m.cpu_percent for m in config_data])
        summary.avg_memory_mb = np.mean([m.memory_mb for m in config_data])
        
        # Quality metrics (if ground truth available)
        if self.ground_truth is not None:
            predictions = []
            truths = []
            
            for m in config_data:
                if m.frame_index in self.ground_truth:
                    predictions.append(m.is_novel_predicted)
                    truths.append(self.ground_truth[m.frame_index])
            
            if predictions and truths:
                precision, recall, f1, accuracy, tp, tn, fp, fn = \
                    self.calculate_classification_metrics(predictions, truths)
                
                summary.precision = precision
                summary.recall = recall
                summary.f1_score = f1
                summary.accuracy = accuracy
                summary.true_positives = tp
                summary.true_negatives = tn
                summary.false_positives = fp
                summary.false_negatives = fn
        
        return summary
    
    def calculate_baseline_metrics(
        self, 
        baseline_frame_data: List[PerFrameMetrics]
    ) -> BaselineMetrics:
        """
        Calculate baseline metrics (no filter, all frames processed).
        
        Args:
            baseline_frame_data: List of PerFrameMetrics from baseline run
            
        Returns:
            BaselineMetrics instance
        """
        if not baseline_frame_data:
            return BaselineMetrics()
        
        baseline = BaselineMetrics()
        baseline.total_frames = len(baseline_frame_data)
        baseline.baseline_total_ai_calls = baseline.total_frames
        
        # Performance
        e2e_times = [m.ai_e2e_ms for m in baseline_frame_data]
        baseline.baseline_total_time_ms = sum(e2e_times)
        baseline.baseline_ms_per_frame = np.mean(e2e_times) if e2e_times else 0.0
        baseline.baseline_fps = 1000.0 / baseline.baseline_ms_per_frame if baseline.baseline_ms_per_frame > 0 else 0.0
        
        # Resources
        baseline.baseline_cpu_percent = np.mean([m.cpu_percent for m in baseline_frame_data])
        baseline.baseline_memory_mb = np.mean([m.memory_mb for m in baseline_frame_data])
        
        return baseline
    
    def calculate_comparison_metrics(
        self, 
        summary: SummaryMetrics, 
        baseline: BaselineMetrics
    ) -> List[ComparisonMetrics]:
        """
        Calculate comparison metrics between filter and baseline.
        
        Args:
            summary: SummaryMetrics for a configuration
            baseline: BaselineMetrics
            
        Returns:
            List of ComparisonMetrics
        """
        comparisons = []
        
        # Time saved
        filter_total_time = summary.avg_ms_per_frame * summary.total_frames
        baseline_total_time = baseline.baseline_total_time_ms
        time_saved_pct = 100.0 * (1.0 - filter_total_time / baseline_total_time) if baseline_total_time > 0 else 0.0
        
        comparisons.append(ComparisonMetrics(
            compression_rank=summary.compression_rank,
            threshold=summary.threshold,
            metric="time_saved",
            baseline_value=baseline_total_time,
            filter_value=filter_total_time,
            improvement_percent=time_saved_pct
        ))
        
        # AI calls reduction
        ai_calls_reduction_pct = 100.0 * (1.0 - summary.forwarding_ratio)
        comparisons.append(ComparisonMetrics(
            compression_rank=summary.compression_rank,
            threshold=summary.threshold,
            metric="ai_calls_reduction",
            baseline_value=baseline.baseline_total_ai_calls,
            filter_value=summary.novel_frames,
            improvement_percent=ai_calls_reduction_pct
        ))
        
        # CPU saved
        cpu_saved = baseline.baseline_cpu_percent - summary.avg_cpu_percent
        cpu_saved_pct = 100.0 * cpu_saved / baseline.baseline_cpu_percent if baseline.baseline_cpu_percent > 0 else 0.0
        comparisons.append(ComparisonMetrics(
            compression_rank=summary.compression_rank,
            threshold=summary.threshold,
            metric="cpu_saved",
            baseline_value=baseline.baseline_cpu_percent,
            filter_value=summary.avg_cpu_percent,
            improvement_percent=cpu_saved_pct
        ))
        
        # Memory saved
        memory_saved = baseline.baseline_memory_mb - summary.avg_memory_mb
        memory_saved_pct = 100.0 * memory_saved / baseline.baseline_memory_mb if baseline.baseline_memory_mb > 0 else 0.0
        comparisons.append(ComparisonMetrics(
            compression_rank=summary.compression_rank,
            threshold=summary.threshold,
            metric="memory_saved",
            baseline_value=baseline.baseline_memory_mb,
            filter_value=summary.avg_memory_mb,
            improvement_percent=memory_saved_pct
        ))
        
        # FPS improvement
        fps_improvement_pct = 100.0 * (summary.avg_fps - baseline.baseline_fps) / baseline.baseline_fps if baseline.baseline_fps > 0 else 0.0
        comparisons.append(ComparisonMetrics(
            compression_rank=summary.compression_rank,
            threshold=summary.threshold,
            metric="fps_improvement",
            baseline_value=baseline.baseline_fps,
            filter_value=summary.avg_fps,
            improvement_percent=fps_improvement_pct
        ))
        
        return comparisons
    
    def reset(self) -> None:
        """Reset all collected data."""
        self.per_frame_data.clear()
        logger.info("Metrics calculator reset")
