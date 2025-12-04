"""
Testing module for SVD novelty detection evaluation.

This module provides comprehensive testing and metrics collection for evaluating
SVD-based novelty detection systems.

Components:
    - baseline_runner: Run baseline measurements without filtering
    - experiment_runner: Run parameter sweep experiments
    - ground_truth_loader: Load and validate ground truth labels
    - metrics_calculator: Calculate all performance and quality metrics
    - csv_logger: Write results to CSV files
    - system_monitor: Track CPU and memory usage
"""

from testing.baseline_runner import BaselineRunner
from testing.experiment_runner import ExperimentRunner
from testing.ground_truth_loader import GroundTruthLoader
from testing.metrics_calculator import (
    MetricsCalculator,
    PerFrameMetrics,
    SummaryMetrics,
    BaselineMetrics,
    ComparisonMetrics
)
from testing.csv_logger import CSVLogger
from testing.system_monitor import SystemMonitor, ResourceTracker

__all__ = [
    'BaselineRunner',
    'ExperimentRunner',
    'GroundTruthLoader',
    'MetricsCalculator',
    'PerFrameMetrics',
    'SummaryMetrics',
    'BaselineMetrics',
    'ComparisonMetrics',
    'CSVLogger',
    'SystemMonitor',
    'ResourceTracker'
]

__version__ = '1.0.0'
