"""
CSV logging module for experiment results.
Writes 5 types of CSV files: per-frame, summary, baseline, comparison, and parameter sweep.
"""

import csv
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from testing.metrics_calculator import (
    PerFrameMetrics, 
    SummaryMetrics, 
    BaselineMetrics, 
    ComparisonMetrics
)

logger = logging.getLogger(__name__)


class CSVLogger:
    """Handles writing all experiment results to CSV files."""
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize CSV logger.
        
        Args:
            output_dir: Directory to save CSV files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for this experiment run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"CSV logger initialized: {self.output_dir}")
    
    def write_per_frame_log(
        self, 
        frame_data: List[PerFrameMetrics], 
        filename: Optional[str] = None
    ) -> str:
        """
        Write CSV 1: per-frame detailed log.
        
        Columns: frame_index, compression_rank, threshold, reconstruction_error,
                 is_novel_predicted, is_novel_truth, gate_ms, ai_called,
                 ai_infer_ms, ai_e2e_ms, cpu_percent, memory_mb, timestamp
        
        Args:
            frame_data: List of PerFrameMetrics
            filename: Optional custom filename (default: per_frame_log_TIMESTAMP.csv)
            
        Returns:
            Path to written CSV file
        """
        if filename is None:
            filename = f"per_frame_log_{self.timestamp}.csv"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'frame_index',
                'compression_rank',
                'threshold',
                'reconstruction_error',
                'is_novel_predicted',
                'is_novel_truth',
                'gate_ms',
                'ai_called',
                'ai_infer_ms',
                'ai_e2e_ms',
                'cpu_percent',
                'memory_mb',
                'timestamp'
            ])
            
            # Data rows
            for m in frame_data:
                writer.writerow([
                    m.frame_index,
                    m.compression_rank,
                    f"{m.threshold:.4f}",
                    f"{m.reconstruction_error:.6f}",
                    int(m.is_novel_predicted),
                    int(m.is_novel_truth) if m.is_novel_truth is not None else '',
                    f"{m.gate_ms:.4f}",
                    int(m.ai_called),
                    f"{m.ai_infer_ms:.4f}",
                    f"{m.ai_e2e_ms:.4f}",
                    f"{m.cpu_percent:.2f}",
                    f"{m.memory_mb:.2f}",
                    f"{m.timestamp:.6f}"
                ])
        
        logger.info(f"Wrote per-frame log: {output_path} ({len(frame_data)} frames)")
        return str(output_path)
    
    def write_summary_metrics(
        self, 
        summaries: List[SummaryMetrics], 
        filename: Optional[str] = None
    ) -> str:
        """
        Write CSV 2: summary metrics per (rank, threshold).
        
        Args:
            summaries: List of SummaryMetrics
            filename: Optional custom filename (default: summary_metrics_TIMESTAMP.csv)
            
        Returns:
            Path to written CSV file
        """
        if filename is None:
            filename = f"summary_metrics_{self.timestamp}.csv"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'compression_rank',
                'threshold',
                'total_frames',
                'novel_frames',
                'ignored_frames',
                'forwarding_ratio',
                'reduction_percentage',
                'avg_gate_ms',
                'avg_ai_infer_ms',
                'avg_ai_e2e_ms',
                'avg_fps',
                'avg_ms_per_frame',
                'avg_cpu_percent',
                'avg_memory_mb',
                'precision',
                'recall',
                'f1_score',
                'filter_accuracy'
            ])
            
            # Data rows
            for s in summaries:
                writer.writerow([
                    s.compression_rank,
                    f"{s.threshold:.4f}",
                    s.total_frames,
                    s.novel_frames,
                    s.ignored_frames,
                    f"{s.forwarding_ratio:.4f}",
                    f"{s.reduction_percentage:.2f}",
                    f"{s.avg_gate_ms:.4f}",
                    f"{s.avg_ai_infer_ms:.4f}",
                    f"{s.avg_ai_e2e_ms:.4f}",
                    f"{s.avg_fps:.2f}",
                    f"{s.avg_ms_per_frame:.4f}",
                    f"{s.avg_cpu_percent:.2f}",
                    f"{s.avg_memory_mb:.2f}",
                    f"{s.precision:.4f}" if s.precision is not None else '',
                    f"{s.recall:.4f}" if s.recall is not None else '',
                    f"{s.f1_score:.4f}" if s.f1_score is not None else '',
                    f"{s.accuracy:.4f}" if s.accuracy is not None else ''
                ])
        
        logger.info(f"Wrote summary metrics: {output_path} ({len(summaries)} configurations)")
        return str(output_path)
    
    def write_baseline_metrics(
        self, 
        baseline: BaselineMetrics, 
        filename: Optional[str] = None
    ) -> str:
        """
        Write CSV 3: baseline metrics.
        
        Args:
            baseline: BaselineMetrics instance
            filename: Optional custom filename (default: baseline_metrics_TIMESTAMP.csv)
            
        Returns:
            Path to written CSV file
        """
        if filename is None:
            filename = f"baseline_metrics_{self.timestamp}.csv"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'total_frames',
                'baseline_fps',
                'baseline_ms_per_frame',
                'baseline_cpu_percent',
                'baseline_memory_mb',
                'baseline_total_ai_calls',
                'baseline_total_time_ms'
            ])
            
            # Data row
            writer.writerow([
                baseline.total_frames,
                f"{baseline.baseline_fps:.2f}",
                f"{baseline.baseline_ms_per_frame:.4f}",
                f"{baseline.baseline_cpu_percent:.2f}",
                f"{baseline.baseline_memory_mb:.2f}",
                baseline.baseline_total_ai_calls,
                f"{baseline.baseline_total_time_ms:.2f}"
            ])
        
        logger.info(f"Wrote baseline metrics: {output_path}")
        return str(output_path)
    
    def write_comparison_metrics(
        self, 
        comparisons: List[ComparisonMetrics], 
        filename: Optional[str] = None
    ) -> str:
        """
        Write CSV 4: comparison metrics.
        
        Args:
            comparisons: List of ComparisonMetrics
            filename: Optional custom filename (default: comparison_metrics_TIMESTAMP.csv)
            
        Returns:
            Path to written CSV file
        """
        if filename is None:
            filename = f"comparison_metrics_{self.timestamp}.csv"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'compression_rank',
                'threshold',
                'metric',
                'baseline_value',
                'filter_value',
                'improvement_percent'
            ])
            
            # Data rows
            for c in comparisons:
                writer.writerow([
                    c.compression_rank,
                    f"{c.threshold:.4f}",
                    c.metric,
                    f"{c.baseline_value:.4f}",
                    f"{c.filter_value:.4f}",
                    f"{c.improvement_percent:.2f}"
                ])
        
        logger.info(f"Wrote comparison metrics: {output_path} ({len(comparisons)} comparisons)")
        return str(output_path)
    
    def write_parameter_sweep(
        self, 
        summaries: List[SummaryMetrics],
        baseline: BaselineMetrics,
        filename: Optional[str] = None
    ) -> str:
        """
        Write CSV 5: parameter sweep table.
        
        Each row is one experiment run with key metrics.
        
        Args:
            summaries: List of SummaryMetrics
            baseline: BaselineMetrics for computing savings
            filename: Optional custom filename (default: parameter_sweep_TIMESTAMP.csv)
            
        Returns:
            Path to written CSV file
        """
        if filename is None:
            filename = f"parameter_sweep_{self.timestamp}.csv"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'compression_rank',
                'threshold',
                'forwarding_ratio',
                'reduction_percentage',
                'avg_fps',
                'avg_ms_per_frame',
                'avg_ai_infer_ms',
                'precision',
                'recall',
                'f1_score',
                'time_saved_percent',
                'ai_calls_reduction_percent',
                'cpu_saved_percent',
                'memory_saved_percent'
            ])
            
            # Data rows
            for s in summaries:
                # Calculate savings
                filter_total_time = s.avg_ms_per_frame * s.total_frames
                baseline_total_time = baseline.baseline_total_time_ms
                time_saved_pct = 100.0 * (1.0 - filter_total_time / baseline_total_time) if baseline_total_time > 0 else 0.0
                
                ai_calls_reduction_pct = 100.0 * (1.0 - s.forwarding_ratio)
                
                cpu_saved_pct = 100.0 * (baseline.baseline_cpu_percent - s.avg_cpu_percent) / baseline.baseline_cpu_percent if baseline.baseline_cpu_percent > 0 else 0.0
                memory_saved_pct = 100.0 * (baseline.baseline_memory_mb - s.avg_memory_mb) / baseline.baseline_memory_mb if baseline.baseline_memory_mb > 0 else 0.0
                
                writer.writerow([
                    s.compression_rank,
                    f"{s.threshold:.4f}",
                    f"{s.forwarding_ratio:.4f}",
                    f"{s.reduction_percentage:.2f}",
                    f"{s.avg_fps:.2f}",
                    f"{s.avg_ms_per_frame:.4f}",
                    f"{s.avg_ai_infer_ms:.4f}",
                    f"{s.precision:.4f}" if s.precision is not None else '',
                    f"{s.recall:.4f}" if s.recall is not None else '',
                    f"{s.f1_score:.4f}" if s.f1_score is not None else '',
                    f"{time_saved_pct:.2f}",
                    f"{ai_calls_reduction_pct:.2f}",
                    f"{cpu_saved_pct:.2f}",
                    f"{memory_saved_pct:.2f}"
                ])
        
        logger.info(f"Wrote parameter sweep: {output_path} ({len(summaries)} configurations)")
        return str(output_path)
    
    def write_all(
        self,
        per_frame_data: List[PerFrameMetrics],
        summaries: List[SummaryMetrics],
        baseline: BaselineMetrics,
        comparisons: List[ComparisonMetrics]
    ) -> dict:
        """
        Write all CSV files at once.
        
        Args:
            per_frame_data: List of PerFrameMetrics
            summaries: List of SummaryMetrics
            baseline: BaselineMetrics
            comparisons: List of ComparisonMetrics
            
        Returns:
            Dictionary mapping CSV type to file path
        """
        output_files = {}
        
        output_files['per_frame'] = self.write_per_frame_log(per_frame_data)
        output_files['summary'] = self.write_summary_metrics(summaries)
        output_files['baseline'] = self.write_baseline_metrics(baseline)
        output_files['comparison'] = self.write_comparison_metrics(comparisons)
        output_files['parameter_sweep'] = self.write_parameter_sweep(summaries, baseline)
        
        logger.info(f"\nAll CSV files written to: {self.output_dir}")
        logger.info(f"  Per-frame log: {Path(output_files['per_frame']).name}")
        logger.info(f"  Summary metrics: {Path(output_files['summary']).name}")
        logger.info(f"  Baseline metrics: {Path(output_files['baseline']).name}")
        logger.info(f"  Comparison metrics: {Path(output_files['comparison']).name}")
        logger.info(f"  Parameter sweep: {Path(output_files['parameter_sweep']).name}")
        
        return output_files
