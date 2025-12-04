"""
Baseline measurement module for running AI detection on all frames without SVD filtering.
Establishes performance baseline for comparison.
"""

import time
import logging
import numpy as np
from typing import List, Optional

from ai.detection_ai_test1 import AIDetectionModule
from preprocessing.image_preprocessing_test1 import FramePreprocessor
from testing.metrics_calculator import PerFrameMetrics
from testing.system_monitor import SystemMonitor

logger = logging.getLogger(__name__)


class BaselineRunner:
    """Run baseline experiment: process all frames through AI without filtering."""
    
    def __init__(
        self, 
        target_size: tuple = (240, 320),
        use_mock_ai: bool = False
    ):
        """
        Initialize baseline runner.
        
        Args:
            target_size: (height, width) for frame resizing
            use_mock_ai: Use mock AI detection instead of real YOLO
        """
        self.target_size = target_size
        self.use_mock_ai = use_mock_ai
        
        self.preprocessor = FramePreprocessor(target_size=target_size)
        self.ai_module = AIDetectionModule(use_mock=use_mock_ai)
        self.monitor = SystemMonitor()
        
        self.baseline_data: List[PerFrameMetrics] = []
        
        logger.info("=" * 70)
        logger.info("Baseline Runner Initialized (No SVD Filter)")
        logger.info(f"  Target size: {target_size}")
        logger.info(f"  Using mock AI: {use_mock_ai}")
        logger.info("=" * 70)
    
    def load_dataset(self, dataset_path: str) -> None:
        """
        Load dataset for baseline testing.
        
        Args:
            dataset_path: Path to folder or .mp4 file
        """
        if dataset_path.endswith('.mp4'):
            logger.info(f"Loading video: {dataset_path}")
            self.preprocessor.load_video(dataset_path, difference_window=1)
        else:
            logger.info(f"Loading images from: {dataset_path}")
            self.preprocessor.load_images(dataset_path)
        
        logger.info(f"Loaded {len(self.preprocessor.frames)} frames for baseline")
    
    def run_baseline(self) -> List[PerFrameMetrics]:
        """
        Run baseline: process all frames through AI.
        
        Returns:
            List of PerFrameMetrics for all frames
        """
        logger.info("\n" + "=" * 70)
        logger.info("Starting Baseline Run (Processing All Frames)")
        logger.info("=" * 70)
        
        self.baseline_data.clear()
        total_frames = len(self.preprocessor.frames)
        
        # Reset preprocessor frame index
        self.preprocessor.frame_index = 0
        
        for frame_idx, frame in self.preprocessor.get_next_frame():
            # Record timestamp
            timestamp_start = time.time()
            
            # No gating - directly run AI
            gate_ms = 0.0  # No SVD gating in baseline
            
            # Run AI detection
            ai_start = time.time()
            detections = self.ai_module.run_detection_ai(frame)
            ai_end = time.time()
            
            ai_infer_ms = (ai_end - ai_start) * 1000
            
            # Total end-to-end time (same as AI time for baseline)
            timestamp_end = time.time()
            e2e_ms = (timestamp_end - timestamp_start) * 1000
            
            # Get resource usage
            cpu_percent = self.monitor.get_cpu_percent(interval=0.01)
            memory_mb = self.monitor.get_memory_mb()
            
            # Create metrics record
            metrics = PerFrameMetrics(
                frame_index=frame_idx,
                compression_rank=0,  # N/A for baseline
                threshold=0.0,  # N/A for baseline
                reconstruction_error=0.0,  # N/A for baseline
                is_novel_predicted=True,  # All frames "pass through" in baseline
                is_novel_truth=None,  # Will be set later if ground truth available
                gate_ms=gate_ms,
                ai_called=True,  # Always true for baseline
                ai_infer_ms=ai_infer_ms,
                ai_e2e_ms=e2e_ms,
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                timestamp=timestamp_start
            )
            
            self.baseline_data.append(metrics)
            
            # Log progress
            if frame_idx % 10 == 0 or frame_idx == total_frames - 1:
                logger.info(
                    f"[Baseline] Frame {frame_idx} ({frame_idx+1}/{total_frames}) | "
                    f"AI: {ai_infer_ms:.2f}ms | "
                    f"E2E: {e2e_ms:.2f}ms | "
                    f"CPU: {cpu_percent:.1f}% | "
                    f"Mem: {memory_mb:.1f}MB"
                )
        
        logger.info("=" * 70)
        logger.info("Baseline Run Complete")
        logger.info(f"  Total frames processed: {len(self.baseline_data)}")
        logger.info(f"  Total AI calls: {len(self.baseline_data)}")
        logger.info("=" * 70)
        
        return self.baseline_data
    
    def get_baseline_summary(self) -> dict:
        """
        Get baseline performance summary.
        
        Returns:
            Dictionary with baseline statistics
        """
        if not self.baseline_data:
            return {}
        
        total_frames = len(self.baseline_data)
        
        # Calculate averages
        avg_ai_ms = sum(m.ai_infer_ms for m in self.baseline_data) / total_frames
        avg_e2e_ms = sum(m.ai_e2e_ms for m in self.baseline_data) / total_frames
        avg_cpu = sum(m.cpu_percent for m in self.baseline_data) / total_frames
        avg_memory = sum(m.memory_mb for m in self.baseline_data) / total_frames
        
        total_time_ms = sum(m.ai_e2e_ms for m in self.baseline_data)
        fps = 1000.0 / avg_e2e_ms if avg_e2e_ms > 0 else 0.0
        
        return {
            'total_frames': total_frames,
            'total_ai_calls': total_frames,
            'avg_ai_infer_ms': avg_ai_ms,
            'avg_e2e_ms': avg_e2e_ms,
            'avg_cpu_percent': avg_cpu,
            'avg_memory_mb': avg_memory,
            'total_time_ms': total_time_ms,
            'fps': fps
        }
    
    def print_baseline_summary(self) -> None:
        """Print baseline performance summary."""
        summary = self.get_baseline_summary()
        
        if not summary:
            logger.warning("No baseline data to summarize")
            return
        
        logger.info("\n" + "=" * 70)
        logger.info("BASELINE PERFORMANCE SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total Frames:              {summary['total_frames']}")
        logger.info(f"Total AI Calls:            {summary['total_ai_calls']}")
        logger.info(f"Avg AI Inference Time:     {summary['avg_ai_infer_ms']:.2f} ms")
        logger.info(f"Avg E2E Time per Frame:    {summary['avg_e2e_ms']:.2f} ms")
        logger.info(f"Throughput (FPS):          {summary['fps']:.2f}")
        logger.info(f"Total Processing Time:     {summary['total_time_ms']/1000:.2f} seconds")
        logger.info(f"Avg CPU Usage:             {summary['avg_cpu_percent']:.1f}%")
        logger.info(f"Avg Memory Usage:          {summary['avg_memory_mb']:.1f} MB")
        logger.info("=" * 70 + "\n")
    
    def reset(self) -> None:
        """Reset baseline data."""
        self.baseline_data.clear()
        logger.info("Baseline runner reset")
