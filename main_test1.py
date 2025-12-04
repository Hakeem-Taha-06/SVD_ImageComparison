import time
import argparse
from dataclasses import dataclass
from typing import Dict, Any
import psutil
import os

from ai.detection_ai_test1 import AIDetectionModule
from preprocessing.image_preprocessing_test1 import FramePreprocessor
from svd.svd_test2 import SVDNoveltyDetector
import csv
import logging
logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """Track system performance metrics."""
    total_frames: int = 0
    novel_frames: int = 0
    ai_calls: int = 0
    total_error: float = 0.0
    total_time: float = 0.0
    frame_times: list = None
    peak_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    avg_cpu_percent: float = 0.0
    
    def __post_init__(self):
        if self.frame_times is None:
            self.frame_times = []

class SVDGatekeeperSystem:
    """Main orchestration class for novelty detection system."""
    
    def __init__(self, threshold: float = 0.15, compression_rank: int = 20,
                 difference_window: int = 1, target_size: tuple = (240, 320),
                 display_plots: bool = False, use_mock_ai: bool = False):
        """
        Initialize the gatekeeper system.
        
        Args:
            threshold: Novelty threshold τ
            compression_rank: SVD rank k
            difference_window: Frame difference window W
            target_size: (height, width) for frame resizing
            display_plots: Whether to display reconstruction plots
            use_mock_ai: Use mock AI detection instead of real YOLO
        """
        self.threshold = threshold
        self.compression_rank = compression_rank
        self.difference_window = difference_window
        self.target_size = target_size
        self.display_plots = display_plots
        
        self.preprocessor = FramePreprocessor(target_size=target_size)
        self.svd_detector = SVDNoveltyDetector(compression_rank=compression_rank)
        self.ai_module = AIDetectionModule(use_mock=use_mock_ai)
        self.metrics = SystemMetrics()
        
        logger.info("=" * 70)
        logger.info("SVD Novelty Detection Gatekeeper Initialized")
        logger.info(f"  Threshold τ = {threshold}")
        logger.info(f"  SVD Rank k = {compression_rank}")
        logger.info(f"  Difference Window W = {difference_window}")
        logger.info("=" * 70)
        
        self.process = psutil.Process(os.getpid())
        self.cpu_samples = []
    
    def process_input(self, input_path: str) -> None:
        """
        Detect input type and load accordingly.
        
        Args:
            input_path: Path to folder or .mp4 file
        """
        if input_path.endswith('.mp4'):
            logger.info(f"Loading video: {input_path}")
            self.preprocessor.load_video(input_path, self.difference_window)
        else:
            logger.info(f"Loading images from: {input_path}")
            self.preprocessor.load_images(input_path)
    
    def _sample_cpu_usage(self) -> None:
        """Sample CPU usage for averaging."""
        try:
            cpu_percent = self.process.cpu_percent(interval=None)
            self.cpu_samples.append(cpu_percent)
        except:
            pass
    
    def process_frames(self) -> None:
        """
        Main processing loop: iterate frames, detect novelty, trigger AI.
        """
        logger.info("\nStarting frame processing...")
        logger.info("-" * 70)
        
        self.reference_frame_error = None
        self.cpu_samples = []
        
        for frame_idx, frame in self.preprocessor.get_next_frame():
            frame_start_time = time.time()
            self.metrics.total_frames += 1
            
            # Compute reconstruction error
            error, frame_reconstructed = self.svd_detector.compare_frames(frame)
            if self.reference_frame_error == None:
                self.reference_frame_error = error  # Set baseline for first frame
                detections = self.ai_module.run_detection_ai(frame)
                self.metrics.ai_calls += 1

            self.metrics.total_error += error
            
            # Decision: is this frame novel?
            is_novel = abs(error - self.reference_frame_error) > self.threshold
            
            if is_novel:
                self.metrics.novel_frames += 1
                logger.warning(f"[Frame {frame_idx}] NOVELTY DETECTED | Error: {error:.4f}")
                logger.warning(f"[Frame {frame_idx}] NOVELTY DETECTED | Error difference: {abs(error - self.reference_frame_error):.4f} > τ:{self.threshold}")
                self.reference_frame_error = error  # Update reference error
                
                # Trigger AI model
                detections = self.ai_module.run_detection_ai(frame)
                self.metrics.ai_calls += 1
                
                # Optionally plot
                if self.display_plots and frame_idx % 10 == 0:
                    plot_name = f"novelty_frame_{frame_idx}.png"
                    self.svd_detector.plot_reconstruction(frame, frame_reconstructed, 
                                                         error, save_path=plot_name)
            else:
                logger.info(f"[Frame {frame_idx}] Skipped (not novel) | Error: {error:.4f}")
                logger.info(f"[Frame {frame_idx}] Skipped (not novel) | Error difference: {abs(error - self.reference_frame_error):.4f}")
            
            frame_time = time.time() - frame_start_time
            self.metrics.frame_times.append(frame_time)
            self.metrics.total_time += frame_time
            self._sample_cpu_usage()
        
        # Calculate resource metrics
        try:
            self.metrics.peak_memory_mb = self.process.memory_info().rss / 1024 / 1024
        except:
            self.metrics.peak_memory_mb = 0.0
        
        if self.cpu_samples:
            self.metrics.avg_cpu_percent = sum(self.cpu_samples) / len(self.cpu_samples)
            self.metrics.peak_cpu_percent = max(self.cpu_samples)
        
        logger.info("-" * 70)
    
    def print_summary_report(self) -> None:
        """Print comprehensive performance summary."""
        if self.metrics.total_frames == 0:
            logger.warning("No frames processed")
            return
        
        efficiency = (1.0 - self.metrics.ai_calls / self.metrics.total_frames) * 100
        avg_error = self.metrics.total_error / self.metrics.total_frames
        avg_time = self.metrics.total_time / self.metrics.total_frames
        
        logger.info("\n" + "=" * 70)
        logger.info("FINAL PERFORMANCE REPORT")
        logger.info("=" * 70)
        logger.info(f"Total Frames Processed:    {self.metrics.total_frames}")
        logger.info(f"Novel Frames Detected:     {self.metrics.novel_frames}")
        logger.info(f"AI Model Calls:            {self.metrics.ai_calls}")
        logger.info(f"Gatekeeper Efficiency:     {efficiency:.1f}% (avoided {self.metrics.total_frames - self.metrics.ai_calls} AI calls)")
        logger.info(f"Average Reconstruction Error: {avg_error:.4f}")
        logger.info(f"Avg Processing Time/Frame: {avg_time*1000:.2f} ms")
        logger.info(f"Total Processing Time:     {self.metrics.total_time:.2f} seconds")
        logger.info(f"Throughput:                {self.metrics.total_frames / self.metrics.total_time:.1f} fps")
        logger.info(f"Peak Memory Usage:         {self.metrics.peak_memory_mb:.1f} MB")
        logger.info(f"Average CPU Usage:         {self.metrics.avg_cpu_percent:.1f}%")
        logger.info(f"Peak CPU Usage:            {self.metrics.peak_cpu_percent:.1f}%")
        logger.info("=" * 70 + "\n")

def warmup_ai(target_size: tuple, use_mock_ai: bool) -> None:
    """Run AI warmup pass before main test loop."""
    logger.info("\n" + "=" * 70)
    logger.info("WARMUP PHASE: Running AI detection once before test loop")
    logger.info("=" * 70)
    
    try:
        ai_module = AIDetectionModule(use_mock=use_mock_ai)
        preprocessor = FramePreprocessor(target_size=target_size)
        preprocessor.load_images("data/")
        
        # Get first frame for warmup
        for frame_idx, frame in preprocessor.get_next_frame():
            logger.info(f"Running warmup detection on first frame...")
            detections = ai_module.run_detection_ai(frame)
            logger.info(f"Warmup complete. AI model loaded and ready.")
            break
    except Exception as e:
        logger.warning(f"Warmup phase encountered an issue (non-critical): {e}")
    
    logger.info("=" * 70 + "\n")

def main():
    """Main entry point with CLI argument parsing."""
    
    # Parse target size

    threshold = 0.01
    rank = 1
    window = 1
    target_size = (240, 320)
    plots = True
    mock_ai = False
    input_path = "data/raw_data" # Default path; replace with actual path or CLI arg

    # Run AI warmup before test loop
    warmup_ai(target_size, mock_ai)

    with open('results.csv', mode='w', newline='') as log_file:
        fieldnames = ['Threshold', 'Rank', 'Total Frames', 'Novel Frames', 
                      'AI Calls', 'Efficiency (%)', 'Avg Time (ms)', 'Throughput (fps)',
                      'Peak Memory (MB)', 'Avg CPU (%)', 'Peak CPU (%)']
        writer = csv.writer(log_file)
        writer.writerow(fieldnames)

        # Create and run system
        
        while rank < 80:
            threshold = 0.01
            while threshold < 0.2:
                system = SVDGatekeeperSystem(
                    threshold=round(threshold, 2),
                    compression_rank=rank,
                    difference_window=window,
                    target_size=target_size,
                    display_plots=plots,
                    use_mock_ai=mock_ai
                )
                try:
                    system.process_input(input_path)
                    system.process_frames()
                    system.print_summary_report()
                    writer.writerow([f"{threshold:.2f}", f"{rank}", f"{system.metrics.total_frames}", f"{system.metrics.novel_frames}", f"{system.metrics.ai_calls}",
                                    f"{(1.0 - system.metrics.ai_calls / system.metrics.total_frames) * 100}",
                                    f"{((system.metrics.total_time / system.metrics.total_frames)*1000):.2f}",
                                    f"{(system.metrics.total_frames / system.metrics.total_time):.2f}",
                                    f"{system.metrics.peak_memory_mb:.1f}",
                                    f"{system.metrics.avg_cpu_percent:.1f}",
                                    f"{system.metrics.peak_cpu_percent:.1f}"])
                except Exception as e:
                    logger.error(f"Error during processing: {e}", exc_info=True)
                threshold += 0.01
                # Reset for next rank loop
                rank+=1
            

if __name__ == "__main__":
    main()

# =============================================================================
# USAGE EXAMPLES
# =============================================================================
# 
# 1. Process images with default settings:
#    python main.py /path/to/image/folder
#
# 2. Process video with custom threshold and rank:
#    python main.py video.mp4 --threshold 0.2 --rank 30
#
# 3. Display plots for novel frames:
#    python main.py images/ --plots
#
# 4. Use mock AI (no YOLO required):
#    python main.py data/ --mock-ai
#
# 5. All parameters:
#    python main.py video.mp4 --threshold 0.15 --rank 25 \
#                  --window 2 --size 480,640 --plots --mock-ai
# =============================================================================