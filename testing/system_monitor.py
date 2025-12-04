"""
System monitoring utilities for tracking CPU and memory usage during experiments.
"""

import psutil
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SystemMonitor:
    """Monitor CPU and memory usage during experiments."""
    
    def __init__(self):
        """Initialize system monitor."""
        self.process = psutil.Process()
        self._baseline_memory = None
        
    def get_cpu_percent(self, interval: float = 0.1) -> float:
        """
        Get current CPU usage percentage.
        
        Args:
            interval: Measurement interval in seconds
            
        Returns:
            CPU usage percentage (0-100)
        """
        try:
            # Get CPU percent for this process
            cpu_percent = self.process.cpu_percent(interval=interval)
            return cpu_percent
        except Exception as e:
            logger.warning(f"Failed to get CPU usage: {e}")
            return 0.0
    
    def get_memory_mb(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in megabytes
        """
        try:
            # Get memory info for this process
            mem_info = self.process.memory_info()
            # Convert bytes to MB
            memory_mb = mem_info.rss / (1024 * 1024)
            return memory_mb
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return 0.0
    
    def get_system_stats(self, interval: float = 0.1) -> dict:
        """
        Get comprehensive system statistics.
        
        Args:
            interval: CPU measurement interval
            
        Returns:
            Dictionary with system statistics
        """
        try:
            cpu_percent = self.get_cpu_percent(interval=interval)
            memory_mb = self.get_memory_mb()
            
            # Get system-wide stats
            sys_cpu = psutil.cpu_percent(interval=0)
            sys_mem = psutil.virtual_memory()
            
            return {
                'process_cpu_percent': cpu_percent,
                'process_memory_mb': memory_mb,
                'system_cpu_percent': sys_cpu,
                'system_memory_percent': sys_mem.percent,
                'system_memory_available_mb': sys_mem.available / (1024 * 1024)
            }
        except Exception as e:
            logger.warning(f"Failed to get system stats: {e}")
            return {
                'process_cpu_percent': 0.0,
                'process_memory_mb': 0.0,
                'system_cpu_percent': 0.0,
                'system_memory_percent': 0.0,
                'system_memory_available_mb': 0.0
            }
    
    def set_baseline(self) -> None:
        """Set baseline memory usage for relative measurements."""
        self._baseline_memory = self.get_memory_mb()
        logger.info(f"Baseline memory set: {self._baseline_memory:.2f} MB")
    
    def get_memory_delta_mb(self) -> Optional[float]:
        """
        Get memory usage relative to baseline.
        
        Returns:
            Memory delta in MB, or None if baseline not set
        """
        if self._baseline_memory is None:
            return None
        
        current_memory = self.get_memory_mb()
        return current_memory - self._baseline_memory
    
    def log_current_usage(self) -> None:
        """Log current resource usage."""
        stats = self.get_system_stats()
        logger.info(
            f"Resource Usage - "
            f"Process CPU: {stats['process_cpu_percent']:.1f}%, "
            f"Process Memory: {stats['process_memory_mb']:.1f} MB, "
            f"System CPU: {stats['system_cpu_percent']:.1f}%, "
            f"System Memory: {stats['system_memory_percent']:.1f}%"
        )


class ResourceTracker:
    """Track resource usage over time with averaging."""
    
    def __init__(self, monitor: Optional[SystemMonitor] = None):
        """
        Initialize resource tracker.
        
        Args:
            monitor: SystemMonitor instance (creates new one if None)
        """
        self.monitor = monitor or SystemMonitor()
        self.cpu_samples = []
        self.memory_samples = []
    
    def record_sample(self, interval: float = 0.1) -> dict:
        """
        Record a resource usage sample.
        
        Args:
            interval: CPU measurement interval
            
        Returns:
            Dictionary with current measurements
        """
        cpu = self.monitor.get_cpu_percent(interval=interval)
        memory = self.monitor.get_memory_mb()
        
        self.cpu_samples.append(cpu)
        self.memory_samples.append(memory)
        
        return {
            'cpu_percent': cpu,
            'memory_mb': memory
        }
    
    def get_averages(self) -> dict:
        """
        Get average resource usage.
        
        Returns:
            Dictionary with average CPU and memory
        """
        if not self.cpu_samples:
            return {'avg_cpu_percent': 0.0, 'avg_memory_mb': 0.0}
        
        return {
            'avg_cpu_percent': sum(self.cpu_samples) / len(self.cpu_samples),
            'avg_memory_mb': sum(self.memory_samples) / len(self.memory_samples)
        }
    
    def get_peak_usage(self) -> dict:
        """
        Get peak resource usage.
        
        Returns:
            Dictionary with peak CPU and memory
        """
        if not self.cpu_samples:
            return {'peak_cpu_percent': 0.0, 'peak_memory_mb': 0.0}
        
        return {
            'peak_cpu_percent': max(self.cpu_samples),
            'peak_memory_mb': max(self.memory_samples)
        }
    
    def reset(self) -> None:
        """Reset all samples."""
        self.cpu_samples.clear()
        self.memory_samples.clear()
    
    def get_statistics(self) -> dict:
        """
        Get comprehensive statistics.
        
        Returns:
            Dictionary with min, max, avg, std for CPU and memory
        """
        if not self.cpu_samples:
            return {}
        
        import statistics
        
        return {
            'cpu': {
                'min': min(self.cpu_samples),
                'max': max(self.cpu_samples),
                'avg': statistics.mean(self.cpu_samples),
                'std': statistics.stdev(self.cpu_samples) if len(self.cpu_samples) > 1 else 0.0
            },
            'memory': {
                'min': min(self.memory_samples),
                'max': max(self.memory_samples),
                'avg': statistics.mean(self.memory_samples),
                'std': statistics.stdev(self.memory_samples) if len(self.memory_samples) > 1 else 0.0
            },
            'sample_count': len(self.cpu_samples)
        }
