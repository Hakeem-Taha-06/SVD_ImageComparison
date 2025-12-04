"""
Ground truth loader for novelty detection evaluation.
Loads and validates CSV files with frame-level ground truth labels.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class GroundTruthLoader:
    """Load and manage ground truth labels for novelty detection."""
    
    def __init__(self):
        """Initialize ground truth loader."""
        self.labels: Dict[int, bool] = {}
        self.dataset_name: Optional[str] = None
    
    def load_from_csv(self, csv_path: str) -> Dict[int, bool]:
        """
        Load ground truth labels from CSV file.
        
        Expected CSV format:
        frame,isNovelTruth
        0,1
        1,0
        2,1
        ...
        
        Args:
            csv_path: Path to CSV file with ground truth labels
            
        Returns:
            Dictionary mapping frame_index -> is_novel (bool)
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV format is invalid
        """
        csv_file = Path(csv_path)
        
        if not csv_file.exists():
            raise FileNotFoundError(f"Ground truth CSV not found: {csv_path}")
        
        labels = {}
        
        try:
            with open(csv_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # Validate headers - support multiple column name formats
                frame_col = None
                novel_col = None
                
                for field in reader.fieldnames:
                    if field.lower() in ['frame', 'frame_index', 'frame_idx']:
                        frame_col = field
                    if field.lower() in ['isnoveltruth', 'is_novel', 'novel', 'is_novel_truth']:
                        novel_col = field
                
                if frame_col is None or novel_col is None:
                    raise ValueError(
                        f"CSV must have frame and novelty columns. "
                        f"Found: {reader.fieldnames}. "
                        f"Expected columns like: 'frame'/'frame_index' and 'isNovelTruth'/'is_novel'"
                    )
                
                logger.info(f"Using columns: frame='{frame_col}', novel='{novel_col}'")
                
                for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                    try:
                        frame_idx = int(row[frame_col])
                        is_novel_str = row[novel_col].strip()
                        
                        # Parse boolean value (accept 1/0, true/false, yes/no)
                        is_novel = self._parse_boolean(is_novel_str)
                        
                        labels[frame_idx] = is_novel
                        
                    except ValueError as e:
                        logger.warning(f"Invalid data at row {row_num}: {e}. Skipping.")
                        continue
        
        except Exception as e:
            raise ValueError(f"Failed to parse CSV {csv_path}: {e}")
        
        if not labels:
            raise ValueError(f"No valid labels found in {csv_path}")
        
        self.labels = labels
        self.dataset_name = csv_file.stem.replace('_labels', '')
        
        logger.info(f"Loaded {len(labels)} ground truth labels from {csv_path}")
        logger.info(f"  Novel frames: {sum(labels.values())}")
        logger.info(f"  Non-novel frames: {len(labels) - sum(labels.values())}")
        
        return labels
    
    def load_from_dataset_folder(self, dataset_path: str) -> Dict[int, bool]:
        """
        Load ground truth CSV from dataset folder.
        
        Looks for CSV file at: data/labels/{dataset_name}_labels.csv
        
        Args:
            dataset_path: Path to dataset folder (e.g., "data/raw_data/dataset1_test")
            
        Returns:
            Dictionary mapping frame_index -> is_novel (bool)
            
        Raises:
            FileNotFoundError: If labels CSV not found
        """
        dataset_folder = Path(dataset_path)
        dataset_name = dataset_folder.name
        
        # Look for labels CSV in data/labels/ folder
        labels_csv = dataset_folder.parent.parent / "labels" / f"{dataset_name}_labels.csv"
        
        if not labels_csv.exists():
            raise FileNotFoundError(
                f"Ground truth CSV not found: {labels_csv}\n"
                f"Expected location: data/labels/{dataset_name}_labels.csv"
            )
        
        return self.load_from_csv(str(labels_csv))
    
    def get_label(self, frame_index: int) -> Optional[bool]:
        """
        Get ground truth label for a specific frame.
        
        Args:
            frame_index: Frame index
            
        Returns:
            True if novel, False if not novel, None if not found
        """
        return self.labels.get(frame_index)
    
    def has_labels(self) -> bool:
        """Check if ground truth labels are loaded."""
        return len(self.labels) > 0
    
    def get_all_labels(self) -> Dict[int, bool]:
        """Get all ground truth labels."""
        return self.labels.copy()
    
    def get_frame_indices(self) -> list:
        """Get list of all frame indices with ground truth."""
        return sorted(self.labels.keys())
    
    def get_statistics(self) -> dict:
        """
        Get statistics about ground truth labels.
        
        Returns:
            Dictionary with label statistics
        """
        if not self.labels:
            return {}
        
        novel_count = sum(self.labels.values())
        total_count = len(self.labels)
        
        return {
            'total_frames': total_count,
            'novel_frames': novel_count,
            'non_novel_frames': total_count - novel_count,
            'novel_ratio': novel_count / total_count if total_count > 0 else 0.0,
            'dataset_name': self.dataset_name
        }
    
    def validate_frame_coverage(self, total_frames: int, allow_partial: bool = True) -> bool:
        """
        Validate that ground truth covers expected frames.
        
        Args:
            total_frames: Expected total number of frames
            allow_partial: If True, allow partial coverage (default: True)
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If validation fails and allow_partial is False
        """
        if not self.labels:
            raise ValueError("No ground truth labels loaded")
        
        max_frame_idx = max(self.labels.keys())
        min_frame_idx = min(self.labels.keys())
        
        # Check if frame indices are within range
        if max_frame_idx >= total_frames:
            msg = f"Ground truth contains frames beyond total_frames: {max_frame_idx} >= {total_frames}"
            if not allow_partial:
                raise ValueError(msg)
            logger.warning(msg)
        
        # Check for gaps
        expected_indices = set(range(total_frames))
        actual_indices = set(self.labels.keys())
        missing_indices = expected_indices - actual_indices
        
        if missing_indices:
            coverage = len(actual_indices) / total_frames * 100
            msg = f"Ground truth coverage: {coverage:.1f}% ({len(actual_indices)}/{total_frames} frames)"
            logger.info(msg)
            
            if not allow_partial and len(missing_indices) > 0:
                raise ValueError(f"Missing ground truth for {len(missing_indices)} frames")
        
        return True
    
    @staticmethod
    def _parse_boolean(value: str) -> bool:
        """
        Parse boolean value from string.
        
        Args:
            value: String representation of boolean
            
        Returns:
            Boolean value
            
        Raises:
            ValueError: If value cannot be parsed
        """
        value_lower = value.lower().strip()
        
        if value_lower in ('1', 'true', 'yes', 't', 'y'):
            return True
        elif value_lower in ('0', 'false', 'no', 'f', 'n'):
            return False
        else:
            raise ValueError(f"Cannot parse boolean from: '{value}'")
    
    @staticmethod
    def create_template_csv(output_path: str, num_frames: int) -> None:
        """
        Create a template ground truth CSV file.
        
        Args:
            output_path: Path for output CSV file
            num_frames: Number of frames to include
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'isNovelTruth'])
            
            for frame_idx in range(num_frames):
                # Default to all frames as non-novel (0)
                writer.writerow([frame_idx, 0])
        
        logger.info(f"Created template ground truth CSV: {output_path}")
        logger.info(f"  Contains {num_frames} frames (all labeled as non-novel)")
        logger.info(f"  Please edit the file to mark novel frames with 1")
