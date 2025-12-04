"""
Utility script to create ground truth CSV templates for datasets.

Usage:
    python create_ground_truth_template.py --dataset data/raw_data/dataset1_test
    python create_ground_truth_template.py --dataset data/raw_data/dataset1_test --output custom_labels.csv
"""

import argparse
import csv
import logging
from pathlib import Path
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def count_frames_in_dataset(dataset_path: str) -> int:
    """
    Count number of frames in a dataset.
    
    Args:
        dataset_path: Path to dataset folder or video file
        
    Returns:
        Number of frames
    """
    dataset_path = Path(dataset_path)
    
    if dataset_path.suffix == '.mp4':
        # Video file
        cap = cv2.VideoCapture(str(dataset_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {dataset_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count
    else:
        # Image folder
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [
            f for f in dataset_path.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        return len(image_files)


def create_template(
    dataset_path: str,
    output_path: str = None,
    default_value: int = 0
) -> str:
    """
    Create ground truth CSV template.
    
    Args:
        dataset_path: Path to dataset
        output_path: Output CSV path (default: data/{dataset_name}_labels.csv)
        default_value: Default label value (0 or 1)
        
    Returns:
        Path to created CSV file
    """
    dataset_path = Path(dataset_path)
    
    # Count frames
    num_frames = count_frames_in_dataset(dataset_path)
    logger.info(f"Detected {num_frames} frames in {dataset_path}")
    
    # Determine output path
    if output_path is None:
        dataset_name = dataset_path.stem if dataset_path.suffix == '.mp4' else dataset_path.name
        # Save to data/labels/ folder
        labels_dir = Path('data/labels')
        labels_dir.mkdir(parents=True, exist_ok=True)
        output_path = labels_dir / f"{dataset_name}_labels.csv"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'isNovelTruth'])
        
        for frame_idx in range(num_frames):
            writer.writerow([frame_idx, default_value])
    
    logger.info(f"✓ Created ground truth template: {output_path}")
    logger.info(f"  Total frames: {num_frames}")
    logger.info(f"  Default label: {default_value} ({'novel' if default_value == 1 else 'non-novel'})")
    logger.info("\nNext steps:")
    logger.info(f"  1. Open {output_path} in a text editor or Excel")
    logger.info("  2. Review each frame and update isNovelTruth:")
    logger.info("     - Set to 1 for frames with novel content")
    logger.info("     - Set to 0 for frames with no novel content")
    logger.info("  3. Save the file")
    logger.info(f"  4. Run tests: python run_tests.py --dataset {dataset_path}")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Create ground truth CSV template for a dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create template for image folder
  python create_ground_truth_template.py --dataset data/dataset1_test
  
  # Create template for video file
  python create_ground_truth_template.py --dataset data/video.mp4
  
  # Custom output path
  python create_ground_truth_template.py --dataset data/dataset1_test --output my_labels.csv
  
  # Default to "novel" (1) instead of "non-novel" (0)
  python create_ground_truth_template.py --dataset data/dataset1_test --default 1
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to dataset folder or video file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV path (default: data/{dataset_name}_labels.csv)'
    )
    
    parser.add_argument(
        '--default',
        type=int,
        choices=[0, 1],
        default=0,
        help='Default label value: 0=non-novel, 1=novel (default: 0)'
    )
    
    args = parser.parse_args()
    
    try:
        create_template(
            dataset_path=args.dataset,
            output_path=args.output,
            default_value=args.default
        )
    except Exception as e:
        logger.error(f"Failed to create template: {e}")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
