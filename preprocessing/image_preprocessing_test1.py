import os
import cv2
import numpy as np
from pathlib import Path
from typing import Generator, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FramePreprocessor:
    """Handles loading and preprocessing images from folders or video files."""
    
    def __init__(self, target_size=(240, 320)):
        """
        Initialize preprocessor.
        
        Args:
            target_size: (height, width) for uniform resizing
        """
        self.target_size = target_size
        self.frames = []
        self.frame_index = 0
        
    def load_images(self, directory_path: str) -> List[np.ndarray]:
        """
        Load and preprocess images from a directory in sorted order.
        
        Args:
            directory_path: Path to folder containing images
            
        Returns:
            List of preprocessed frames (grayscale, normalized)
            
        Raises:
            ValueError: If directory doesn't exist or no images found
        """
        dir_path = Path(directory_path)
        if not dir_path.exists():
            raise ValueError(f"Directory not found: {directory_path}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = sorted([
            f for f in dir_path.iterdir() 
            if f.suffix.lower() in image_extensions
        ])
        
        if not image_files:
            raise ValueError(f"No images found in {directory_path}")
        
        logger.info(f"Found {len(image_files)} images in {directory_path}")
        
        frames = []
        for idx, img_path in enumerate(image_files):
            frame = cv2.imread(str(img_path))
            if frame is None:
                logger.warning(f"Failed to load {img_path}")
                continue
            
            # Convert to grayscale and preprocess
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, self.target_size[::-1])
            frame = frame.astype(np.float32) / 255.0
            
            frames.append(frame)
            logger.info(f"Loaded frame {idx}: {img_path.name}")
        
        self.frames = frames
        self.frame_index = 0
        return frames
    
    def load_video(self, video_path: str, difference_window: int = 1) -> None:
        """
        Load video file for frame extraction.
        
        Args:
            video_path: Path to .mp4 video file
            difference_window: Frames to look back for difference metric
            
        Raises:
            ValueError: If video file doesn't exist or cannot be opened
        """
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video loaded: {total_frames} frames at {video_path}")
        
        frames = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, self.target_size[::-1])
            frame = frame.astype(np.float32) / 255.0
            
            frames.append(frame)
            if idx % 30 == 0:
                logger.info(f"Extracted frame {idx}")
            idx += 1
        
        cap.release()
        self.frames = frames
        self.frame_index = 0
        self.difference_window = difference_window
        logger.info(f"Loaded {len(frames)} frames from video")
    
    def get_next_frame(self) -> Generator[tuple, None, None]:
        """
        Yield frames sequentially as (frame_index, frame_data).
        
        Yields:
            Tuple of (frame_index, frame_array)
        """
        while self.frame_index < len(self.frames):
            frame = self.frames[self.frame_index]
            yield self.frame_index, frame
            self.frame_index += 1