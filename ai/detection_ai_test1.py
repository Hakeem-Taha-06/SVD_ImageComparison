import logging
import numpy as np
logger = logging.getLogger(__name__)

class AIDetectionModule:
    """Wraps heavy AI model (YOLO) for object detection on novel frames."""
    
    def __init__(self, use_mock: bool = False):
        """
        Initialize AI module.
        
        Args:
            use_mock: If True, use mock detections (YOLO not required)
        """
        self.use_mock = use_mock
        self.model = None
        self.detection_count = 0
        
        if not use_mock:
            self._load_yolo_model()
    
    def _load_yolo_model(self) -> None:
        """Attempt to load YOLOv8 model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO('yolov8n.pt')
            logger.info("YOLOv8 model loaded successfully")
        except ImportError:
            logger.warning("ultralytics not installed. Using mock detections.")
            self.use_mock = True
        except Exception as e:
            logger.warning(f"Failed to load YOLO model: {e}. Using mock detections.")
            self.use_mock = True
    
    def run_detection_ai(self, frame: np.ndarray) -> list:
        """
        Run object detection on a frame.
        
        Args:
            frame: Grayscale frame (2D array, values 0-1) or BGR frame (3D)
            
        Returns:
            List of detections with object names and confidence scores
        """
        self.detection_count += 1
        
        if self.use_mock or self.model is None:
            logger.info("Mock detection: frame processed")
            return [{"object": "placeholder_detection", "confidence": 0.95}]
        
        # Convert grayscale to BGR if needed
        if frame.ndim == 2:
            frame_bgr = np.stack([frame] * 3, axis=2)
        else:
            frame_bgr = frame
        
        # Ensure uint8 range
        frame_bgr = (frame_bgr * 255).astype(np.uint8)
        
        # Run detection
        results = self.model(frame_bgr, verbose=False)
        detections = []
        
        for result in results:
            for box in result.boxes:
                obj_name = result.names[int(box.cls)]
                confidence = float(box.conf)
                detections.append({
                    "object": obj_name,
                    "confidence": confidence
                })
                logger.info(f"  Detected: {obj_name} ({confidence:.2f})")
        
        if not detections:
            logger.info("  No objects detected")
        
        return detections if detections else [{"object": "none", "confidence": 0.0}]
