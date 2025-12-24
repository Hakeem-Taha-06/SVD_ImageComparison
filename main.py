"""
SVD Gatekeeping Prototype - Real-Time Novelty Detection

A real-time application that processes live video feed, detects novelty
using SVD reconstruction error, and runs AI detection on novel frames.

Controls:
- Rank slider: Adjust SVD compression rank (click and drag)
- Threshold slider: Adjust novelty detection sensitivity (click and drag)
- ESC: Exit application
"""

import cv2
import numpy as np
import time
from src.core.svd_detector import SVDNoveltyDetector
from src.core.ai_detector import AIDetectionModule

# ============================================================================
# CONFIGURATION
# ============================================================================

VIDEO_URL = "http://192.168.1.61:8080/video"  # IP camera URL
INITIAL_RANK = 20          # Starting SVD compression rank
INITIAL_THRESHOLD = 5      # Starting threshold (scaled by 100 for slider)
MAX_RANK = 100             # Maximum compression rank
MAX_THRESHOLD = 50         # Maximum threshold (will be divided by 100)

# UI Color Theme (BGR format - use hex codes converted to BGR)
# Format: (B, G, R) - note: OpenCV uses BGR, not RGB
COLOR_PRIMARY = (144, 119, 4)     # #286498 - Deep blue (panels, accents)
COLOR_SECONDARY = (221, 233, 238)     # #3C3C3C - Dark gray (backgrounds)
COLOR_ACCENT = (246, 252, 255)     # #FAC864 - Gold/yellow (highlights, sliders)
COLOR_ALERT = (60, 60, 220)        # #DC3C3C - Red (novelty alert)
COLOR_TEXT = (50, 50, 50)       # #F0F0F0 - Dark gray (text)
COLOR_TEXT_DIM = (140, 140, 140)   # #8C8C8C - Dimmed text

# ============================================================================


def draw_rounded_rect(img, pt1, pt2, color, radius=15, thickness=-1):
    """Draw a rounded rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Ensure coordinates are valid
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    # Clamp radius to half the smallest dimension
    radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    
    if thickness == -1:
        # Filled rounded rectangle
        # Draw rectangles
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        # Draw corner circles
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        # Outline only
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


class CustomSlider:
    """Modern slider with circle handle."""
    
    def __init__(self, x, y, width, min_val, max_val, initial_val, label):
        self.x = x
        self.y = y
        self.width = width
        self.height = 30
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.dragging = False
        self.handle_radius = 12
        
    def get_handle_x(self):
        """Get handle X position based on value."""
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        return int(self.x + 20 + ratio * (self.width - 40))
    
    def set_value_from_x(self, mouse_x):
        """Set value based on mouse X position."""
        track_start = self.x + 20
        track_end = self.x + self.width - 20
        ratio = (mouse_x - track_start) / (track_end - track_start)
        ratio = max(0, min(1, ratio))
        self.value = self.min_val + ratio * (self.max_val - self.min_val)
        # Round to integer for rank slider
        if self.min_val == 1 or self.max_val > 10:
            self.value = int(round(self.value))
    
    def is_handle_clicked(self, mouse_x, mouse_y):
        """Check if mouse is on handle."""
        handle_x = self.get_handle_x()
        handle_y = self.y + self.height // 2
        dist = np.sqrt((mouse_x - handle_x)**2 + (mouse_y - handle_y)**2)
        return dist <= self.handle_radius + 5
    
    def is_track_clicked(self, mouse_x, mouse_y):
        """Check if mouse is on track area."""
        return (self.x <= mouse_x <= self.x + self.width and 
                self.y <= mouse_y <= self.y + self.height)
    
    def draw(self, img):
        """Draw the slider on image."""
        # Track background (rounded)
        track_y = self.y + self.height // 2 - 4
        draw_rounded_rect(img, (self.x + 15, track_y), 
                         (self.x + self.width - 15, track_y + 8), 
                         COLOR_SECONDARY, radius=4)
        
        # Track fill (up to handle)
        handle_x = self.get_handle_x()
        if handle_x > self.x + 20:
            draw_rounded_rect(img, (self.x + 15, track_y), 
                             (handle_x, track_y + 8), 
                             COLOR_PRIMARY, radius=4)
        
        # Handle (circle)
        handle_y = self.y + self.height // 2
        cv2.circle(img, (handle_x, handle_y), self.handle_radius, COLOR_ACCENT, -1)
        cv2.circle(img, (handle_x, handle_y), self.handle_radius, COLOR_TEXT, 2)
        
        # Label
        cv2.putText(img, self.label, (self.x, self.y - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PRIMARY, 2)
        
        # Value display
        if isinstance(self.value, int) or self.value == int(self.value):
            val_text = f"{int(self.value)}"
        else:
            val_text = f"{self.value:.2f}"
        cv2.putText(img, val_text, (self.x + self.width - 40, self.y - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT_DIM, 2)


class GatekeeperApp:
    """Real-time SVD gatekeeping application with modern UI."""
    
    def __init__(self, video_url: str):
        self.video_url = video_url
        self.window_name = "SVD Gatekeeper"
        
        # Current parameters
        self.current_rank = INITIAL_RANK
        self.current_threshold = INITIAL_THRESHOLD / 100.0
        
        # SVD detector
        self.svd = SVDNoveltyDetector(compression_rank=self.current_rank)
        
        # AI detector
        self.ai = AIDetectionModule(use_mock=False)
        
        # State
        self.detected_objects = []
        self.reference_error = None
        self.is_novel = False
        self.show_alert = False
        self.alert_start_time = 0.0
        self.alert_duration = 0.3
        self.last_error = 0.0
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        # UI components (will be initialized in run())
        self.sliders = []
        self.base_display_width = 1920  # HD resolution
        self.base_display_height = 1080
        self.current_window_width = 1920  # Current window size (for scaling)
        self.current_window_height = 1080
        self.scale_x = 1.0
        self.scale_y = 1.0
        
        # Video connection state
        self.video_connected = False
        self.reconnect_requested = False
        
        # Reconnect button position (base coords, will be set in build_display)
        self.reconnect_btn = {'x': 0, 'y': 0, 'w': 120, 'h': 40}
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for sliders."""
        # For each slider, check clicks using scaled positions
        # Scale factors convert base positions to current window positions
        
        if event == cv2.EVENT_LBUTTONDOWN:
            for slider in self.sliders:
                # Scale slider bounds to current window size
                scaled_x = slider.x * self.scale_x
                scaled_y = slider.y * self.scale_y
                scaled_width = slider.width * self.scale_x
                scaled_height = slider.height * self.scale_y
                scaled_handle_x = slider.get_handle_x() * self.scale_x
                scaled_handle_y = (slider.y + slider.height // 2) * self.scale_y
                scaled_radius = slider.handle_radius * max(self.scale_x, self.scale_y)
                
                # Check handle click
                dist = np.sqrt((x - scaled_handle_x)**2 + (y - scaled_handle_y)**2)
                handle_clicked = dist <= scaled_radius + 5
                
                # Check track click
                track_clicked = (scaled_x <= x <= scaled_x + scaled_width and 
                                scaled_y - 10 <= y <= scaled_y + scaled_height + 10)
                
                if handle_clicked or track_clicked:
                    slider.dragging = True
                    # Convert x back to base coords for value calculation
                    base_x = x / self.scale_x if self.scale_x > 0 else x
                    slider.set_value_from_x(base_x)
                    self.update_from_sliders()
                    
        elif event == cv2.EVENT_MOUSEMOVE:
            for slider in self.sliders:
                if slider.dragging:
                    # Convert x back to base coords for value calculation
                    base_x = x / self.scale_x if self.scale_x > 0 else x
                    slider.set_value_from_x(base_x)
                    self.update_from_sliders()
                    
        elif event == cv2.EVENT_LBUTTONUP:
            for slider in self.sliders:
                slider.dragging = False
            
            # Check reconnect button click (only when video not connected)
            if not self.video_connected:
                btn = self.reconnect_btn
                scaled_btn_x = btn['x'] * self.scale_x
                scaled_btn_y = btn['y'] * self.scale_y
                scaled_btn_w = btn['w'] * self.scale_x
                scaled_btn_h = btn['h'] * self.scale_y
                if (scaled_btn_x <= x <= scaled_btn_x + scaled_btn_w and
                    scaled_btn_y <= y <= scaled_btn_y + scaled_btn_h):
                    self.reconnect_requested = True
                    print("Reconnect button clicked")
    
    def update_from_sliders(self):
        """Update parameters from slider values."""
        self.current_rank = max(1, int(self.sliders[0].value))
        self.svd = SVDNoveltyDetector(compression_rank=self.current_rank)
        self.current_threshold = self.sliders[1].value / 100.0
        
    def build_display(self, original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
        """Build HD display with Option A layout: vertical frames + side control panel."""
        
        # === HD Layout Constants ===
        TOTAL_WIDTH = 1920
        TOTAL_HEIGHT = 1080
        CONTROL_PANEL_WIDTH = 400
        FRAME_AREA_WIDTH = TOTAL_WIDTH - CONTROL_PANEL_WIDTH
        FRAME_WIDTH = FRAME_AREA_WIDTH - 40  # Padding
        FRAME_HEIGHT = (TOTAL_HEIGHT - 60) // 2  # Two frames with gap
        PADDING = 20
        
        # === Create main canvas ===
        display = np.zeros((TOTAL_HEIGHT, TOTAL_WIDTH, 3), dtype=np.uint8)
        display[:] = COLOR_SECONDARY
        
        # === Process frames ===
        recon_display = np.clip(reconstructed, 0, 1)
        recon_display = (recon_display * 255).astype(np.uint8)
        original_display = (original * 255).astype(np.uint8)
        
        # Resize frames to fit layout
        original_resized = cv2.resize(original_display, (FRAME_WIDTH, FRAME_HEIGHT))
        recon_resized = cv2.resize(recon_display, (FRAME_WIDTH, FRAME_HEIGHT))
        
        # Convert to BGR
        original_bgr = cv2.cvtColor(original_resized, cv2.COLOR_GRAY2BGR)
        recon_bgr = cv2.cvtColor(recon_resized, cv2.COLOR_GRAY2BGR)
        
        # === Draw frame containers ===
        frame_x = PADDING
        frame1_y = PADDING
        frame2_y = PADDING + FRAME_HEIGHT + 20
        
        # Frame 1: Original
        draw_rounded_rect(display, (frame_x - 5, frame1_y - 5), 
                         (frame_x + FRAME_WIDTH + 5, frame1_y + FRAME_HEIGHT + 5), 
                         COLOR_PRIMARY, radius=10)
        display[frame1_y:frame1_y + FRAME_HEIGHT, frame_x:frame_x + FRAME_WIDTH] = original_bgr
        cv2.putText(display, "ORIGINAL", (frame_x + 15, frame1_y + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_SECONDARY, 2)
        
        # Frame 2: Reconstructed
        draw_rounded_rect(display, (frame_x - 5, frame2_y - 5), 
                         (frame_x + FRAME_WIDTH + 5, frame2_y + FRAME_HEIGHT + 5), 
                         COLOR_PRIMARY, radius=10)
        display[frame2_y:frame2_y + FRAME_HEIGHT, frame_x:frame_x + FRAME_WIDTH] = recon_bgr
        cv2.putText(display, "SVD RECONSTRUCTED", (frame_x + 15, frame2_y + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_SECONDARY, 2)
        cv2.putText(display, f"Rank: {self.current_rank}", (frame_x + 15, frame2_y + 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_ACCENT, 1)
        
        # === Control Panel (right side) ===
        panel_x = FRAME_AREA_WIDTH
        panel_y = PADDING
        panel_width = CONTROL_PANEL_WIDTH - PADDING
        panel_height = TOTAL_HEIGHT - 2 * PADDING
        
        # Panel background
        draw_rounded_rect(display, (panel_x, panel_y), 
                         (panel_x + panel_width, panel_y + panel_height), 
                         COLOR_PRIMARY, radius=15)
        
        # === Status Section ===
        status_y = panel_y + 20
        status_height = 80
        draw_rounded_rect(display, (panel_x + 15, status_y), 
                         (panel_x + panel_width - 15, status_y + status_height), 
                         COLOR_SECONDARY, radius=10)
        
        # Status text
        if self.show_alert:
            status_text = "NOVELTY DETECTED!"
            status_color = COLOR_ALERT
            draw_rounded_rect(display, (panel_x + 15, status_y), 
                             (panel_x + panel_width - 15, status_y + status_height), 
                             COLOR_ALERT, radius=10, thickness=3)
        else:
            status_text = "NORMAL"
            status_color = (100, 220, 100)  # Green
        
        cv2.putText(display, "STATUS", (panel_x + 30, status_y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PRIMARY, 2)
        cv2.putText(display, status_text, (panel_x + 30, status_y + 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # === Metrics Section ===
        metrics_y = status_y + status_height + 20
        metrics_height = 120
        draw_rounded_rect(display, (panel_x + 15, metrics_y), 
                         (panel_x + panel_width - 15, metrics_y + metrics_height), 
                         COLOR_SECONDARY, radius=10)
        
        cv2.putText(display, "METRICS", (panel_x + 30, metrics_y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PRIMARY, 2)
        cv2.putText(display, f"Error: {self.last_error:.4f}", (panel_x + 30, metrics_y + 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT_DIM, 2)
        cv2.putText(display, f"Threshold: {self.current_threshold:.4f}", (panel_x + 30, metrics_y + 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT_DIM, 2)
        cv2.putText(display, f"FPS: {self.fps:.1f}", (panel_x + 30, metrics_y + 105), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT_DIM, 2)
        
        # === Rank Slider Section ===
        rank_y = metrics_y + metrics_height + 20
        rank_height = 80
        draw_rounded_rect(display, (panel_x + 15, rank_y), 
                         (panel_x + panel_width - 15, rank_y + rank_height), 
                         COLOR_SECONDARY, radius=10)
        
        # Position rank slider
        self.sliders[0].x = panel_x + 30
        self.sliders[0].y = rank_y + 45
        self.sliders[0].width = panel_width - 60
        self.sliders[0].label = "COMPRESSION RANK"
        self.sliders[0].draw(display)
        
        # === Threshold Slider Section ===
        threshold_y = rank_y + rank_height + 20
        threshold_height = 80
        draw_rounded_rect(display, (panel_x + 15, threshold_y), 
                         (panel_x + panel_width - 15, threshold_y + threshold_height), 
                         COLOR_SECONDARY, radius=10)
        
        # Position threshold slider
        self.sliders[1].x = panel_x + 30
        self.sliders[1].y = threshold_y + 45
        self.sliders[1].width = panel_width - 60
        self.sliders[1].label = "THRESHOLD"
        self.sliders[1].draw(display)
        
        # === Detected Objects Section ===
        objects_y = threshold_y + threshold_height + 20
        objects_height = panel_height - (objects_y - panel_y) - 20
        draw_rounded_rect(display, (panel_x + 15, objects_y), 
                         (panel_x + panel_width - 15, objects_y + objects_height), 
                         COLOR_SECONDARY, radius=10)
        
        cv2.putText(display, "DETECTED OBJECTS", (panel_x + 30, objects_y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PRIMARY, 2)
        
        if self.detected_objects:
            y_offset = 55
            for i, obj in enumerate(self.detected_objects[:8]):  # Show up to 8 objects
                text = f"- {obj['object']} ({obj['confidence']:.2f})"
                cv2.putText(display, text, (panel_x + 30, objects_y + y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT_DIM, 2)
                y_offset += 25
                if y_offset > objects_height - 10:
                    break
        else:
            cv2.putText(display, "No objects detected", (panel_x + 30, objects_y + 55), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT_DIM, 2)
        
        # === Reconnect Button (only shows when video disconnected) ===
        if not self.video_connected:
            btn_w = 120
            btn_h = 40
            btn_x = panel_x + panel_width - btn_w - 15
            btn_y = objects_y + objects_height - btn_h - 10
            
            # Store button position for click detection
            self.reconnect_btn = {'x': btn_x, 'y': btn_y, 'w': btn_w, 'h': btn_h}
            
            # Draw button
            draw_rounded_rect(display, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), 
                             COLOR_PRIMARY, radius=8)
            
            # Center text in button
            text = "RECONNECT"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = btn_x + (btn_w - text_size[0]) // 2
            text_y = btn_y + (btn_h + text_size[1]) // 2
            cv2.putText(display, text, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_SECONDARY, 2)
        
        # === Update base dimensions ===
        self.base_display_width = TOTAL_WIDTH
        self.base_display_height = TOTAL_HEIGHT
        
        return display
    
    def run(self):
        """Main application loop."""
        print(f"Connecting to video feed: {self.video_url}")
        cap = cv2.VideoCapture(self.video_url)
        
        self.video_connected = cap.isOpened()
        if not self.video_connected:
            print(f"WARNING: Could not open video feed at {self.video_url}")
            print("Running with blank display - click RECONNECT button to retry")
        else:
            print("Video feed connected!")
        
        print("Controls: Drag sliders to adjust | ESC to exit | F to toggle fullscreen")
        
        # Create window - using WINDOW_NORMAL for proper close button detection
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Track fullscreen state
        self.fullscreen = False
        
        # Initialize sliders
        self.sliders = [
            CustomSlider(10, 60, 350, 1, MAX_RANK, INITIAL_RANK, "Compression Rank"),
            CustomSlider(370, 60, 350, 1, MAX_THRESHOLD, INITIAL_THRESHOLD, "Threshold (x100)")
        ]
        
        # Default blank frame for when video is not available
        blank_frame = np.zeros((240, 320), dtype=np.float32)
        blank_reconstructed = np.zeros((240, 320), dtype=np.float32)
        
        try:
            while True:
                # Handle reconnect button click
                if self.reconnect_requested:
                    self.reconnect_requested = False
                    print("Attempting to reconnect...")
                    cap.release()
                    cap = cv2.VideoCapture(self.video_url)
                    self.video_connected = cap.isOpened()
                    if self.video_connected:
                        print("Reconnected successfully!")
                    else:
                        print("Reconnection failed. Try again.")
                
                # Try to read frame if connected
                frame = None
                if self.video_connected or cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        self.video_connected = False
                        frame = None
                
                # Use actual frame or blank
                if frame is not None:
                    # Process frame
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, (320, 240))
                    gray_normalized = gray.astype(np.float32) / 255.0
                    
                    # SVD reconstruction
                    error, reconstructed = self.svd.compare_frames(gray_normalized)
                    self.last_error = error
                    
                    # Novelty detection
                    if self.reference_error is None:
                        self.reference_error = error
                        self.is_novel = True
                    else:
                        self.is_novel = abs(error - self.reference_error) > self.current_threshold
                        if self.is_novel:
                            self.reference_error = error
                            self.show_alert = True
                            self.alert_start_time = time.time()
                    
                    # Alert persistence
                    if self.show_alert and (time.time() - self.alert_start_time) > self.alert_duration:
                        self.show_alert = False
                    
                    # AI detection
                    if self.is_novel:
                        frame_resized = cv2.resize(frame, (320, 240))
                        frame_normalized = frame_resized.astype(np.float32) / 255.0
                        self.detected_objects = self.ai.run_detection_ai(frame_normalized)
                else:
                    # No video - use blank frames
                    gray_normalized = blank_frame
                    reconstructed = blank_reconstructed
                    self.last_error = 0.0
                
                # FPS calculation
                self.frame_count += 1
                if time.time() - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count / (time.time() - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = time.time()
                
                # Build display at base size
                display = self.build_display(gray_normalized, reconstructed)
                base_h, base_w = display.shape[:2]
                
                # Get current window size and resize display to match
                try:
                    rect = cv2.getWindowImageRect(self.window_name)
                    if rect is not None:
                        _, _, win_w, win_h = rect
                        if win_w > 100 and win_h > 100:
                            self.current_window_width = win_w
                            self.current_window_height = win_h
                            # Resize display to window size (makes mouse coords 1:1)
                            display = cv2.resize(display, (win_w, win_h))
                            # Update scale factors for slider positions
                            self.scale_x = win_w / base_w
                            self.scale_y = win_h / base_h
                except Exception:
                    pass
                
                cv2.imshow(self.window_name, display)
                
                # Key handling - waitKey must be called to process window events
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('f') or key == ord('F'):
                    # Toggle fullscreen
                    self.fullscreen = not self.fullscreen
                    if self.fullscreen:
                        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                
                # Check if window was closed (X button) - must be AFTER waitKey
                # When window is closed, getWindowProperty returns -1
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Closed.")


def main():
    print("=" * 50)
    print("  SVD Gatekeeping Prototype")
    print("  Modern UI Edition")
    print("=" * 50)
    
    app = GatekeeperApp(VIDEO_URL)
    app.run()


if __name__ == "__main__":
    main()
