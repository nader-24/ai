import cv2
import time
import logging
import csv
import os
from datetime import datetime
from camera_module import Camera
from yolo_detector import YOLODetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = "yolov8n_int8.tflite"
CAMERA_INDEX = 0
WIDTH = 640
HEIGHT = 480
CONF_THRESH = 0.5
TARGET_FPS = 10
LOG_FILE = "detections.csv"
LOG_INTERVAL = 1.0  # Seconds between log writes

class DetectionLogger:
    def __init__(self, filename):
        self.filename = filename
        self.detections = []
        self.last_write_time = time.monotonic()
        self._ensure_directory()
        self._initialize_file()
        
    def _ensure_directory(self):
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        
    def _initialize_file(self):
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2'])
    
    def add_detection(self, detection):
        self.detections.append(detection)
        
    def write_to_file(self):
        if not self.detections:
            return
            
        try:
            with open(self.filename, 'a', newline='') as f:
                writer = csv.writer(f)
                for det in self.detections:
                    writer.writerow([
                        det['timestamp'],
                        det['class_name'],
                        f"{det['confidence']:.4f}",
                        det['x1'],
                        det['y1'],
                        det['x2'],
                        det['y2']
                    ])
            logger.info(f"Logged {len(self.detections)} detections to {self.filename}")
            self.detections = []
        except Exception as e:
            logger.error(f"Error writing to log file: {str(e)}")

def main():
    # Initialize components
    logger.info("Initializing camera...")
    camera = Camera(camera_index=CAMERA_INDEX, width=WIDTH, height=HEIGHT)
    
    logger.info("Loading YOLO model...")
    detector = YOLODetector(model_path=MODEL_PATH, conf_thresh=CONF_THRESH)
    
    logger.info("Initializing detection logger...")
    detection_logger = DetectionLogger(LOG_FILE)
    
    # Warm-up camera and model
    logger.info("Warming up camera and model...")
    warm_up_frames = 10
    for i in range(warm_up_frames):
        frame = camera.get_frame()
        if frame is not None and i == warm_up_frames - 1:
            _ = detector.detect(frame)
    
    logger.info("Starting detection loop. Press 'q' to exit...")
    
    # For FPS calculation
    detection_times = []
    last_log_time = time.monotonic()
    
    try:
        while True:
            start_time = time.monotonic()
            
            # Capture frame
            frame = camera.get_frame()
            if frame is None:
                logger.warning("No frame captured! Retrying...")
                time.sleep(0.1)
                continue
                
            # Run detection
            detections = detector.detect(frame)
            
            # Process and log detections
            if detections:
                current_time = datetime.now().isoformat()
                frame_height, frame_width = frame.shape[:2]
                
                for det in detections:
                    # Convert normalized coordinates to pixel values
                    box = det['box']
                    x1 = int(box[0] * frame_width)
                    y1 = int(box[1] * frame_height)
                    x2 = int(box[2] * frame_width)
                    y2 = int(box[3] * frame_height)
                    
                    # Add to logger
                    detection_logger.add_detection({
                        'timestamp': current_time,
                        'class_name': det['class_name'],
                        'confidence': det['confidence'],
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2
                    })
            
            # Draw results
            processed_frame = detector.draw_detections(frame, detections)
            
            # Calculate and display FPS
            detection_time = time.monotonic() - start_time
            detection_times.append(detection_time)
            if len(detection_times) > 10:
                detection_times.pop(0)
                
            avg_detection_time = sum(detection_times) / len(detection_times)
            detection_fps = 1.0 / avg_detection_time if avg_detection_time > 0 else 0
            
            # Display camera FPS
            cv2.putText(processed_frame, f"Cam FPS: {camera.fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display detection FPS
            cv2.putText(processed_frame, f"Det FPS: {detection_fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display detections count
            cv2.putText(processed_frame, f"Detections: {len(detections)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Print detections to console
            if detections:
                detection_log = f"Detections: "
                for i, det in enumerate(detections):
                    if i > 0:
                        detection_log += ", "
                    detection_log += f"{det['class_name']}({det['confidence']:.2f})"
                logger.info(detection_log)
            
            # Display output
            cv2.imshow('YOLO Object Detection', processed_frame)
            
            # Write to log file periodically
            current_time = time.monotonic()
            if current_time - last_log_time >= LOG_INTERVAL:
                detection_logger.write_to_file()
                last_log_time = current_time
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Maintain target FPS
            elapsed = time.monotonic() - start_time
            if elapsed < 1/TARGET_FPS:
                time.sleep(1/TARGET_FPS - elapsed)
                
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
    finally:
        # Write any remaining detections
        detection_logger.write_to_file()
        
        logger.info("Releasing resources...")
        camera.release()
        cv2.destroyAllWindows()
        logger.info("Application exited cleanly")

if __name__ == "__main__":
    main()