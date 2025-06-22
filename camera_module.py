import cv2
import time

class Camera:
    def __init__(self, camera_index=0, width=640, height=480):
        self.width = width
        self.height = height
        self.cap = self._init_camera(camera_index)
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps = 0
        
    def _init_camera(self, index):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            for i in range(1, 4):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    break
            if not cap.isOpened():
                raise RuntimeError("Camera not found! Check connections and drivers")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        time.sleep(2.0)  # Camera warm-up
        return cap
    
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Calculate FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_frame_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_frame_time)
            self.frame_count = 0
            self.last_frame_time = current_time
            
        return frame
    
    def release(self):
        self.cap.release()