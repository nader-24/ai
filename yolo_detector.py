import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

class YOLODetector:
    def __init__(self, model_path, class_names=None, conf_thresh=0.5):
        self.interpreter = Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input shape
        self.input_shape = self.input_details[0]['shape'][1:3]
        self.conf_thresh = conf_thresh
        
        # Load class names
        self.class_names = class_names or self._load_default_names()
    
    def _load_default_names(self):
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def preprocess(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, self.input_shape)
        return np.expand_dims(img_resized, axis=0).astype(np.uint8)
    
    def detect(self, image):
        input_tensor = self.preprocess(image)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        
        # Get outputs (adjust indices based on your model)
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        
        detections = []
        for i in range(len(scores)):
            if scores[i] > self.conf_thresh:
                class_id = int(classes[i])
                detections.append({
                    'class_id': class_id,
                    'class_name': self.class_names[class_id],
                    'confidence': float(scores[i]),
                    'box': boxes[i].tolist()
                })
        return detections
    
    def draw_detections(self, image, detections):
        h, w = image.shape[:2]
        for det in detections:
            box = det['box']
            x1, y1, x2, y2 = int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            label = f"{det['class_name']} {det['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - 20), (x1 + tw, y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(image, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return image