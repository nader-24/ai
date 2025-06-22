import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

class YOLODetector:
    def __init__(self, model_path, class_names=None, conf_thresh=0.5, nms_thresh=0.5):
        self.interpreter = Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Get input shape
        self.input_shape = self.input_details[0]['shape'][1:3]  # [height, width]
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        # Load class names
        self.class_names = class_names or self._load_default_names()

        # Log model details
        logger.info(f"Input shape: {self.input_shape}")
        logger.info(f"Input dtype: {self.input_details[0]['dtype']}")
        logger.info(f"Output details: {self.output_details}")

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
        """Improved preprocessing with normalization and proper resizing"""
        # Convert to RGB and resize
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.input_shape[1], self.input_shape[0]))

        # Normalize to [0,1] and convert to float32
        img_normalized = img_resized.astype(np.float32) / 255.0

        # Add batch dimension
        input_tensor = np.expand_dims(img_normalized, axis=0)
        return input_tensor

    def detect(self, image):
        if image is None:
            return []

        input_tensor = self.preprocess(image)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()

        # Get outputs - YOLOv8 has a single output tensor
        output = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Process output (shape: [1, 84, 8400])
        predictions = np.squeeze(output).T  # Transpose to [8400, 84]

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_thresh, :]
        scores = scores[scores > self.conf_thresh]

        if len(scores) == 0:
            return []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Extract boxes from predictions (center x, center y, width, height)
        boxes = predictions[:, :4]

        # Convert boxes to (x1, y1, x2, y2)
        boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2)  # x1
        boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2)  # y1
        boxes[:, 2] = (boxes[:, 0] + boxes[:, 2])       # x2
        boxes[:, 3] = (boxes[:, 1] + boxes[:, 3])       # y2

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), self.conf_thresh, self.nms_thresh
        )

        if len(indices) == 0:
            return []

        indices = indices.flatten()
        detections = []
        for idx in indices:
            box = boxes[idx]
            class_id = class_ids[idx]
            confidence = scores[idx]

            detections.append({
                'class_id': int(class_id),
                'class_name': self.class_names[int(class_id)],
                'confidence': float(confidence),
                'box': box.tolist()  # [x1, y1, x2, y2] normalized
            })

        return detections

    def draw_detections(self, image, detections):
        if image is None:
            return None

        h, w = image.shape[:2]
        for det in detections:
            box = det['box']
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)

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