import cv2
import numpy as np
from ultralytics import YOLO
import torch
from PIL import Image
import os

class StoneDetector:
    def __init__(self, model_path=None, confidence_threshold=0.5):
        # Try to load your trained model first, fallback to others
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"Loaded custom model: {model_path}")
        elif os.path.exists('models/best_kidney_stone_yolov8n.pt'):
            self.model = YOLO('models/best_kidney_stone_yolov8n.pt')
            print("Loaded trained kidney stone model")
        elif os.path.exists('models/quick_kidney_stone.pt'):
            self.model = YOLO('models/quick_kidney_stone.pt')
            print("Loaded quick training model")
        else:
            # Fallback to pre-trained COCO model (for testing)
            self.model = YOLO('yolov8n.pt')
            print("WARNING: Using COCO pre-trained model (not trained for kidney stones)")
        
        self.confidence_threshold = confidence_threshold
        
    def create_custom_annotated_image(self, image, detections):
        """Create custom annotated image with stone numbers as labels"""
        annotated = image.copy()
        
        for i, detection in enumerate(detections, 1):  # Start numbering from 1
            bbox = detection['bbox']
            
            # Draw bounding box
            cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Add label with stone number
            label = f"Stone {i}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(annotated, 
                         (bbox[0], bbox[1] - label_size[1] - 10),
                         (bbox[0] + label_size[0], bbox[1]),
                         (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(annotated, label,
                       (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return annotated
        
    def detect_stones(self, image_path):
        """Detect kidney stones in the image"""
        try:
            results = self.model(image_path, conf=self.confidence_threshold)
            
            detections = []
            annotated_image = None
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for i, box in enumerate(boxes):
                        # Extract bounding box coordinates - FIXED
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'center': [(int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2],
                            'width': int(x2 - x1),
                            'height': int(y2 - y1)
                        }
                        detections.append(detection)
                
                # Create custom annotated image with stone numbers
                if len(detections) > 0:
                    # Get original image from result
                    original_image = result.orig_img
                    annotated_image = self.create_custom_annotated_image(original_image, detections)
            
            return detections, annotated_image
            
        except Exception as e:
            print(f"Error during detection: {e}")
            return [], None
    
    def detect_from_array(self, image_array):
        """Detect stones from numpy array (for Streamlit uploaded images)"""
        try:
            results = self.model(image_array, conf=self.confidence_threshold)
            
            detections = []
            annotated_image = None
            
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # Extract bounding box coordinates - FIXED
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'center': [(int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2],
                            'width': int(x2 - x1),
                            'height': int(y2 - y1)
                        }
                        detections.append(detection)
                
                # Create custom annotated image with stone numbers
                if len(detections) > 0:
                    annotated_image = self.create_custom_annotated_image(image_array, detections)
            
            return detections, annotated_image
            
        except Exception as e:
            print(f"Error during detection from array: {e}")
            return [], None

    def get_model_info(self):
        """Get information about the loaded model"""
        try:
            model_info = {
                'model_type': 'YOLOv8n',
                'classes': self.model.names,
                'device': next(self.model.model.parameters()).device,
                'confidence_threshold': self.confidence_threshold
            }
            return model_info
        except:
            return {'error': 'Could not retrieve model info'}