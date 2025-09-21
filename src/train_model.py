from ultralytics import YOLO
import yaml
import os
from datetime import datetime

class KidneyStoneTrainer:
    def __init__(self, data_config='data/dataset/data.yaml'):
        self.data_config = data_config
        # Load pre-trained YOLOv8n model (will download automatically if not present)
        self.model = YOLO('yolov8n.pt')  # Pre-trained on COCO dataset
        
    def setup_transfer_learning(self):
        """Configure model for transfer learning from COCO to kidney stones"""
        print("Loading YOLOv8n model pre-trained on COCO dataset...")
        print(f"Model loaded: {self.model.model}")
        
        # The model will automatically adapt the final layer for our single class
        # during training when it reads the data.yaml file
        
    def train_with_transfer_learning(self, epochs=20, img_size=640, batch_size=16):
        """Train using transfer learning from COCO pre-trained weights"""
        
        print("Starting transfer learning training...")
        print(f"Base model: YOLOv8n (COCO pre-trained)")
        print(f"Target: Kidney stone detection (1 class)")
        
        # Training configuration optimized for transfer learning
        training_config = {
            'data': self.data_config,
            'epochs': epochs,
            'imgsz': img_size,
            'batch': batch_size,
            'name': f'kidney_stone_transfer_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'project': 'models',
            'exist_ok': True,
            
            # Transfer learning specific settings
            'patience': 30,        # Early stopping patience
            'save_period': 5,      # Save checkpoint every 5 epochs
            'workers': 8,
            'device': 'cpu',       # Use CPU (no GPU available)
            
            # Optimizer settings for transfer learning
            'optimizer': 'AdamW',
            'lr0': 0.001,          # Lower initial learning rate for fine-tuning
            'lrf': 0.01,           # Final learning rate
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            
            # Loss function weights
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            
            # Data augmentation (reduce for medical images)
            'hsv_h': 0.015,        # Hue augmentation
            'hsv_s': 0.7,          # Saturation augmentation  
            'hsv_v': 0.4,          # Value augmentation
            'degrees': 10,         # Rotation degrees
            'translate': 0.1,      # Translation
            'scale': 0.2,          # Scale variation
            'shear': 2.0,          # Shear
            'perspective': 0.0,    # Perspective (disable for medical images)
            'flipud': 0.0,         # Vertical flip (disable for medical images)
            'fliplr': 0.5,         # Horizontal flip
            'mosaic': 1.0,         # Mosaic augmentation
            'mixup': 0.0,          # Mixup (disable for medical data)
            
            # Other settings
            'cache': False,        # Don't cache images (medical images can be large)
            'rect': False,         # Rectangular training
            'cos_lr': True,        # Cosine learning rate scheduler
            'close_mosaic': 10,    # Close mosaic augmentation in last epochs
            'resume': False,
            'amp': True,           # Automatic Mixed Precision
            'fraction': 1.0,       # Dataset fraction to use
            'profile': False,
            'freeze': None         # Don't freeze any layers (fine-tune all)
        }
        
        # Start training
        print("Training configuration:")
        for key, value in training_config.items():
            print(f"  {key}: {value}")
        
        results = self.model.train(**training_config)
        
        # Save the best model with a specific name
        best_model_path = "models/best_kidney_stone_yolov8n.pt"
        
        # The trained model is automatically saved, copy it to our preferred location
        import shutil
        training_dir = f"models/{training_config['name']}"
        if os.path.exists(f"{training_dir}/weights/best.pt"):
            shutil.copy(f"{training_dir}/weights/best.pt", best_model_path)
            print(f"Best model copied to: {best_model_path}")
        
        return results, best_model_path
    
    def validate_model(self):
        """Validate the trained model"""
        print("Validating trained model...")
        metrics = self.model.val()
        
        # Extract and display key metrics
        if hasattr(metrics, 'box'):
            print(f"mAP@0.5: {metrics.box.map50:.4f}")
            print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
            print(f"Precision: {metrics.box.mp:.4f}")
            print(f"Recall: {metrics.box.mr:.4f}")
        
        return metrics
    
    def export_model(self, format='onnx'):
        """Export model for deployment"""
        print(f"Exporting model to {format} format...")
        self.model.export(format=format)
        print("Model export completed!")

# Training execution script
if __name__ == "__main__":
    # Validate data first
    from data_preprocessing import DataValidator
    
    validator = DataValidator()
    if not validator.validate_dataset_structure():
        print("Please fix dataset structure before training!")
        exit(1)
    
    validator.count_files()
    
    # Start training
    trainer = KidneyStoneTrainer()
    trainer.setup_transfer_learning()
    
    # Train with transfer learning (optimized epochs for kidney stone dataset)
    results, model_path = trainer.train_with_transfer_learning(
        epochs=20,        # Optimal epochs for 1,054 training images
        batch_size=16,    # Adjust based on your GPU memory
        img_size=640
    )
    
    # Validate the trained model
    metrics = trainer.validate_model()
    
    print(f"\nTraining completed successfully!")
    print(f"Best model saved at: {model_path}")
    print("You can now use this model for detection in your Streamlit app!")
