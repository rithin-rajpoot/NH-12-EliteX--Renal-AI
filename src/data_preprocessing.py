import cv2
import numpy as np
import os
from pathlib import Path

class DataValidator:
    def __init__(self, dataset_root="data/dataset"):
        self.dataset_root = dataset_root
        
    def validate_dataset_structure(self):
        """Validate the dataset structure matches expected format"""
        required_dirs = [
            "train/images", "train/labels",
            "test/images", "test/labels", 
            "valid/images", "valid/labels"
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            full_path = os.path.join(self.dataset_root, dir_path)
            if not os.path.exists(full_path):
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            print(f"Missing directories: {missing_dirs}")
            return False
        
        print("Dataset structure validation passed!")
        return True
    
    def count_files(self):
        """Count images and labels in each split"""
        splits = ["train", "test", "valid"]
        
        for split in splits:
            images_dir = os.path.join(self.dataset_root, split, "images")
            labels_dir = os.path.join(self.dataset_root, split, "labels")
            
            if os.path.exists(images_dir) and os.path.exists(labels_dir):
                image_count = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                label_count = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
                
                print(f"{split.capitalize()}: {image_count} images, {label_count} labels")
                
                if image_count != label_count:
                    print(f"WARNING: Mismatch in {split} - {image_count} images vs {label_count} labels")
    
    def validate_yolo_annotations(self, split="train"):
        """Validate YOLO format annotations"""
        labels_dir = os.path.join(self.dataset_root, split, "labels")
        
        valid_files = 0
        invalid_files = []
        
        for label_file in Path(labels_dir).glob("*.txt"):
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    
                for line_num, line in enumerate(lines, 1):
                    if line.strip():  # Skip empty lines
                        parts = line.strip().split()
                        if len(parts) != 5:
                            invalid_files.append(f"{label_file}:line{line_num} - Expected 5 values, got {len(parts)}")
                            continue
                            
                        try:
                            class_id, x_center, y_center, width, height = map(float, parts)
                            
                            # Validate class_id (should be 0 for single class)
                            if class_id != 0:
                                invalid_files.append(f"{label_file}:line{line_num} - Invalid class_id: {class_id}")
                                
                            # Validate normalized coordinates (0-1)
                            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
                                invalid_files.append(f"{label_file}:line{line_num} - Invalid coordinates")
                                
                        except ValueError as e:
                            invalid_files.append(f"{label_file}:line{line_num} - Cannot convert to float: {e}")
                
                if not invalid_files or label_file.name not in str(invalid_files):
                    valid_files += 1
                    
            except Exception as e:
                invalid_files.append(f"{label_file} - Error reading file: {e}")
        
        print(f"Validation complete: {valid_files} valid files")
        if invalid_files:
            print("Invalid files found:")
            for error in invalid_files[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(invalid_files) > 10:
                print(f"  ... and {len(invalid_files) - 10} more errors")
        
        return len(invalid_files) == 0

# Usage
if __name__ == "__main__":
    validator = DataValidator()
    validator.validate_dataset_structure()
    validator.count_files()
    validator.validate_yolo_annotations("train")
