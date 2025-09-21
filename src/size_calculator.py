import numpy as np
import cv2
from typing import Dict, List, Tuple

class SizeCalculator:
    def __init__(self, pixel_to_mm_ratio=0.264):  # Default: 96 DPI
        self.pixel_to_mm_ratio = pixel_to_mm_ratio
        
    def calculate_stone_size(self, detection: Dict) -> Dict:
        """Calculate real-world size of detected kidney stone"""
        width_pixels = detection['width']
        height_pixels = detection['height']
        
        # Convert pixels to millimeters
        width_mm = width_pixels * self.pixel_to_mm_ratio
        height_mm = height_pixels * self.pixel_to_mm_ratio
        
        # Calculate area and equivalent diameter
        area_pixels = width_pixels * height_pixels
        area_mm2 = area_pixels * (self.pixel_to_mm_ratio ** 2)
        
        # Equivalent circular diameter
        equivalent_diameter = 2 * np.sqrt(area_mm2 / np.pi)
        
        size_info = {
            'width_mm': round(width_mm, 2),
            'height_mm': round(height_mm, 2),
            'area_mm2': round(area_mm2, 2),
            'equivalent_diameter_mm': round(equivalent_diameter, 2),
            'width_pixels': width_pixels,
            'height_pixels': height_pixels,
            'area_pixels': area_pixels
        }
        
        return size_info
    
    def determine_treatment_recommendation(self, size_mm: float) -> Dict:
        """Determine treatment recommendation based on stone size"""
        if size_mm <= 4:
            recommendation = {
                'category': 'Small Stone',
                'treatment': 'Conservative management with increased fluid intake',
                'urgency': 'Low',
                'pass_probability': 'High (>90%)'
            }
        elif size_mm <= 6:
            recommendation = {
                'category': 'Medium Stone',
                'treatment': 'Medical therapy, possible intervention if symptomatic',
                'urgency': 'Moderate',
                'pass_probability': 'Moderate (60-80%)'
            }
        elif size_mm <= 10:
            recommendation = {
                'category': 'Large Stone', 
                'treatment': 'Likely requires intervention (SWL, ureteroscopy)',
                'urgency': 'High',
                'pass_probability': 'Low (10-30%)'
            }
        else:
            recommendation = {
                'category': 'Very Large Stone',
                'treatment': 'Surgical intervention required',
                'urgency': 'Urgent',
                'pass_probability': 'Very Low (<10%)'
            }
        
        return recommendation
    
    def analyze_stone_location(self, detection: Dict, image_shape: Tuple) -> Dict:
        """Analyze the anatomical location of the stone"""
        center_x, center_y = detection['center']
        img_height, img_width = image_shape[:2]
        
        # Determine relative position
        relative_x = center_x / img_width
        relative_y = center_y / img_height
        
        # Anatomical location mapping (simplified)
        if relative_y < 0.3:
            anatomical_region = "Upper pole of kidney"
        elif relative_y < 0.7:
            anatomical_region = "Mid-kidney/Renal pelvis"
        else:
            anatomical_region = "Lower pole of kidney"
            
        if relative_x < 0.5:
            side = "Left"
        else:
            side = "Right"
        
        location_info = {
            'anatomical_region': anatomical_region,
            'side': side,
            'relative_position': {
                'x': round(relative_x, 3),
                'y': round(relative_y, 3)
            },
            'pixel_coordinates': {
                'x': center_x,
                'y': center_y
            }
        }
        
        return location_info
