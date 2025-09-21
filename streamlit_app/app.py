import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.detect_stones import StoneDetector
from src.size_calculator import SizeCalculator
from src.report_generator import ReportGenerator
from datetime import datetime
import tempfile

def preprocess_medical_image(image):
    """
    Preprocess medical image by converting to RGB and cropping to 550x550 from center
    Returns RGB version for AI analysis only (no display of preprocessed image)
    
    Args:
        image: PIL Image object
        
    Returns:
        tuple: (rgb_for_ai, preprocessing_info)
    """
    # Get original dimensions
    original_width, original_height = image.size
    original_mode = image.mode
    
    # Convert to RGB if not already (for AI analysis)
    if image.mode != 'RGB':
        rgb_image = image.convert('RGB')
    else:
        rgb_image = image.copy()
    
    # Crop to 550x550 from center
    target_size = 550
    
    if original_width > target_size or original_height > target_size:
        # Calculate center crop coordinates
        left = max(0, (original_width - target_size) // 2)
        top = max(0, (original_height - target_size) // 2)
        right = min(original_width, left + target_size)
        bottom = min(original_height, top + target_size)
        
        # Crop the image
        cropped_image = rgb_image.crop((left, top, right, bottom))
        was_cropped = True
        crop_area = (left, top, right, bottom)
    else:
        # Image is smaller than 550x550, no cropping needed
        cropped_image = rgb_image
        was_cropped = False
        crop_area = (0, 0, original_width, original_height)
    
    # Create preprocessing info
    preprocessing_info = {
        'original_size': (original_width, original_height),
        'processed_size': cropped_image.size,
        'original_mode': original_mode,
        'was_cropped': was_cropped,
        'crop_area': crop_area,
        'target_size': target_size
    }
    
    return cropped_image, preprocessing_info

# Page configuration
st.set_page_config(
    page_title="RenalAI - AI Powered Kidney Stone Diagnostic System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.375rem;
        color: #155724;
    }
    .report-section {
        width="stretch";
        margin-top: 1rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class KidneyStoneApp:
    def __init__(self):
        self.detector = None
        self.size_calculator = SizeCalculator()
        self.report_generator = ReportGenerator()
        
    def initialize_detector(self, model_path, confidence_threshold):
        """Initialize the stone detector with selected model"""
        try:
            self.detector = StoneDetector(model_path, confidence_threshold)
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def main(self):
        # Initialize session state first
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False
        if 'detections' not in st.session_state:
            st.session_state.detections = None
        if 'analyses' not in st.session_state:
            st.session_state.analyses = None
        if 'patient_info' not in st.session_state:
            st.session_state.patient_info = None
        if 'annotated_image' not in st.session_state:
            st.session_state.annotated_image = None
        if 'uploaded_image' not in st.session_state:
            st.session_state.uploaded_image = None
        if 'preprocessing_info' not in st.session_state:
            st.session_state.preprocessing_info = None
        if 'current_file_info' not in st.session_state:
            st.session_state.current_file_info = None
        if 'current_confidence' not in st.session_state:
            st.session_state.current_confidence = 0.25
            
        st.markdown('<h1 class="main-header">🏥 RenalAI - AI Powered Kidney Stone Diagnostic System</h1>', unsafe_allow_html=True)
        
        # Sidebar configuration
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Get the best model (no selection needed)
            available_models = self.get_available_models()
            model_path = available_models[0]  # Use the first (and only) model
            
            # Display which model is being used
            st.info(f"🤖 **Using Model:** {model_path.split('/')[-1]}")
            
            st.markdown("---")
            
            # Confidence threshold adjustment
            st.subheader("🎯 Detection Settings")
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=0.9,
                value=st.session_state.current_confidence,
                step=0.05,
                help="Higher values = more confident detections but may miss some stones. Lower values = more detections but may include false positives."
            )
            
            # Display confidence info
            if confidence_threshold <= 0.3:
                st.info("🔍 **High Sensitivity**: May detect more stones but with some false positives")
            elif confidence_threshold <= 0.6:
                st.success("⚖️ **Balanced**: Good balance between detection and accuracy")
            else:
                st.warning("🎯 **High Precision**: Very confident detections only, may miss smaller stones")
            
            st.markdown("---")
            st.subheader("📊 Model Info")
        
        # Initialize detector only if confidence has changed or first time
        if (self.detector is None or 
            confidence_threshold != st.session_state.current_confidence):
            
            if not self.initialize_detector(model_path, confidence_threshold):
                st.error("Failed to initialize detector. Please check your model files.")
                return
            
            # Update stored confidence and clear previous results if confidence changed
            if confidence_threshold != st.session_state.current_confidence:
                st.session_state.current_confidence = confidence_threshold
                # Clear previous analysis results when confidence changes
                st.session_state.detections = None
                st.session_state.analyses = None
                st.session_state.analysis_complete = False
            
        # Display model information
        with st.sidebar:
            try:
                model_info = self.detector.get_model_info()
                if 'error' not in model_info:
                    st.write(f"**Model Type:** {model_info.get('model_type', 'Unknown')}")
                    st.write(f"**Classes:** {list(model_info.get('classes', {}).values())}")
                    st.write(f"**Device:** {model_info.get('device', 'Unknown')}")
                    st.write(f"**Current Confidence:** {confidence_threshold:.2f}")
            except:
                st.write("Model info not available")
            
        # Main interface
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("📸 Image Upload & Patient Info")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Choose a kidney scan image",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                help="Upload CT scan, X-ray, ultrasound, or other medical image"
            )
            
            if uploaded_file is not None:
                # Check if this is a new file (different from previously uploaded)
                current_file_info = {
                    'name': uploaded_file.name,
                    'size': uploaded_file.size,
                    'type': uploaded_file.type
                }
                
                # Compare with previously uploaded file
                previous_file_info = st.session_state.get('current_file_info', None)
                is_new_file = (previous_file_info is None or 
                              current_file_info != previous_file_info)
                
                if is_new_file:
                    # Clear all analysis results when new file is uploaded
                    st.session_state.analysis_complete = False
                    st.session_state.detections = None
                    st.session_state.analyses = None
                    st.session_state.annotated_image = None
                    st.session_state.preprocessing_info = None
                    st.session_state.patient_info = None  # Also clear patient info for new file
                    st.session_state.current_file_info = current_file_info
                    
                    # Show notification that results were cleared
                    if previous_file_info is not None:
                        st.info("🔄 New file detected. Previous analysis results have been cleared.")
                
                # Load and preprocess the uploaded image
                original_image = Image.open(uploaded_file)
                rgb_for_ai, preprocessing_info = preprocess_medical_image(original_image)
                
                # Display original image
                st.subheader("📷 Medical Image")
                st.image(original_image, caption="Medical Image for Analysis", width=400)
                
                # Image information
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.write(f"**Original Size:** {preprocessing_info['original_size'][0]} x {preprocessing_info['original_size'][1]} pixels")
                    st.write(f"**Image Mode:** {preprocessing_info['original_mode']}")
                
                with col_info2:
                    st.write(f"**AI Processing:** {preprocessing_info['processed_size'][0]} x {preprocessing_info['processed_size'][1]} pixels")
                    if preprocessing_info['was_cropped']:
                        st.write(f"**Status:** Center-cropped to {preprocessing_info['target_size']}x{preprocessing_info['target_size']}")
                    else:
                        st.write(f"**Status:** No cropping needed")
                
                # Show preprocessing information
                if preprocessing_info['was_cropped']:
                    st.info("🔧 **AI Preprocessing:** Image will be center-cropped to 550x550 for optimal AI analysis")
                else:
                    st.success("✅ Image size optimal for AI analysis!")
                
                # Patient information form
                st.subheader("👤 Patient Information")
                col1a, col1b = st.columns(2)
                
                with col1a:
                    patient_name = st.text_input("Patient Name", value="anonymous", placeholder="Enter patient name")
                    patient_age = st.number_input("Age", min_value=1, max_value=120, value=45)
                
                with col1b:
                    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                    examination_date = st.date_input("Examination Date", datetime.now().date())
                
                patient_id = st.text_input("Patient ID", placeholder="Optional patient identifier")
                
                # Show notification if confidence changed and image was previously analyzed
                if (st.session_state.get('analysis_complete', False) and 
                    'detections' not in st.session_state):
                    st.warning("⚠️ Confidence threshold changed. Please re-analyze the image to see updated results.")
                
                # Analysis button
                if st.button("🔍 Analyze Image", type="primary", width="stretch"):
                    if not patient_name.strip():
                        st.warning("Please enter patient name before analysis.")
                    else:
                        # Store data in session state (use RGB image for AI analysis)
                        st.session_state.uploaded_image = rgb_for_ai
                        st.session_state.preprocessing_info = preprocessing_info
                        st.session_state.patient_info = {
                            'name': patient_name,
                            'age': patient_age,
                            'gender': patient_gender,
                            'examination_date': examination_date,
                            'patient_id': patient_id
                        }
                        # Perform analysis and update session state (using RGB for AI)
                        self.analyze_image(rgb_for_ai)
            else:
                # No file uploaded - clear all analysis results and file info
                if st.session_state.get('current_file_info') is not None:
                    # Only clear if there was previously a file
                    st.session_state.analysis_complete = False
                    st.session_state.detections = None
                    st.session_state.analyses = None
                    st.session_state.annotated_image = None
                    st.session_state.preprocessing_info = None
                    st.session_state.patient_info = None
                    st.session_state.uploaded_image = None
                    st.session_state.current_file_info = None
                
                # Show upload instruction when no file is selected
                st.info("👆 Please upload a kidney scan image to begin analysis.")
        
        with col2:
            # Display results section header
            st.header("🔬 Analysis Results")
            
            # Display analysis results if available
            if st.session_state.get('analysis_complete', False) and st.session_state.get('detections') is not None:
                self.display_analysis_results()
            else:
                st.info("Upload an image and analyze it to see results here.")
        
        # Display report generation section separately - full width
        self.display_report_generation_section()
    
    def get_available_models(self):
        """Get the best trained model"""
        # Use only the best performing model
        best_model = "models/kidney_stone_transfer_20250920_192030/weights/epoch10.pt"
        
        if os.path.exists(best_model):
            return [best_model]
        else:
            # Fallback to pre-trained COCO model if best model not found
            st.warning("Trained kidney stone model not found. Using pre-trained COCO model.")
            return ["yolov8n.pt"]
    
    def analyze_image(self, image):
        """Analyze the uploaded image for kidney stones - only stores results, doesn't display"""
        with st.spinner("🤖 AI is analyzing the image..."):
            # Convert PIL image to numpy array
            image_array = np.array(image)
            
            # Detect stones
            detections, annotated_image = self.detector.detect_from_array(image_array)
            
            # Store results in session state
            st.session_state.detections = detections
            st.session_state.annotated_image = annotated_image
            
            if detections:
                # Analyze each detection
                all_analyses = []
                for i, detection in enumerate(detections):
                    analysis = self.analyze_single_stone_data(detection, image_array.shape, i+1)
                    all_analyses.append(analysis)
                
                # Store analyses in session state
                st.session_state.analyses = all_analyses
                st.session_state.analysis_complete = True
            else:
                # No detections found
                st.session_state.analyses = []
                st.session_state.analysis_complete = True
    
    def display_analysis_results(self):
        """Display the analysis results stored in session state"""
        detections = st.session_state.detections
        annotated_image = st.session_state.annotated_image
        analyses = st.session_state.analyses
        
        if detections:
            st.markdown(f'<div class="success-box">✅ <strong>Detected {len(detections)} kidney stone(s)</strong></div>', 
                      unsafe_allow_html=True)
            
            # Display annotated image
            if annotated_image is not None:
                st.image(annotated_image, caption="🎯 Detection Results (Annotated)", width=400)
            
            # Create tabs for detailed analysis
            if analyses:
                tabs = st.tabs([f"Stone #{i+1}" for i in range(len(detections))] + ["📊 Summary"])
                
                # Display each stone analysis
                for i, analysis in enumerate(analyses):
                    with tabs[i]:
                        self.display_stone_analysis(analysis, i+1)
                
                # Summary tab
                with tabs[-1]:
                    self.display_summary(analyses)
        else:
            st.info("ℹ️ No kidney stones detected in the image.")
            st.write("This could mean:")
            st.write("- No stones are present")
            st.write("- Stones are too small to detect")  
            st.write("- Image quality needs improvement")
            st.write("- Confidence threshold might be too high")
            
            # Suggest adjustments
            st.write("**Suggestions:**")
            st.write("- Try lowering the confidence threshold")
            st.write("- Ensure good image quality and contrast")
            st.write("- Verify the image shows the kidney region clearly")
    
    def display_report_generation_section(self):
        """Display the report generation section separately"""
        if st.session_state.get('analysis_complete', False) and st.session_state.get('detections') is not None:
            # Create a separate section for report generation
            st.markdown("---")
            st.header("📄 Medical Report Generation")
            
            # Center the generate report button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("📄 Generate Comprehensive Medical Report", type="secondary", width="stretch"):
                    patient_info = st.session_state.patient_info
                    detections = st.session_state.detections
                    analyses = st.session_state.analyses
                    self.generate_and_display_report(detections, analyses, patient_info)
    
    def analyze_single_stone_data(self, detection, image_shape, stone_number):
        """Analyze a single detected stone and return data (no display)"""
        
        # Calculate size
        size_info = self.size_calculator.calculate_stone_size(detection)
        
        # Get location
        location_info = self.size_calculator.analyze_stone_location(detection, image_shape)
        
        # Get treatment recommendation
        treatment_rec = self.size_calculator.determine_treatment_recommendation(
            size_info['equivalent_diameter_mm']
        )
        
        return {
            'detection': detection,
            'size_info': size_info,
            'location_info': location_info, 
            'treatment_rec': treatment_rec
        }
    
    def display_stone_analysis(self, analysis, stone_number):
        """Display detailed analysis for a single stone"""
        st.subheader(f"🔍 Stone #{stone_number} Analysis")
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        
        size_info = analysis['size_info']
        detection = analysis['detection']
        treatment_rec = analysis['treatment_rec']
        
        with col1:
            st.metric("🔍 Size (mm)", f"{size_info['equivalent_diameter_mm']:.1f}")
            st.metric("📏 Width (mm)", f"{size_info['width_mm']:.1f}")
            
        with col2:
            st.metric("📐 Height (mm)", f"{size_info['height_mm']:.1f}") 
            st.metric("📊 Area (mm²)", f"{size_info['area_mm2']:.1f}")
            
        with col3:
            confidence_key = 'confidence' if 'confidence' in detection else 'conf'
            confidence_val = detection.get(confidence_key, 0)
            st.metric("🎯 Confidence", f"{confidence_val:.1%}")
            
            # Color-coded urgency
            urgency = treatment_rec['urgency']
            if urgency == 'Urgent':
                st.error(f"⚠️ {urgency}")
            elif urgency == 'High':
                st.warning(f"⚡ {urgency}")
            elif urgency == 'Moderate':
                st.info(f"📋 {urgency}")
            else:
                st.success(f"✅ {urgency}")
        
        # Detailed information
        st.subheader("📍 Location & Clinical Details")
        
        col1, col2 = st.columns(2)
        
        location_info = analysis['location_info']
        
        with col1:
            st.write(f"**Anatomical Location:** {location_info['side']} {location_info['anatomical_region']}")
            st.write(f"**Stone Category:** {treatment_rec['category']}")
            st.write(f"**Natural Passage Probability:** {treatment_rec['pass_probability']}")
            
        with col2:
            st.write(f"**Recommended Treatment:** {treatment_rec['treatment']}")
            st.write(f"**Bounding Box:** {detection['bbox']}")
            st.write(f"**Center Coordinates:** {detection['center']}")

    def display_summary(self, analyses):
        """Display summary of all detected stones"""
        if not analyses:
            return
            
        st.subheader("📈 Detection Summary")
        
        # Overall statistics
        total_stones = len(analyses)
        sizes = [analysis['size_info']['equivalent_diameter_mm'] for analysis in analyses]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Stones", total_stones)
        with col2:
            st.metric("Largest (mm)", f"{max(sizes):.1f}")
        with col3:
            st.metric("Smallest (mm)", f"{min(sizes):.1f}")  
        with col4:
            st.metric("Average (mm)", f"{np.mean(sizes):.1f}")
        
        # Risk assessment
        st.subheader("⚕️ Clinical Assessment")
        
        urgent_count = sum(1 for analysis in analyses if analysis['treatment_rec']['urgency'] in ['Urgent', 'High'])
        
        if urgent_count > 0:
            st.error(f"🚨 {urgent_count} stone(s) require urgent/high priority treatment")
        else:
            st.success("✅ No stones require urgent intervention")
        
        # Treatment summary
        treatments = {}
        for analysis in analyses:
            treatment = analysis['treatment_rec']['treatment']
            treatments[treatment] = treatments.get(treatment, 0) + 1
        
        st.write("**Treatment Summary:**")
        for treatment, count in treatments.items():
            st.write(f"- {treatment}: {count} stone(s)")
    
    def display_report_content(self, report_data):
        """Display the report content in the UI"""
        st.markdown('<div class="report-section" style="width: 100%">', unsafe_allow_html=True)
        st.header("📋 Medical Report")
        
        # Patient Information
        st.subheader("👤 Patient Information")
        patient_info = report_data['patient_info']
        
        col1a, col2a = st.columns(2)
        with col1a:
            st.write(f"**Name:** {patient_info['name']}")
            st.write(f"**Age:** {patient_info['age']} years")
            st.write(f"**Gender:** {patient_info['gender']}")
        with col2a:
            st.write(f"**Patient ID:** {patient_info.get('patient_id', 'N/A')}")
            st.write(f"**Examination Date:** {patient_info['examination_date']}")
            st.write(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Executive Summary
        st.subheader("📊 Executive Summary")
        num_stones = len(report_data['detections'])
        st.write(f"**Number of kidney stones detected:** {num_stones}")
        
        if num_stones > 0:
            largest_stone = max(report_data['analysis_results'], 
                               key=lambda x: x['size_info']['equivalent_diameter_mm'])
            st.write(f"**Largest stone size:** {largest_stone['size_info']['equivalent_diameter_mm']:.1f} mm")
            st.write(f"**Treatment urgency:** {largest_stone['treatment_rec']['urgency']}")
        
        # Detailed Findings
        st.subheader("🔍 Detailed Findings")
        for i, analysis in enumerate(report_data['analysis_results']):
            st.write(f"**Stone #{i+1}:**")
            
            size_info = analysis['size_info']
            location_info = analysis['location_info']
            treatment_rec = analysis['treatment_rec']
            
            col1b, col2b = st.columns(2)
            with col1b:
                st.write(f"• Size: {size_info['equivalent_diameter_mm']:.1f} mm (equivalent diameter)")
                st.write(f"• Dimensions: {size_info['width_mm']:.1f} × {size_info['height_mm']:.1f} mm")
                st.write(f"• Area: {size_info['area_mm2']:.1f} mm²")
            with col2b:
                st.write(f"• Location: {location_info['side']} {location_info['anatomical_region']}")
                st.write(f"• Category: {treatment_rec['category']}")
                st.write(f"• Recommended Treatment: {treatment_rec['treatment']}")
                st.write(f"• Urgency Level: {treatment_rec['urgency']}")
        
        # Clinical Recommendations
        st.subheader("⚕️ Clinical Recommendations")
        recommendations = self._generate_clinical_recommendations(report_data)
        for rec in recommendations:
            st.write(f"• {rec}")
        
        # Technical Information
        st.subheader("🔧 Technical Information")
        st.write("• Analysis performed using YOLOv8 deep learning model")
        st.write("• Size measurements calibrated for medical imaging standards")
        st.write("• This report is generated by AI and should be reviewed by a qualified physician")
        st.write("• For clinical decision-making, correlation with symptoms and other imaging is recommended")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _generate_clinical_recommendations(self, report_data):
        """Generate clinical recommendations based on findings"""
        recommendations = []
        
        num_stones = len(report_data['detections'])
        if num_stones == 0:
            recommendations.append("No kidney stones detected. Continue routine follow-up.")
            return recommendations
        
        # Analyze all stones for recommendations
        large_stones = 0
        urgent_stones = 0
        
        for analysis in report_data['analysis_results']:
            size_mm = analysis['size_info']['equivalent_diameter_mm']
            urgency = analysis['treatment_rec']['urgency']
            
            if size_mm > 6:
                large_stones += 1
            if urgency in ['High', 'Urgent']:
                urgent_stones += 1
        
        # General recommendations
        recommendations.append("Increase fluid intake to 2-3 liters per day unless contraindicated")
        
        if urgent_stones > 0:
            recommendations.append("URGENT: Immediate urological consultation recommended")
            recommendations.append("Consider pain management if patient is symptomatic")
        elif large_stones > 0:
            recommendations.append("Urological consultation recommended within 1-2 weeks")
            recommendations.append("Consider medical expulsive therapy if appropriate")
        else:
            recommendations.append("Conservative management with watchful waiting")
            recommendations.append("Follow-up imaging in 3-6 months")
        
        recommendations.append("Dietary counseling for stone prevention")
        recommendations.append("Metabolic evaluation if recurrent stones")
        
        return recommendations
    
    def generate_and_display_report(self, detections, analyses, patient_info):
        """Generate comprehensive medical report and display it in the UI"""
        
        with st.spinner("📄 Generating comprehensive medical report..."):
            try:
                # Extract patient info from dictionary
                patient_name = patient_info.get('name', 'Unknown')
                patient_age = patient_info.get('age', 0)
                patient_gender = patient_info.get('gender', 'Unknown')
                examination_date = patient_info.get('examination_date', datetime.now().date())
                patient_id = patient_info.get('patient_id', '')
                
                report_data = {
                    'patient_info': {
                        'name': patient_name,
                        'age': patient_age,
                        'gender': patient_gender,
                        'examination_date': examination_date,
                        'patient_id': patient_id
                    },
                    'detections': detections,
                    'analysis_results': analyses
                }
                
                # Display report content in UI first
                self.display_report_content(report_data)
                
                # Generate PDF report
                try:
                    # Get annotated image from session state
                    annotated_image = st.session_state.get('annotated_image', None)
                    
                    pdf_path = self.report_generator.generate_comprehensive_report(report_data, annotated_image)
                    
                    if pdf_path and os.path.exists(pdf_path):
                        with open(pdf_path, "rb") as pdf_file:
                            pdf_data = pdf_file.read()
                            
                            # Center the PDF success messages and download button
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                st.success("✅ Medical report generated successfully!")
                                
                                # Download button
                                filename = f"kidney_stone_report_{patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                                
                                st.download_button(
                                    label="📥 Download Medical Report (PDF)",
                                    data=pdf_data,
                                    file_name=filename,
                                    mime="application/pdf",
                                    width="stretch",
                                    type="primary"
                                )
                                
                                # Display report info
                                st.info(f"📋 Report contains detailed analysis of {len(detections)} detected stone(s)")
                    else:
                        st.warning("⚠️ Report content displayed above, but PDF generation failed. You can still view all the analysis results.")
                        st.error("PDF file could not be created. Please check the file system permissions.")
                        
                except Exception as pdf_error:
                    st.warning("⚠️ Report content displayed above, but PDF generation failed.")
                    st.error(f"PDF Generation Error: {str(pdf_error)}")
                    st.info("The analysis results are still available above. You may need to install required PDF libraries.")
                    
            except Exception as e:
                st.error(f"❌ Error generating report: {str(e)}")
                st.info("Please check that all required dependencies are installed.")
                import traceback
                st.error(f"Full error: {traceback.format_exc()}")  # Debug

if __name__ == "__main__":
    app = KidneyStoneApp()
    app.main()