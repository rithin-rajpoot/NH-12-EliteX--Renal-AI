#!/usr/bin/env python3
"""
Kidney Stone Detector - Deployment Script
==========================================

This script provides an easy way to deploy and run the Kidney Stone Detection
Streamlit application with proper environment setup and error handling.

Usage:
    python deploy.py [options]

Options:
    --port PORT         Port number to run the application (default: 8501)
    --host HOST         Host address to bind to (default: localhost)
    --dev               Run in development mode with debug features
    --check             Check system requirements and dependencies
    --install           Install missing dependencies
    --help              Show this help message

Example:
    python deploy.py --port 8080 --host 0.0.0.0
"""

import os
import sys
import subprocess
import argparse
import webbrowser
import time
from pathlib import Path
import importlib.util

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_colored(message, color=Colors.WHITE):
    """Print colored message to terminal"""
    print(f"{color}{message}{Colors.END}")

def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                       🏥 RenalAI 🏥                         ║
    ║                 AI-Powered Medical Imaging                   ║
    ║                     Deployment Script                        ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print_colored(banner, Colors.CYAN)

def check_python_version():
    """Check if Python version is compatible"""
    print_colored("🐍 Checking Python version...", Colors.BLUE)
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_colored("❌ Python 3.8+ is required!", Colors.RED)
        print_colored(f"   Current version: {version.major}.{version.minor}.{version.micro}", Colors.RED)
        return False
    
    print_colored(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible", Colors.GREEN)
    return True

def check_virtual_environment():
    """Check if running in virtual environment"""
    print_colored("📦 Checking virtual environment...", Colors.BLUE)
    
    in_venv = (
        hasattr(sys, 'real_prefix') or 
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
        'VIRTUAL_ENV' in os.environ
    )
    
    if in_venv:
        venv_path = os.environ.get('VIRTUAL_ENV', 'Unknown')
        print_colored(f"✅ Virtual environment active: {venv_path}", Colors.GREEN)
    else:
        print_colored("⚠️  Not running in virtual environment", Colors.YELLOW)
        print_colored("   Recommendation: Use virtual environment for better dependency management", Colors.YELLOW)
    
    return in_venv

def check_dependencies():
    """Check if required dependencies are installed"""
    print_colored("📋 Checking dependencies...", Colors.BLUE)
    
    required_packages = [
        'streamlit',
        'opencv-python',
        'ultralytics',
        'pillow',
        'numpy',
        'pandas',
        'matplotlib',
        'fpdf2',
        'torch',
        'torchvision'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            # Handle special cases
            if package == 'opencv-python':
                importlib.import_module('cv2')
            elif package == 'pillow':
                importlib.import_module('PIL')
            elif package == 'fpdf2':
                importlib.import_module('fpdf')
            else:
                importlib.import_module(package)
            
            print_colored(f"   ✅ {package}", Colors.GREEN)
        except ImportError:
            print_colored(f"   ❌ {package}", Colors.RED)
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies(packages):
    """Install missing dependencies"""
    if not packages:
        return True
    
    print_colored(f"🔧 Installing {len(packages)} missing packages...", Colors.BLUE)
    
    try:
        for package in packages:
            print_colored(f"   Installing {package}...", Colors.YELLOW)
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print_colored(f"   ✅ {package} installed", Colors.GREEN)
        
        print_colored("✅ All dependencies installed successfully!", Colors.GREEN)
        return True
    
    except subprocess.CalledProcessError as e:
        print_colored(f"❌ Failed to install dependencies: {e}", Colors.RED)
        return False

def check_project_structure():
    """Check if required project files exist"""
    print_colored("🗂️  Checking project structure...", Colors.BLUE)
    
    required_files = [
        'streamlit_app/app.py',
        'src/detect_stones.py',
        'src/size_calculator.py',
        'src/report_generator.py',
        'requirements.txt'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print_colored(f"   ✅ {file_path}", Colors.GREEN)
        else:
            print_colored(f"   ❌ {file_path}", Colors.RED)
            missing_files.append(file_path)
    
    return missing_files

def check_models():
    """Check if model files exist"""
    print_colored("🤖 Checking model files...", Colors.BLUE)
    
    model_paths = [
        'models/kidney_stone_transfer_20250920_192030/weights/best.pt',
        'models/best_kidney_stone_yolov8n.pt',
        'models/quick_kidney_stone.pt',
        'yolov8n.pt'
    ]
    
    found_models = []
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print_colored(f"   ✅ {model_path}", Colors.GREEN)
            found_models.append(model_path)
        else:
            print_colored(f"   ❌ {model_path}", Colors.YELLOW)
    
    if not found_models:
        print_colored("⚠️  No trained models found. The app will download yolov8n.pt automatically.", Colors.YELLOW)
    
    return found_models

def run_system_check():
    """Run complete system check"""
    print_colored("🔍 Running system diagnostics...", Colors.BOLD)
    print()
    
    # Check Python version
    if not check_python_version():
        return False
    print()
    
    # Check virtual environment
    check_virtual_environment()
    print()
    
    # Check dependencies
    missing_deps = check_dependencies()
    print()
    
    # Check project structure
    missing_files = check_project_structure()
    print()
    
    # Check models
    available_models = check_models()
    print()
    
    # Summary
    print_colored("📊 System Check Summary:", Colors.BOLD)
    
    if missing_files:
        print_colored(f"❌ Missing {len(missing_files)} required files", Colors.RED)
        return False
    
    if missing_deps:
        print_colored(f"⚠️  Missing {len(missing_deps)} dependencies", Colors.YELLOW)
        return len(missing_deps) == 0  # Return False if there are missing dependencies
    
    print_colored("✅ All checks passed! System ready for deployment.", Colors.GREEN)
    return True

def create_startup_script():
    """Create platform-specific startup scripts"""
    print_colored("📝 Creating startup scripts...", Colors.BLUE)
    
    # Windows batch file
    batch_content = """@echo off
title Kidney Stone Detector
echo Starting Kidney Stone Detector...
python deploy.py
pause
"""
    
    with open('start_kidney_detector.bat', 'w') as f:
        f.write(batch_content)
    
    # Unix shell script
    shell_content = """#!/bin/bash
echo "Starting Kidney Stone Detector..."
python3 deploy.py
read -p "Press Enter to continue..."
"""
    
    with open('start_kidney_detector.sh', 'w') as f:
        f.write(shell_content)
    
    # Make shell script executable on Unix systems
    try:
        os.chmod('start_kidney_detector.sh', 0o755)
    except:
        pass
    
    print_colored("✅ Startup scripts created:", Colors.GREEN)
    print_colored("   - start_kidney_detector.bat (Windows)", Colors.GREEN)
    print_colored("   - start_kidney_detector.sh (Unix/Linux/Mac)", Colors.GREEN)

def run_streamlit_app(host='localhost', port=8501, dev_mode=False):
    """Run the Streamlit application"""
    print_colored(f"🚀 Starting Kidney Stone Detector on {host}:{port}...", Colors.BOLD)
    
    # Change to project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Streamlit command
    cmd = [
        sys.executable, '-m', 'streamlit', 'run',
        'streamlit_app/app.py',
        '--server.port', str(port),
        '--server.address', host,
        '--server.headless', 'false' if dev_mode else 'true'
    ]
    
    if dev_mode:
        cmd.extend(['--server.runOnSave', 'true'])
        cmd.extend(['--server.allowRunOnSave', 'true'])
    
    try:
        # Print startup information
        print_colored("🌐 Application Details:", Colors.CYAN)
        print_colored(f"   URL: http://{host}:{port}", Colors.WHITE)
        print_colored(f"   Mode: {'Development' if dev_mode else 'Production'}", Colors.WHITE)
        print_colored(f"   Directory: {os.getcwd()}", Colors.WHITE)
        print()
        
        print_colored("🎯 Opening browser in 3 seconds...", Colors.YELLOW)
        
        # Start the Streamlit process
        process = subprocess.Popen(cmd)
        
        # Wait a moment then open browser
        time.sleep(3)
        
        # Open browser
        if host in ['localhost', '127.0.0.1']:
            webbrowser.open(f'http://localhost:{port}')
        
        print_colored("✅ Application started successfully!", Colors.GREEN)
        print_colored("💡 Press Ctrl+C to stop the application", Colors.YELLOW)
        
        # Wait for the process
        process.wait()
        
    except KeyboardInterrupt:
        print_colored("\n🛑 Application stopped by user", Colors.YELLOW)
        return True
    except Exception as e:
        print_colored(f"❌ Failed to start application: {e}", Colors.RED)
        return False

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(
        description='Deploy Kidney Stone Detector Application',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--port', type=int, default=8501,
                        help='Port number (default: 8501)')
    parser.add_argument('--host', default='localhost',
                        help='Host address (default: localhost)')
    parser.add_argument('--dev', action='store_true',
                        help='Run in development mode')
    parser.add_argument('--check', action='store_true',
                        help='Run system check only')
    parser.add_argument('--install', action='store_true',
                        help='Install missing dependencies')
    parser.add_argument('--create-scripts', action='store_true',
                        help='Create startup scripts')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Handle create scripts option
    if args.create_scripts:
        create_startup_script()
        return
    
    # Run system check
    system_ready = run_system_check()
    
    # Handle check-only mode
    if args.check:
        if system_ready:
            print_colored("🎉 System is ready for deployment!", Colors.GREEN)
            sys.exit(0)
        else:
            print_colored("❌ System check failed. Use --install to fix dependencies.", Colors.RED)
            sys.exit(1)
    
    # Handle install mode
    if args.install or not system_ready:
        missing_deps = check_dependencies()
        if missing_deps:
            if install_dependencies(missing_deps):
                print_colored("🔄 Re-running system check...", Colors.BLUE)
                system_ready = run_system_check()
            else:
                print_colored("❌ Installation failed. Please install dependencies manually.", Colors.RED)
                sys.exit(1)
    
    # Start application if system is ready
    if system_ready:
        print()
        print_colored("🚀 All systems go! Starting application...", Colors.BOLD)
        print()
        run_streamlit_app(args.host, args.port, args.dev)
    else:
        print_colored("❌ Cannot start application. Please fix the issues above.", Colors.RED)
        print_colored("💡 Try running: python deploy.py --install", Colors.YELLOW)
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_colored("\n👋 Deployment cancelled by user", Colors.YELLOW)
        sys.exit(0)
    except Exception as e:
        print_colored(f"\n💥 Deployment failed: {e}", Colors.RED)
        sys.exit(1)