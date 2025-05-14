"""
Hand Gesture Recognition Setup Script
"""

import subprocess
import sys

def setup_environment():
    """Setup the conda environment and install packages"""
    
    print("Setting up Hand Gesture Recognition environment...")
    
    # Create conda environment
    subprocess.run([
        "conda", "create", "-n", "hand_gesture", "python=3.9", "-y"
    ])
    
    # Install packages
    packages = [
        "opencv-python",
        "mediapipe",
        "numpy",
        "scikit-learn"
    ]
    
    print("Installing required packages...")
    subprocess.run([
        sys.executable, "-m", "pip", "install"
    ] + packages)
    
    print("\nSetup complete!")
    print("To run the application:")
    print("1. Activate the environment: conda activate hand_gesture")
    print("2. Run the script: python hand_gesture_recognition.py")

if __name__ == "__main__":
    setup_environment()