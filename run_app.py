"""
Quick start script for the AI Image Colorization app.
This script activates the virtual environment and runs the Flask app.
"""

import subprocess
import sys
import os

def main():
    # Check if we're in a virtual environment
    venv_python = os.path.join('venv', 'Scripts', 'python.exe')
    
    if os.path.exists(venv_python):
        print("Starting Flask application with virtual environment...")
        print("=" * 60)
        print("AI Image Colorization Web Application")
        print("=" * 60)
        print("\nServer will start at: http://localhost:5000")
        print("Press Ctrl+C to stop the server\n")
        
        # Run the app
        subprocess.run([venv_python, 'app.py'])
    else:
        print("Virtual environment not found!")
        print("Please run: python -m venv venv")
        print("Then: .\\venv\\Scripts\\Activate.ps1")
        print("Then: pip install -r requirements.txt")
        sys.exit(1)

if __name__ == '__main__':
    main()

