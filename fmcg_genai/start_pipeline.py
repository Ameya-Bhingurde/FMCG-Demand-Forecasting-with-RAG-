#!/usr/bin/env python3
"""
Simple startup script for FMCG Pipeline
Provides user-friendly interface to run the pipeline
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    """Print welcome banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    FMCG Sales Analytics Pipeline                             ║
║                                                                              ║
║  Welcome to the FMCG Sales Analytics & Forecasting Pipeline!                ║
║  This system provides end-to-end ML + Generative AI for business insights.  ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_environment():
    """Check if the environment is properly set up"""
    print("Checking environment...")
    
    # Check if we're in the right directory
    if not Path("config.yaml").exists():
        print("Error: config.yaml not found. Please run this script from the fmcg_genai directory.")
        return False
    
    # Check if data directory exists
    if not Path("data/raw").exists():
        print("Error: data/raw directory not found.")
        return False
    
    # Check if requirements are installed
    try:
        import pandas
        import numpy
        import yaml
        print("Basic dependencies found")
        print("Environment check passed!")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    return True

def show_menu():
    """Show the main menu"""
    menu = """
Available Options:

1. Test Setup - Verify everything is working
2. Run Full Pipeline - Complete end-to-end workflow
3. Launch Dashboard - Start the Streamlit dashboard
4. Run Individual Steps - Choose specific pipeline steps
5. View Documentation - Open README
6. Exit

Enter your choice (1-6): """
    
    return input(menu).strip()

def run_test_setup():
    """Run the test setup script"""
    print("\nRunning setup test...")
    try:
        result = subprocess.run([sys.executable, "test_setup.py"], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Test failed: {e}")
        return False

def run_full_pipeline():
    """Run the complete pipeline"""
    print("\nStarting full pipeline...")
    print("This will run all steps: preprocessing -> feature engineering -> training -> evaluation -> SHAP -> RAG")
    print("Estimated time: 10-30 minutes depending on your system")
    
    confirm = input("\nContinue? (y/n): ").lower().strip()
    if confirm != 'y':
        print("Pipeline cancelled.")
        return
    
    try:
        result = subprocess.run([sys.executable, "run_pipeline.py"], 
                              capture_output=False)
        if result.returncode == 0:
            print("\nPipeline completed successfully!")
        else:
            print("\nPipeline failed. Check the logs for details.")
    except Exception as e:
        print(f"Pipeline failed: {e}")

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("\nLaunching Streamlit dashboard...")
    print("The dashboard will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/dashboard_app.py"])
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except Exception as e:
        print(f"Failed to launch dashboard: {e}")

def run_individual_steps():
    """Run individual pipeline steps"""
    steps = {
        "1": ("Data Preprocessing", "src/data_preprocessing.py"),
        "2": ("Feature Engineering", "src/feature_engineering.py"),
        "3": ("Model Training", "src/train_models.py"),
        "4": ("Model Evaluation", "src/evaluate_models.py"),
        "5": ("SHAP Explainability", "src/explainability.py"),
        "6": ("RAG Pipeline", "src/rag_pipeline.py")
    }
    
    print("\nAvailable Steps:")
    for key, (name, script) in steps.items():
        print(f"{key}. {name}")
    
    choice = input("\nEnter step number (1-6): ").strip()
    
    if choice in steps:
        name, script = steps[choice]
        print(f"\nRunning {name}...")
        try:
            subprocess.run([sys.executable, script])
        except Exception as e:
            print(f"Failed to run {name}: {e}")
    else:
        print("Invalid choice.")

def view_documentation():
    """Open the README file"""
    readme_path = Path("README.md")
    if readme_path.exists():
        print("\nOpening README.md...")
        try:
            if sys.platform == "win32":
                os.startfile(readme_path)
            else:
                subprocess.run(["xdg-open", str(readme_path)])
        except Exception as e:
            print(f"Could not open README: {e}")
            print("You can manually open README.md in your text editor.")
    else:
        print("README.md not found.")

def main():
    """Main function"""
    print_banner()
    
    if not check_environment():
        return
    
    while True:
        choice = show_menu()
        
        if choice == "1":
            run_test_setup()
        elif choice == "2":
            run_full_pipeline()
        elif choice == "3":
            launch_dashboard()
        elif choice == "4":
            run_individual_steps()
        elif choice == "5":
            view_documentation()
        elif choice == "6":
            print("\nGoodbye! Thanks for using the FMCG Pipeline.")
            break
        else:
            print("Invalid choice. Please enter 1-6.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main() 