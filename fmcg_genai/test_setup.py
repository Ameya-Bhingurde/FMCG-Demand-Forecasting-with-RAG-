# -*- coding: utf-8 -*-
"""
Test script to verify FMCG pipeline setup
Checks imports, data availability, and basic functionality
"""

import sys
import io
from pathlib import Path
import yaml

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import yaml
        print("✅ yaml imported successfully")
    except ImportError as e:
        print(f"❌ yaml import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ matplotlib imported successfully")
    except ImportError as e:
        print(f"❌ matplotlib import failed: {e}")
        return False
    
    try:
        import plotly.graph_objects as go
        print("✅ plotly imported successfully")
    except ImportError as e:
        print(f"❌ plotly import failed: {e}")
        return False
    
    try:
        import streamlit as st
        print("✅ streamlit imported successfully")
    except ImportError as e:
        print(f"❌ streamlit import failed: {e}")
        return False
    
    try:
        import xgboost as xgb
        print("✅ xgboost imported successfully")
    except ImportError as e:
        print(f"❌ xgboost import failed: {e}")
        return False
    
    try:
        from prophet import Prophet
        print("✅ prophet imported successfully")
    except ImportError as e:
        print(f"❌ prophet import failed: {e}")
        return False
    
    try:
        import shap
        print("✅ shap imported successfully")
    except ImportError as e:
        print(f"❌ shap import failed: {e}")
        return False
    
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        print("✅ langchain imported successfully")
    except ImportError as e:
        print(f"❌ langchain import failed: {e}")
        return False
    
    try:
        import faiss
        print("✅ faiss imported successfully")
    except ImportError as e:
        print(f"❌ faiss import failed: {e}")
        return False
    
    return True

def test_project_structure():
    """Test if project structure is correct"""
    print("\n🔍 Testing project structure...")
    
    required_dirs = [
        "data/raw",
        "data/processed", 
        "src",
        "models",
        "vector_store",
        "reports",
        "reports/shap"
    ]
    
    required_files = [
        "config.yaml",
        "requirements.txt",
        "run_pipeline.py",
        "README.md",
        "src/data_preprocessing.py",
        "src/feature_engineering.py",
        "src/train_models.py",
        "src/evaluate_models.py",
        "src/explainability.py",
        "src/rag_pipeline.py",
        "src/dashboard_app.py"
    ]
    
    all_good = True
    
    # Check directories
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ Directory exists: {dir_path}")
        else:
            print(f"❌ Directory missing: {dir_path}")
            all_good = False
    
    # Check files
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ File exists: {file_path}")
        else:
            print(f"❌ File missing: {file_path}")
            all_good = False
    
    return all_good

def test_data_availability():
    """Test if data files are available"""
    print("\n🔍 Testing data availability...")
    
    raw_data_dir = Path("data/raw")
    if not raw_data_dir.exists():
        print("❌ Raw data directory not found")
        return False
    
    # Check for CSV files
    csv_files = list(raw_data_dir.glob("*.csv"))
    if csv_files:
        print(f"✅ Found {len(csv_files)} CSV files in data/raw/")
        for file in csv_files:
            print(f"   - {file.name}")
    else:
        print("❌ No CSV files found in data/raw/")
        return False
    
    # Check for parquet files
    parquet_files = list(raw_data_dir.glob("*.parquet"))
    if parquet_files:
        print(f"✅ Found {len(parquet_files)} parquet files in data/raw/")
        for file in parquet_files:
            print(f"   - {file.name}")
    
    return True

def test_config():
    """Test if configuration file is valid"""
    print("\n🔍 Testing configuration...")
    
    try:
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['data', 'models', 'reports', 'preprocessing', 'features', 'models_config', 'rag']
        
        for section in required_sections:
            if section in config:
                print(f"✅ Config section found: {section}")
            else:
                print(f"❌ Config section missing: {section}")
                return False
        
        print("✅ Configuration file is valid")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_src_imports():
    """Test if src modules can be imported"""
    print("\n🔍 Testing src module imports...")
    
    # Add src to path
    sys.path.append(str(Path("src")))
    
    modules = [
        "data_preprocessing",
        "feature_engineering", 
        "train_models",
        "evaluate_models",
        "explainability",
        "rag_pipeline"
    ]
    
    all_good = True
    
    for module in modules:
        try:
            __import__(module)
            print(f"✅ Module imported successfully: {module}")
        except Exception as e:
            print(f"❌ Module import failed: {module} - {e}")
            all_good = False
    
    return all_good

def main():
    """Run all tests"""
    # Fix encoding for Windows
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("FMCG Pipeline Setup Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Project Structure", test_project_structure),
        ("Data Availability", test_data_availability),
        ("Configuration", test_config),
        ("Source Modules", test_src_imports)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 30)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your FMCG pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Run the pipeline: python run_pipeline.py")
        print("2. Launch dashboard: streamlit run src/dashboard_app.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 