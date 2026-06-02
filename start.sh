#!/bin/bash

# Hugging Face Spaces startup script
# This script runs before the Streamlit app starts

echo "🚀 Starting FMCG Demand Forecasting Platform..."

# Navigate to the fmcg_genai directory
cd fmcg_genai

# Check if models exist, if not run the pipeline
if [ ! -f "models/prophet.pkl" ] || [ ! -f "models/xgboost_sales.pkl" ]; then
    echo "📦 Models not found. Running pipeline to train models..."
    python run_pipeline.py
else
    echo "✅ Models found. Skipping training."
fi

# Check if vector store exists; if not, build it via run_rag_pipeline()
# (the public method that handles load_sales_data → create_documents →
#  create_vector_store → persist faiss_index.bin + documents.pkl + embeddings.pkl)
if [ ! -f "vector_store/faiss_index.bin" ]; then
    echo "🔍 Vector store not found. Building it now..."
    python -c "from src.rag_pipeline import FMCGRAGPipeline; FMCGRAGPipeline('config.yaml').run_rag_pipeline()"
else
    echo "✅ Vector store found."
fi

echo "✨ Setup complete! Launching dashboard..."

# Start the Streamlit app
streamlit run src/dashboard_app_enhanced.py --server.port=7860 --server.address=0.0.0.0
