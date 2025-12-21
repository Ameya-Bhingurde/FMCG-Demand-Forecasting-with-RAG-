# 🚀 FMCG Pipeline - Quick Start Guide

## ⚡ Get Started in 5 Minutes

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test Your Setup
```bash
python test_setup.py
```

### 3. Run the Pipeline
```bash
# Option A: Use the startup script (recommended)
python start_pipeline.py

# Option B: Run directly
python run_pipeline.py
```

### 4. Launch Dashboard
```bash
streamlit run src/dashboard_app.py
```

## 📁 What You Get

After running the pipeline, you'll have:

### 🎯 Trained Models
- **Prophet**: Time series forecasting model
- **XGBoost**: Sales prediction model
- Location: `models/`

### 📊 Reports & Visualizations
- Model evaluation metrics
- SHAP explainability plots
- Interactive visualizations
- Location: `reports/`

### 🤖 RAG System
- FAISS vector database
- Natural language querying
- Business insights
- Location: `vector_store/`

### 📈 Dashboard Features
- Sales overview and trends
- Real-time forecasting
- Model explanations
- AI-powered business queries

## 🔍 Sample Queries

Ask the RAG system questions like:
- "What were the total sales in 2023?"
- "Which product had the highest sales?"
- "How did promotions affect sales performance?"
- "What caused the sales dip in Q2 2023?"

## 🛠️ Troubleshooting

### Common Issues
1. **Memory errors**: Reduce batch sizes in `config.yaml`
2. **Import errors**: Run `pip install -r requirements.txt`
3. **Model loading**: Ensure all dependencies are installed
4. **Dashboard issues**: Check Streamlit installation

### Get Help
- Check logs in `logs/fmcg_pipeline.log`
- Review `README.md` for detailed documentation
- Run `python test_setup.py` to diagnose issues

## 🎉 Success!

Your FMCG analytics pipeline is ready! You now have:
- ✅ Production-ready ML models
- ✅ Interactive dashboard
- ✅ AI-powered business insights
- ✅ Comprehensive documentation

**Ready to transform your FMCG business with AI! 🚀** 