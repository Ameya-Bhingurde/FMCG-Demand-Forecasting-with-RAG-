# 🚀 FMCG Sales Analytics & Forecasting Pipeline

A comprehensive **end-to-end ML + Generative AI (RAG)** workflow for FMCG sales forecasting and business insights. This production-grade system combines traditional machine learning with cutting-edge generative AI to provide actionable business intelligence.

## 📊 Project Overview

This project demonstrates a complete **resume-ready** implementation of:
- **Data Preprocessing & Feature Engineering**
- **Time Series Forecasting** (Prophet)
- **Machine Learning** (XGBoost)
- **Model Explainability** (SHAP)
- **Generative AI** (RAG with LangChain + FAISS)
- **Interactive Dashboard** (Streamlit)

## 🏗️ Architecture

```
fmcg_genai/
│── data/
│   ├── raw/                  # Raw FMCG datasets
│   └── processed/            # Cleaned & engineered data
├── src/
│   ├── data_preprocessing.py     # Data cleaning & time-series split
│   ├── feature_engineering.py    # Feature creation & encoding
│   ├── train_models.py           # Prophet & XGBoost training
│   ├── evaluate_models.py        # Model evaluation & metrics
│   ├── explainability.py         # SHAP explanations
│   ├── rag_pipeline.py           # RAG system for queries
│   └── dashboard_app.py          # Streamlit dashboard
├── models/                   # Trained models (joblib/pkl)
├── vector_store/             # FAISS index for RAG
├── reports/                  # Evaluation reports & SHAP plots
├── config.yaml               # Central configuration
├── requirements.txt          # Dependencies
├── run_pipeline.py           # Main orchestrator
└── README.md                 # This file
```

## 🛠️ Tech Stack

### Core ML & Data Science
- **Python 3.8+** - Main programming language
- **Pandas & NumPy** - Data manipulation
- **Scikit-learn** - Machine learning utilities
- **XGBoost** - Gradient boosting for sales prediction
- **Prophet** - Time series forecasting

### Generative AI & RAG
- **LangChain** - RAG pipeline orchestration
- **FAISS** - Vector similarity search
- **HuggingFace Transformers** - Embeddings & LLMs
- **Sentence Transformers** - Text embeddings

### Visualization & Explainability
- **SHAP** - Model explainability
- **Plotly** - Interactive visualizations
- **Matplotlib & Seaborn** - Static plots

### Dashboard & Deployment
- **Streamlit** - Interactive web dashboard
- **PyYAML** - Configuration management

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd fmcg_genai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure the Project

Edit `config.yaml` to customize:
- Data paths
- Model parameters
- RAG settings
- API keys (if using OpenAI)

### 3. Run the Complete Pipeline

```bash
# Run all steps
python run_pipeline.py

# Or run specific steps
python run_pipeline.py --skip preprocessing feature_engineering
```

### 4. Launch the Dashboard

```bash
streamlit run src/dashboard_app.py
```

## 📋 Pipeline Steps

### 1. Data Preprocessing (`data_preprocessing.py`)
- Loads FMCG sales data from `data/raw/`
- Handles missing values and outliers
- Creates time-series train/test split
- Saves cleaned data to `data/processed/`

**Key Features:**
- Automatic outlier detection using IQR method
- Time-series aware splitting (train up to mid-2023, test late 2023-2024)
- Comprehensive data validation and cleaning

### 2. Feature Engineering (`feature_engineering.py`)
- Creates lag features (1, 7, 14, 30 days)
- Generates rolling averages and statistics
- Adds time-based features (month, quarter, day-of-week)
- Includes holiday and seasonal features
- Encodes categorical variables

**Key Features:**
- 50+ engineered features
- Holiday calendar integration
- Seasonal decomposition
- Categorical encoding with label encoders

### 3. Model Training (`train_models.py`)
- Trains **Prophet** for time series forecasting
- Trains **XGBoost** for sales prediction
- Saves models to `models/` directory

**Models:**
- **Prophet**: Captures trends, seasonality, and holidays
- **XGBoost**: Handles complex feature interactions

### 4. Model Evaluation (`evaluate_models.py`)
- Evaluates both models using multiple metrics
- Generates comprehensive visualizations
- Creates interactive plots with Plotly

**Metrics:**
- MAE, RMSE, MAPE, R²
- Directional accuracy
- Bias analysis

### 5. SHAP Explainability (`explainability.py`)
- Generates global feature importance
- Creates local explanations for individual predictions
- Produces business insights and recommendations

**Outputs:**
- SHAP summary plots
- Waterfall plots for individual predictions
- Feature interaction analysis
- Business recommendations

### 6. RAG Pipeline (`rag_pipeline.py`)
- Converts sales data into text documents
- Builds FAISS vector database
- Implements LangChain QA system
- Answers natural language queries

**Capabilities:**
- Natural language querying
- Context-aware responses
- Source attribution
- Multiple document types (daily summaries, product analysis, regional performance)

### 7. Dashboard (`dashboard_app.py`)
- Interactive Streamlit interface
- Real-time model predictions
- SHAP visualizations
- RAG query interface

**Tabs:**
- Sales Overview
- Forecasting
- Model Explainability
- Business Queries

## 🎯 Usage Examples

### Running Individual Components

```bash
# Data preprocessing only
python src/data_preprocessing.py

# Feature engineering only
python src/feature_engineering.py

# Train models only
python src/train_models.py

# Evaluate models only
python src/evaluate_models.py

# SHAP analysis only
python src/explainability.py

# RAG pipeline only
python src/rag_pipeline.py
```

### Custom Queries via RAG

The RAG system can answer questions like:
- "What were the total sales in 2023?"
- "Which product had the highest sales?"
- "How did promotions affect sales performance?"
- "What caused the sales dip in Q2 2023?"
- "Which region performed best?"

### Dashboard Features

1. **Sales Overview**: Key metrics, trends, regional analysis
2. **Forecasting**: Prophet predictions with confidence intervals
3. **Explainability**: SHAP plots and feature importance
4. **Business Queries**: Natural language Q&A interface

## 📊 Sample Outputs

### Model Performance
```
Prophet Model Metrics:
  MAE: 245.32
  RMSE: 312.45
  MAPE: 8.67%

XGBoost Model Metrics:
  MAE: 198.76
  RMSE: 289.34
  R²: 0.847
  MAPE: 7.23%
```

### Business Insights
- Price optimization is crucial for sales performance
- Historical sales patterns significantly influence future sales
- Promotional activities have strong impact on sales
- Seasonal patterns are important for sales forecasting

## 🔧 Configuration

The `config.yaml` file controls all aspects of the pipeline:

```yaml
# Data paths
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"

# Model parameters
models_config:
  prophet:
    changepoint_prior_scale: 0.05
    seasonality_prior_scale: 10.0
  
  xgboost:
    n_estimators: 1000
    max_depth: 6

# RAG settings
rag:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  llm_model: "google/flan-t5-base"
  chunk_size: 1000
```

## 🚀 Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "src/dashboard_app.py", "--server.port=8501"]
```

### Cloud Deployment
- **AWS**: Deploy on EC2 with Streamlit
- **GCP**: Use Cloud Run for containerized deployment
- **Azure**: Deploy on App Service

## 📈 Performance Optimization

### For Large Datasets
- Use chunked processing in data preprocessing
- Implement parallel feature engineering
- Use GPU acceleration for SHAP calculations
- Optimize FAISS index for faster retrieval

### Memory Management
- Process data in batches
- Use memory-efficient data types
- Implement garbage collection in loops

## 🔍 Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch sizes in config
2. **Model Loading**: Ensure all dependencies are installed
3. **RAG Pipeline**: Check HuggingFace model availability
4. **Dashboard**: Verify Streamlit installation

### Debug Mode
```bash
# Run with verbose logging
python run_pipeline.py --config config_debug.yaml
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Prophet** by Facebook Research
- **XGBoost** by DMLC
- **SHAP** by Microsoft Research
- **LangChain** by Harrison Chase
- **Streamlit** for the dashboard framework

## 📞 Support

For questions or issues:
- Create an issue on GitHub
- Check the logs in `logs/fmcg_pipeline.log`
- Review the configuration in `config.yaml`

---

**🎉 Ready to transform your FMCG business with AI-powered insights!** 