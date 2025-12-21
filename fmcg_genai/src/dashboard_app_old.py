"""
Streamlit Dashboard for FMCG Sales Analytics & Forecasting
Provides interactive interface for business queries, forecasts, and model explanations
"""

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
import sys
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add project root to sys.path to allow imports from src
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Visualization imports
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# ML and RAG imports
from prophet import Prophet
import shap

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Caching the model loading to improve performance and prevent reload on every interaction
@st.cache_resource
def get_models(config):
    """Load trained models with caching"""
    models = {}
    try:
        # Load Prophet model
        prophet_path = Path(config['models']['prophet_model'])
        if prophet_path.exists():
            with open(prophet_path, 'rb') as f:
                models['prophet'] = joblib.load(f)
        else:
             logger.error(f"Prophet model not found at {prophet_path}")

        # Load XGBoost model
        xgboost_path = Path(config['models']['xgboost_model'])
        if xgboost_path.exists():
            with open(xgboost_path, 'rb') as f:
                models['xgboost'] = joblib.load(f)
        else:
            logger.error(f"XGBoost model not found at {xgboost_path}")
            
        return models
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return None

@st.cache_resource
def get_rag_pipeline():
    """Load and setup RAG pipeline with caching"""
    try:
        # Import RAG components - moved here to avoid circular imports or path issues at top level
        from src.rag_pipeline import FMCGRAGPipeline
        
        rag_pipeline = FMCGRAGPipeline()
        
        # Load vector store and setup QA chain
        if rag_pipeline.load_vector_store():
            # Ensure components are ready
            # Note: FMCGRAGPipeline.__init__ initializes embeddings and models
            return rag_pipeline
            
        return None
    except ImportError:
        # Fallback if src import fails (e.g. running from inside src)
        try:
            from rag_pipeline import FMCGRAGPipeline
            rag_pipeline = FMCGRAGPipeline()
            if rag_pipeline.load_vector_store():
                return rag_pipeline
            return None
        except Exception as e:
            logger.error(f"Error importing RAG pipeline fallback: {e}")
            return None
    except Exception as e:
        logger.error(f"Error setting up RAG pipeline: {e}")
        return None

class FMCGDashboard:
    def __init__(self, config_path="config.yaml"):
        """Initialize the dashboard with configuration"""
        # Load config
        try:
            # Handle config path relative to project root or current dir
            if os.path.exists(config_path):
                with open(config_path, 'r') as file:
                    self.config = yaml.safe_load(file)
            else:
                # Try finding it in project root if running from src
                root_config = project_root / config_path
                if root_config.exists():
                    with open(root_config, 'r') as file:
                        self.config = yaml.safe_load(file)
                else:
                    st.error(f"Config file not found at {config_path} or {root_config}")
                    st.stop()
        except Exception as e:
            st.error(f"Error loading config: {e}")
            st.stop()
        
        self.processed_dir = Path(self.config['data']['processed_dir'])
        # Handle relative paths in config by prepending project_root if needed
        if not self.processed_dir.is_absolute():
             self.processed_dir = project_root / self.processed_dir

        self.models_dir = Path(self.config['models']['prophet_model']).parent
        self.reports_dir = Path(self.config['reports']['shap_dir'])
        
        # Initialize components using cached resources
        self.models = get_models(self.config)
        self.prophet_model = self.models.get('prophet') if self.models else None
        self.xgboost_model = self.models.get('xgboost') if self.models else None
        
        self.rag_pipeline = get_rag_pipeline()
        
    def load_data(self):
        """Load processed data"""
        try:
            # Load test data
            test_path = self.processed_dir / "test_features.csv"
            cleaned_path = self.processed_dir / "cleaned.csv"
            
            if not test_path.exists() or not cleaned_path.exists():
                st.error(f"Data files not found in {self.processed_dir}")
                return None, None

            test_df = pd.read_csv(test_path)
            test_df['date'] = pd.to_datetime(test_df['date'])
            
            cleaned_df = pd.read_csv(cleaned_path)
            cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])
            
            return test_df, cleaned_df
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            return None, None
    
    def create_sales_overview(self, cleaned_df):
        """Create sales overview dashboard"""
        st.subheader("📊 Sales Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_sales = cleaned_df['units_sold'].sum()
            st.metric("Total Sales", f"{total_sales:,.0f} units")
        
        with col2:
            avg_price = cleaned_df['price_unit'].mean()
            st.metric("Average Price", f"${avg_price:.2f}")
        
        with col3:
            total_promotions = cleaned_df['promotion_flag'].sum()
            st.metric("Total Promotions", f"{total_promotions:,}")
        
        with col4:
            unique_products = cleaned_df['sku'].nunique()
            st.metric("Unique Products", f"{unique_products}")
        
        # Sales trend over time
        st.subheader("📈 Sales Trend Over Time")
        
        daily_sales = cleaned_df.groupby('date')['units_sold'].sum().reset_index()
        
        fig = px.line(
            daily_sales, 
            x='date', 
            y='units_sold',
            title='Daily Sales Trend',
            labels={'units_sold': 'Units Sold', 'date': 'Date'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Sales by region
        st.subheader("🌍 Sales by Region")
        
        regional_sales = cleaned_df.groupby('region')['units_sold'].sum().reset_index()
        
        fig = px.bar(
            regional_sales,
            x='region',
            y='units_sold',
            title='Total Sales by Region',
            labels={'units_sold': 'Units Sold', 'region': 'Region'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Sales by category
        st.subheader("📦 Sales by Category")
        
        category_sales = cleaned_df.groupby('category')['units_sold'].sum().reset_index()
        
        fig = px.pie(
            category_sales,
            values='units_sold',
            names='category',
            title='Sales Distribution by Category'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def create_forecast_tab(self, test_df, cleaned_df):
        """Create forecast visualization tab with interactive levels"""
        st.subheader("🔮 Sales Forecasting & Analysis")
        
        # Analysis Level Selector
        analysis_level = st.selectbox(
            "Select Analysis Level",
            ["Global Forecast (Prophet)", "Category Analysis", "Regional Analysis", "Product (SKU) Analysis"]
        )
        
        if analysis_level == "Global Forecast (Prophet)":
            if self.prophet_model is None:
                st.warning("⚠️ Prophet model not loaded. Please check model files.")
                return
            
            # Forecast Logic
            daily_sales = cleaned_df.groupby('date')['units_sold'].sum().reset_index()
            daily_sales.columns = ['ds', 'y']
            
            future_dates = pd.date_range(
                start=daily_sales['ds'].max() + pd.Timedelta(days=1),
                periods=30,
                freq='D'
            )
            future_df = pd.DataFrame({'ds': future_dates})
            
            with st.spinner("Generating Global Forecast..."):
                forecast = self.prophet_model.predict(future_df)
            
            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily_sales['ds'], y=daily_sales['y'], mode='lines', name='Historical', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='red', dash='dash')))
            fig.add_trace(go.Scatter(
                x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                fill='toself', fillcolor='rgba(255,0,0,0.2)', line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            ))
            fig.update_layout(title='Global Sales Forecast (Next 30 Days)', xaxis_title='Date', yaxis_title='Units Sold', height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Forecast", f"{forecast['yhat'].mean():.0f}")
            c2.metric("Peak Forecast", f"{forecast['yhat'].max():.0f}")
            c3.metric("Min Forecast", f"{forecast['yhat'].min():.0f}")

        else:
            # Granular Analysis
            target_col = {
                "Category Analysis": "category",
                "Regional Analysis": "region",
                "Product (SKU) Analysis": "sku"
            }[analysis_level]
            
            options = sorted(cleaned_df[target_col].unique())
            selection = st.selectbox(f"Select {target_col.capitalize()}", options)
            
            # Filter Data
            subset = cleaned_df[cleaned_df[target_col] == selection]
            daily_subset = subset.groupby('date')['units_sold'].sum().reset_index()
            
            # Plot Historical
            fig = px.line(daily_subset, x='date', y='units_sold', title=f'Sales Trend: {selection}')
            
            # Add Trend Line (Simple Moving Average)
            daily_subset['MA7'] = daily_subset['units_sold'].rolling(window=7).mean()
            fig.add_trace(go.Scatter(x=daily_subset['date'], y=daily_subset['MA7'], mode='lines', name='7-Day Moving Avg', line=dict(color='orange')))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Simple Naive Forecast (Last 30 days average projected)
            avg_last_30 = daily_subset.tail(30)['units_sold'].mean()
            st.info(f"💡 Naive Projection (based on last 30 days): Expect ~{avg_last_30:.0f} units/day.")

        # XGBoost Output (Global)
        if self.xgboost_model is not None:
             st.markdown("---")
             st.subheader("🎯 Global Drivers (XGBoost Feature Importance)")
             try:
                 feature_importance = self.xgboost_model.feature_importances_
                 feature_names = getattr(self.xgboost_model, 'feature_names_in_', [f'Feature_{i}' for i in range(len(feature_importance))])
                 importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance}).sort_values('importance', ascending=False).head(10)
                 fig = px.bar(importance_df, x='importance', y='feature', orientation='h', title='Key Sales Drivers')
                 st.plotly_chart(fig, use_container_width=True)
             except Exception as e:
                 st.info(f"Feature importance details unavailable: {e}")
    
    
    def create_query_tab(self):
        """Create business query tab"""
        st.subheader("🤖 AI-Powered Business Queries")
        
        if self.rag_pipeline is None:
            st.error("❌ RAG pipeline not currently available. Please check logs for setup errors.")
            return
        
        # Sample queries
        sample_queries = [
            "What were the total sales in 2023?",
            "Which product had the highest sales?",
            "How did promotions affect sales performance?",
            "What are the sales trends by region?",
            "Which month had the highest sales in 2024?",
            "What is the average price per unit?",
            "How many brands are in the dataset?",
            "What caused the sales dip in Q2 2023?",
            "Which region performed best?",
            "What is the impact of stock availability on sales?"
        ]
        
        # Query input
        st.subheader("💭 Ask a Question")
        
        # Quick query buttons
        st.write("**Quick Questions:**")
        cols = st.columns(2)
        for i, query in enumerate(sample_queries[:6]):
            with cols[i % 2]:
                if st.button(query, key=f"btn_{i}"):
                    st.session_state.user_query = query
        
        # Custom query input
        user_query = st.text_input(
            "Or type your own question:",
            value=st.session_state.get('user_query', ''),
            placeholder="e.g., What were the total sales in 2023?"
        )
        
        if st.button("🔍 Ask Question") and user_query:
            with st.spinner("🤔 Thinking..."):
                try:
                    result = self.rag_pipeline.answer_query(user_query)
                    # Display answer
                    st.subheader("💡 Answer")
                    st.write(result['answer'])
                    # Display sources
                    if result['sources']:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(result['sources'], 1):
                                st.write(f"**Source {i}:** {source}")
                except Exception as e:
                    st.error(f"❌ Error processing query: {e}")
        
        # Query history
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        
        if user_query and user_query not in st.session_state.query_history:
            st.session_state.query_history.append(user_query)
        
        if st.session_state.query_history:
            st.subheader("📝 Recent Questions")
            for query in st.session_state.query_history[-5:]:
                st.write(f"• {query}")
    
    def run_dashboard(self):
        """Run the main dashboard"""
        st.set_page_config(
            page_title="FMCG Sales Analytics & Forecasting",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Header
        st.title("📊 FMCG Sales Analytics & Forecasting")
        st.markdown("---")
        
        # Sidebar
        st.sidebar.title("🔧 Dashboard Controls")
        
        # Check status in sidebar
        st.sidebar.subheader("System Status")
        model_status = "✅ Loaded" if self.prophet_model else "❌ Not Loaded"
        st.sidebar.write(f"Models: {model_status}")
        
        rag_status = "✅ Active" if self.rag_pipeline else "❌ Inactive"
        st.sidebar.write(f"RAG Pipeline: {rag_status}")

        if st.sidebar.button("🔄 Reload Resources"):
             st.cache_resource.clear()
             st.rerun()
        
        # Load data
        test_df, cleaned_df = self.load_data()
        
        if cleaned_df is None:
            st.error("❌ Could not load data. Please ensure data preprocessing has been completed.")
            return
        
        # Main tabs
        tab1, tab2, tab3 = st.tabs([
            "📊 Sales Overview", 
            "🔮 Forecasting", 
            "🤖 Business Queries"
        ])

        with tab1:
            self.create_sales_overview(cleaned_df)

        with tab2:
            self.create_forecast_tab(test_df, cleaned_df)

        with tab3:
            self.create_query_tab()
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: gray;'>
                <p>FMCG Sales Analytics & Forecasting Dashboard | Powered by ML & Generative AI</p>
            </div>
            """,
            unsafe_allow_html=True
        )

def main():
    """Main function to run the dashboard"""
    dashboard = FMCGDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main() 
    