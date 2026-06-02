"""
Enhanced Streamlit Dashboard for FMCG Sales Analytics & Forecasting
2-Page Layout: Dashboard & Forecasting | AI Q&A Portal
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
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Add project root to sys.path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Visualization imports
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ML and RAG imports
from prophet import Prophet
import shap

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="FMCG Analytics & Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .insight-box {
        background: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        border-radius: 5px;
        margin: 1rem 0;
        color: #1f2937;
    }
    .insight-box h4 {
        color: #111827;
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    .insight-box p {
        color: #1f2937;
        margin: 0.3rem 0;
    }
    .insight-box ol, .insight-box ul {
        color: #1f2937;
        margin: 0.3rem 0 0.3rem 1.2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Caching functions
@st.cache_resource
def get_models(config):
    """Load trained models with caching"""
    models = {}
    try:
        prophet_path = Path(config['models']['prophet_model'])
        if prophet_path.exists():
            with open(prophet_path, 'rb') as f:
                models['prophet'] = joblib.load(f)
        
        xgboost_path = Path(config['models']['xgboost_model'])
        if xgboost_path.exists():
            with open(xgboost_path, 'rb') as f:
                models['xgboost'] = joblib.load(f)
        
        return models
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return None

@st.cache_resource
def get_rag_pipeline():
    """Load and setup RAG pipeline with caching"""
    try:
        from src.rag_pipeline import FMCGRAGPipeline
        
        config_path = project_root / "config.yaml"
        if not config_path.exists():
            logger.error(f"Config file not found at {config_path}")
            return None
        
        logger.info(f"Initializing RAG pipeline with config: {config_path}")
        rag_pipeline = FMCGRAGPipeline(config_path=str(config_path))
        
        vector_store_path = project_root / "vector_store" / "faiss_index.bin"
        if not vector_store_path.exists():
            logger.warning(f"Vector store not found at {vector_store_path}")
            return None
        
        if rag_pipeline.load_vector_store():
            logger.info("RAG pipeline loaded successfully")
            return rag_pipeline
        else:
            logger.error("Failed to load vector store")
            return None
            
    except Exception as e:
        logger.error(f"Error setting up RAG pipeline: {e}", exc_info=True)
        return None

@st.cache_data
def load_data(processed_dir):
    """Load processed data with caching"""
    try:
        test_path = processed_dir / "test_features.csv"
        cleaned_path = processed_dir / "cleaned.csv"
        
        if not test_path.exists() or not cleaned_path.exists():
            return None, None
        
        test_df = pd.read_csv(test_path)
        test_df['date'] = pd.to_datetime(test_df['date'])
        
        cleaned_df = pd.read_csv(cleaned_path)
        cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])
        
        return test_df, cleaned_df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None

class EnhancedFMCGDashboard:
    def __init__(self, config_path="config.yaml"):
        """Initialize the enhanced dashboard"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as file:
                    self.config = yaml.safe_load(file)
            else:
                root_config = project_root / config_path
                if root_config.exists():
                    with open(root_config, 'r') as file:
                        self.config = yaml.safe_load(file)
                else:
                    st.error(f"Config file not found")
                    st.stop()
        except Exception as e:
            st.error(f"Error loading config: {e}")
            st.stop()
        
        self.processed_dir = Path(self.config['data']['processed_dir'])
        if not self.processed_dir.is_absolute():
            self.processed_dir = project_root / self.processed_dir
        
        self.models = get_models(self.config)
        self.prophet_model = self.models.get('prophet') if self.models else None
        self.xgboost_model = self.models.get('xgboost') if self.models else None
        self.rag_pipeline = get_rag_pipeline()
    
    def create_kpi_cards(self, cleaned_df):
        """Create enhanced KPI cards with insights"""
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate metrics
        total_sales = cleaned_df['units_sold'].sum()
        total_revenue = (cleaned_df['units_sold'] * cleaned_df['price_unit']).sum()
        avg_price = cleaned_df['price_unit'].mean()
        unique_products = cleaned_df['sku'].nunique()
        
        # Calculate growth (compare last 30 days vs previous 30 days)
        latest_date = cleaned_df['date'].max()
        last_30 = cleaned_df[cleaned_df['date'] > (latest_date - pd.Timedelta(days=30))]['units_sold'].sum()
        prev_30 = cleaned_df[(cleaned_df['date'] <= (latest_date - pd.Timedelta(days=30))) & 
                             (cleaned_df['date'] > (latest_date - pd.Timedelta(days=60)))]['units_sold'].sum()
        growth = ((last_30 - prev_30) / prev_30 * 100) if prev_30 > 0 else 0
        
        with col1:
            st.metric(
                label="📦 Total Sales Volume",
                value=f"{total_sales:,.0f} units",
                delta=f"{growth:.1f}% vs prev 30d"
            )
        
        with col2:
            st.metric(
                label="💰 Total Revenue",
                value=f"${total_revenue:,.0f}",
                delta=f"${total_revenue/total_sales:.2f} per unit"
            )
        
        with col3:
            st.metric(
                label="💵 Average Price",
                value=f"${avg_price:.2f}",
                delta=f"{cleaned_df['promotion_flag'].sum():,} promotions"
            )
        
        with col4:
            st.metric(
                label="🏷️ Product Portfolio",
                value=f"{unique_products} SKUs",
                delta=f"{cleaned_df['brand'].nunique()} brands"
            )
    
    def create_insights_section(self, cleaned_df):
        """Generate AI-powered insights"""
        st.markdown("### 🔍 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top performing products — render the whole card as one HTML block
            # so the list sits inside the styled .insight-box div.
            top_products = cleaned_df.groupby('sku')['units_sold'].sum().nlargest(3)
            top_list_html = "".join(
                f"<li><strong>{sku}</strong>: {sales:,.0f} units</li>"
                for sku, sales in top_products.items()
            )
            st.markdown(
                f"""
                <div class="insight-box">
                    <h4>🏆 Top Performers</h4>
                    <ol>{top_list_html}</ol>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            # Promotion effectiveness — guard against datasets with no
            # promotion-flagged rows (current dataset has 0 promotions),
            # which would otherwise produce a NaN-filled blank card.
            promo_rows = cleaned_df[cleaned_df['promotion_flag'] == 1]
            regular_rows = cleaned_df[cleaned_df['promotion_flag'] == 0]

            if len(promo_rows) == 0 or len(regular_rows) == 0:
                promo_html = (
                    "<p>No promotion-flagged transactions in the loaded dataset. "
                    "Promotion impact metrics will populate once promotional "
                    "periods are tagged in the source data.</p>"
                )
            else:
                promo_sales = promo_rows['units_sold'].mean()
                regular_sales = regular_rows['units_sold'].mean()
                promo_lift = (
                    ((promo_sales - regular_sales) / regular_sales * 100)
                    if regular_sales > 0 else 0
                )
                promo_html = (
                    f"<p><strong>Sales Lift:</strong> {promo_lift:.1f}%</p>"
                    f"<p><strong>Promo Avg:</strong> {promo_sales:.0f} units/day</p>"
                    f"<p><strong>Regular Avg:</strong> {regular_sales:.0f} units/day</p>"
                )

            st.markdown(
                f"""
                <div class="insight-box">
                    <h4>🎯 Promotion Impact</h4>
                    {promo_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        # Seasonal patterns
        cleaned_df['month'] = cleaned_df['date'].dt.month
        monthly_avg = cleaned_df.groupby('month')['units_sold'].mean()
        peak_month = monthly_avg.idxmax()
        low_month = monthly_avg.idxmin()
        
        month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                      7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        
        st.info(f"📅 **Seasonal Pattern**: Peak sales in **{month_names[peak_month]}** ({monthly_avg[peak_month]:.0f} units/day), lowest in **{month_names[low_month]}** ({monthly_avg[low_month]:.0f} units/day)")
    
    def create_advanced_visualizations(self, cleaned_df):
        """Create advanced interactive visualizations"""
        
        # Sales trend with multiple metrics
        st.markdown("### 📈 Sales Performance Trends")
        
        daily_sales = cleaned_df.groupby('date').agg({
            'units_sold': 'sum',
            'price_unit': 'mean',
            'promotion_flag': 'sum'
        }).reset_index()
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Daily Sales Volume & Price Trends', 'Promotion Activity'),
            row_heights=[0.7, 0.3],
            vertical_spacing=0.1,
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )
        
        # Sales volume
        fig.add_trace(
            go.Scatter(x=daily_sales['date'], y=daily_sales['units_sold'],
                      name='Sales Volume', line=dict(color='#1f77b4', width=2)),
            row=1, col=1, secondary_y=False
        )
        
        # Moving average
        daily_sales['MA30'] = daily_sales['units_sold'].rolling(window=30).mean()
        fig.add_trace(
            go.Scatter(x=daily_sales['date'], y=daily_sales['MA30'],
                      name='30-Day MA', line=dict(color='#ff7f0e', width=2, dash='dash')),
            row=1, col=1, secondary_y=False
        )
        
        # Average price
        fig.add_trace(
            go.Scatter(x=daily_sales['date'], y=daily_sales['price_unit'],
                      name='Avg Price', line=dict(color='#2ca02c', width=1)),
            row=1, col=1, secondary_y=True
        )
        
        # Promotions
        fig.add_trace(
            go.Bar(x=daily_sales['date'], y=daily_sales['promotion_flag'],
                  name='Promotions', marker_color='#d62728'),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Units Sold", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Avg Price ($)", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="# Promotions", row=2, col=1)
        
        fig.update_layout(height=600, showlegend=True, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        # Regional and category breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🌍 Regional Performance")
            regional_sales = cleaned_df.groupby('region').agg({
                'units_sold': 'sum',
                'price_unit': 'mean'
            }).reset_index()
            regional_sales['revenue'] = regional_sales['units_sold'] * regional_sales['price_unit']
            
            fig = px.bar(regional_sales, x='region', y='units_sold',
                        color='revenue', color_continuous_scale='Blues',
                        title='Sales Volume by Region',
                        labels={'units_sold': 'Units Sold', 'revenue': 'Revenue ($)'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📦 Category Distribution")
            category_sales = cleaned_df.groupby('category')['units_sold'].sum().reset_index()
            fig = px.pie(category_sales, values='units_sold', names='category',
                        title='Sales Distribution by Category',
                        hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)
    
    def create_enhanced_forecast(self, cleaned_df):
        """Create enhanced forecasting with multiple scenarios"""
        st.markdown("### 🔮 Advanced Sales Forecasting")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            forecast_days = st.slider("Forecast Horizon (days)", 7, 90, 30)
            confidence_level = st.select_slider("Confidence Level", 
                                               options=[80, 85, 90, 95], 
                                               value=95)
            
            include_scenarios = st.checkbox("Show Scenarios (Best/Worst Case)", value=True)
        
        with col1:
            if self.prophet_model is None:
                st.warning("⚠️ Prophet model not loaded")
                return
            
            # Prepare data
            daily_sales = cleaned_df.groupby('date')['units_sold'].sum().reset_index()
            daily_sales.columns = ['ds', 'y']
            
            # Generate forecast
            future_dates = pd.date_range(
                start=daily_sales['ds'].max() + pd.Timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            future_df = pd.DataFrame({'ds': future_dates})
            
            with st.spinner("🔄 Generating forecast..."):
                forecast = self.prophet_model.predict(future_df)
            
            # Create visualization
            fig = go.Figure()
            
            # Historical data (last 90 days for context)
            recent_history = daily_sales.tail(90)
            fig.add_trace(go.Scatter(
                x=recent_history['ds'], y=recent_history['y'],
                mode='lines', name='Historical',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat'],
                mode='lines', name='Forecast',
                line=dict(color='#ff7f0e', width=3, dash='dash')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                fill='toself', fillcolor='rgba(255,127,14,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{confidence_level}% Confidence'
            ))
            
            # Scenarios
            if include_scenarios:
                # Best case: +20% from upper bound
                best_case = forecast['yhat_upper'] * 1.2
                fig.add_trace(go.Scatter(
                    x=forecast['ds'], y=best_case,
                    mode='lines', name='Best Case (+20%)',
                    line=dict(color='green', width=1, dash='dot')
                ))
                
                # Worst case: -20% from lower bound
                worst_case = forecast['yhat_lower'] * 0.8
                fig.add_trace(go.Scatter(
                    x=forecast['ds'], y=worst_case,
                    mode='lines', name='Worst Case (-20%)',
                    line=dict(color='red', width=1, dash='dot')
                ))
            
            fig.update_layout(
                title=f'{forecast_days}-Day Sales Forecast',
                xaxis_title='Date',
                yaxis_title='Units Sold',
                height=500,
                hovermode='x unified',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Forecast metrics
        st.markdown("### 📊 Forecast Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        total_forecast = forecast['yhat'].sum()
        avg_daily = forecast['yhat'].mean()
        peak_day = forecast.loc[forecast['yhat'].idxmax(), 'ds']
        low_day = forecast.loc[forecast['yhat'].idxmin(), 'ds']
        
        with col1:
            st.metric("Total Forecast", f"{total_forecast:,.0f} units")
        with col2:
            st.metric("Daily Average", f"{avg_daily:,.0f} units")
        with col3:
            st.metric("Peak Day", peak_day.strftime('%b %d'))
        with col4:
            st.metric("Low Day", low_day.strftime('%b %d'))
        
        # Trend analysis
        trend_change = ((forecast['yhat'].iloc[-1] - forecast['yhat'].iloc[0]) / forecast['yhat'].iloc[0] * 100)
        if trend_change > 5:
            st.success(f"📈 **Growing Trend**: Forecast shows {trend_change:.1f}% increase over the period")
        elif trend_change < -5:
            st.warning(f"📉 **Declining Trend**: Forecast shows {trend_change:.1f}% decrease over the period")
        else:
            st.info(f"➡️ **Stable Trend**: Forecast shows relatively stable sales ({trend_change:.1f}% change)")
        
        # Decomposition insights
        if hasattr(self.prophet_model, 'seasonalities'):
            st.markdown("### 🔄 Seasonality Components")
            
            # Show trend, weekly, yearly patterns
            components_fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Trend', 'Weekly Pattern', 'Overall Seasonality')
            )
            
            # This is a simplified view - actual Prophet components would be more detailed
            st.info("💡 **Insight**: The forecast incorporates weekly and seasonal patterns learned from historical data")
    
    def create_feature_importance(self):
        """Show XGBoost feature importance with insights"""
        if self.xgboost_model is None:
            return
        
        st.markdown("### 🎯 Sales Drivers Analysis")
        
        try:
            feature_importance = self.xgboost_model.feature_importances_
            feature_names = getattr(self.xgboost_model, 'feature_names_in_', 
                                   [f'Feature_{i}' for i in range(len(feature_importance))])
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False).head(15)
            
            # Clean feature names for display
            importance_df['feature_clean'] = importance_df['feature'].str.replace('_', ' ').str.title()
            
            fig = px.bar(importance_df, x='importance', y='feature_clean',
                        orientation='h',
                        title='Top 15 Sales Drivers',
                        labels={'importance': 'Importance Score', 'feature_clean': 'Feature'},
                        color='importance',
                        color_continuous_scale='Viridis')
            
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Insights
            top_3 = importance_df.head(3)['feature_clean'].tolist()
            st.info(f"💡 **Key Drivers**: The top 3 factors influencing sales are: **{', '.join(top_3)}**")
            
        except Exception as e:
            st.warning(f"Feature importance not available: {e}")
    
    def create_qa_portal(self):
        """Create AI Q&A portal"""
        st.markdown('<p class="main-header">🤖 AI-Powered Q&A Portal</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Ask questions about your FMCG sales data in natural language</p>', unsafe_allow_html=True)
        
        if self.rag_pipeline is None:
            st.error("""
            ❌ **RAG Pipeline Not Available**
            
            The AI Q&A system requires the RAG pipeline to be initialized. 
            
            **To fix this:**
            1. Run: `python run_pipeline.py` to create the vector store
            2. Restart the dashboard
            
            Check the logs for more details.
            """)
            return
        
        # Sample queries organized by category
        st.markdown("### 💭 Quick Questions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📊 Sales Performance**")
            if st.button("What were total sales in 2023?", key="q1"):
                st.session_state.user_query = "What were the total sales in 2023?"
            if st.button("Which product sold the most?", key="q2"):
                st.session_state.user_query = "Which product had the highest sales?"
            if st.button("Best performing region?", key="q3"):
                st.session_state.user_query = "Which region performed best?"
        
        with col2:
            st.markdown("**🎯 Promotions & Pricing**")
            if st.button("Impact of promotions?", key="q4"):
                st.session_state.user_query = "How did promotions affect sales performance?"
            if st.button("Average price per unit?", key="q5"):
                st.session_state.user_query = "What is the average price per unit?"
            if st.button("Price vs sales correlation?", key="q6"):
                st.session_state.user_query = "How does pricing affect sales volume?"
        
        with col3:
            st.markdown("**📅 Trends & Patterns**")
            if st.button("Seasonal patterns?", key="q7"):
                st.session_state.user_query = "What are the seasonal sales patterns?"
            if st.button("Peak sales month?", key="q8"):
                st.session_state.user_query = "Which month had the highest sales in 2024?"
            if st.button("Stock availability impact?", key="q9"):
                st.session_state.user_query = "What is the impact of stock availability on sales?"
        
        # Custom query input
        st.markdown("### ✍️ Ask Your Own Question")
        user_query = st.text_area(
            "Type your question here:",
            value=st.session_state.get('user_query', ''),
            placeholder="e.g., What were the sales trends for beverages in the North region during Q2 2024?",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("🔍 Ask Question", type="primary", use_container_width=True)
        with col2:
            if st.button("🔄 Clear", use_container_width=True):
                st.session_state.user_query = ''
                st.rerun()
        
        if ask_button and user_query:
            with st.spinner("🤔 Analyzing your question..."):
                try:
                    result = self.rag_pipeline.answer_query(user_query)
                    
                    # Display answer in a nice format
                    st.markdown("---")
                    st.markdown("### 💡 Answer")
                    st.markdown(f"""
                    <div style="background: #f0f7ff; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #1f77b4;">
                        <p style="font-size: 1.1rem; color: #333; margin: 0;">{result['answer']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display sources
                    if result['sources']:
                        with st.expander("📚 View Sources & Context", expanded=False):
                            st.markdown("**Retrieved Information:**")
                            for i, source in enumerate(result['sources'][:3], 1):
                                st.markdown(f"{i}. {source}")
                    
                    # Add to history
                    if 'query_history' not in st.session_state:
                        st.session_state.query_history = []
                    
                    if user_query not in [q['question'] for q in st.session_state.query_history]:
                        st.session_state.query_history.insert(0, {
                            'question': user_query,
                            'answer': result['answer'],
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
                        })
                        # Keep only last 10
                        st.session_state.query_history = st.session_state.query_history[:10]
                    
                except Exception as e:
                    st.error(f"❌ Error processing query: {e}")
        
        # Query history
        if 'query_history' in st.session_state and st.session_state.query_history:
            st.markdown("---")
            st.markdown("### 📝 Recent Questions")
            for item in st.session_state.query_history[:5]:
                with st.expander(f"❓ {item['question']}", expanded=False):
                    st.markdown(f"**Answer:** {item['answer']}")
                    st.caption(f"Asked on: {item['timestamp']}")
    
    def run_dashboard(self):
        """Run the main dashboard"""
        
        # Sidebar
        with st.sidebar:
            st.image("https://img.icons8.com/clouds/200/000000/business-report.png", width=150)
            st.markdown("## 🎛️ Dashboard Controls")
            
            # System status
            st.markdown("### System Status")
            model_status = "✅ Loaded" if self.prophet_model else "❌ Not Loaded"
            st.write(f"**Models:** {model_status}")
            
            rag_status = "✅ Active" if self.rag_pipeline else "❌ Inactive"
            st.write(f"**AI Q&A:** {rag_status}")
            
            st.markdown("---")
            
            # Navigation
            page = st.radio(
                "📍 Navigate",
                ["📊 Dashboard & Forecasting", "🤖 AI Q&A Portal"],
                label_visibility="collapsed"
            )
            
            st.markdown("---")
            
            # Info
            st.markdown("### ℹ️ About")
            st.info("""
            **FMCG Analytics Platform**
            
            Powered by:
            - Prophet (Forecasting)
            - XGBoost (ML)
            - RAG (AI Q&A)
            
            Data: 2022-2024
            """)
            
            if st.button("🔄 Refresh Data"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
        
        # Load data
        test_df, cleaned_df = load_data(self.processed_dir)
        
        if cleaned_df is None:
            st.error("❌ Could not load data. Please ensure data preprocessing has been completed.")
            return
        
        # Main content based on page selection
        if page == "📊 Dashboard & Forecasting":
            # Header
            st.markdown('<p class="main-header">📊 FMCG Sales Analytics & Forecasting</p>', unsafe_allow_html=True)
            st.markdown('<p class="sub-header">Real-time insights and AI-powered forecasts for your FMCG business</p>', unsafe_allow_html=True)
            
            # KPIs
            self.create_kpi_cards(cleaned_df)
            
            st.markdown("---")
            
            # Insights
            self.create_insights_section(cleaned_df)
            
            st.markdown("---")
            
            # Visualizations
            self.create_advanced_visualizations(cleaned_df)
            
            st.markdown("---")
            
            # Forecasting
            self.create_enhanced_forecast(cleaned_df)
            
            st.markdown("---")
            
            # Feature importance
            self.create_feature_importance()
            
        else:  # AI Q&A Portal
            self.create_qa_portal()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem 0;'>
            <p><strong>FMCG Sales Analytics & Forecasting Platform</strong></p>
            <p>Powered by Machine Learning & Generative AI | © 2024</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main function to run the dashboard"""
    dashboard = EnhancedFMCGDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
