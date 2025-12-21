"""
SHAP Explainability Module for FMCG Sales Data
Provides global and local explanations for XGBoost model predictions
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# SHAP and visualization imports
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FMCGExplainability:
    def __init__(self, config_path="config.yaml"):
        """Initialize the explainability module with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.processed_dir = Path(self.config['data']['processed_dir'])
        self.models_dir = Path(self.config['models']['xgboost_model']).parent
        self.reports_dir = Path(self.config['reports']['shap_dir'])
        self.xgboost_model_path = self.config['models']['xgboost_model']
        self.target_column = self.config['features']['target_column']
        
        # Create reports directory if it doesn't exist
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SHAP explainer
        self.explainer = None
        self.shap_values = None
        
    def load_model_and_data(self):
        """Load XGBoost model and test data"""
        logger.info("Loading XGBoost model and test data...")
        
        # Load XGBoost model
        with open(self.xgboost_model_path, 'rb') as f:
            self.model = joblib.load(f)
        
        # Load test data
        test_df = pd.read_csv(self.processed_dir / "test_features.csv")
        test_df['date'] = pd.to_datetime(test_df['date'])
        
        logger.info(f"Loaded test data: {test_df.shape}")
        return test_df
    
    def prepare_data_for_shap(self, df):
        """Prepare data for SHAP analysis"""
        logger.info("Preparing data for SHAP analysis...")
        
        # Select features for SHAP (exclude date and target)
        exclude_cols = ['date', self.target_column]
        feature_cols = [col for col in df.columns if col not in exclude_cols and not col.startswith('holiday_name')]
        
        # Handle categorical columns
        categorical_cols = [col for col in feature_cols if df[col].dtype == 'object']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category').cat.codes
        
        X = df[feature_cols].fillna(0)
        y = df[self.target_column]
        
        logger.info(f"SHAP data shape: X={X.shape}, y={y.shape}")
        return X, y, feature_cols
    
    def create_shap_explainer(self, X):
        """Create SHAP explainer for the model"""
        logger.info("Creating SHAP explainer...")
        
        # Create TreeExplainer for XGBoost
        self.explainer = shap.TreeExplainer(self.model)
        
        # Calculate SHAP values (use a sample for efficiency)
        sample_size = min(1000, len(X))
        X_sample = X.sample(n=sample_size, random_state=42)
        
        logger.info(f"Calculating SHAP values for {sample_size} samples...")
        self.shap_values = self.explainer.shap_values(X_sample)
        
        return X_sample
    
    def create_global_explanations(self, X_sample, feature_cols):
        """Create global SHAP explanations"""
        logger.info("Creating global SHAP explanations...")
        
        # 1. Summary Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values, 
            X_sample,
            feature_names=feature_cols,
            show=False,
            plot_size=(12, 8)
        )
        plt.title('SHAP Summary Plot - Global Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.reports_dir / 'shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Bar Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values, 
            X_sample,
            feature_names=feature_cols,
            plot_type="bar",
            show=False
        )
        plt.title('SHAP Feature Importance (Bar Plot)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.reports_dir / 'shap_bar_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Feature Importance DataFrame
        feature_importance = np.abs(self.shap_values).mean(0)
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        importance_df.to_csv(self.reports_dir / 'shap_feature_importance.csv', index=False)
        
        logger.info("Global SHAP explanations created")
        return importance_df
    
    def create_local_explanations(self, X_sample, feature_cols, num_samples=5):
        """Create local SHAP explanations for individual predictions"""
        logger.info("Creating local SHAP explanations...")
        
        # Select random samples for local explanations
        sample_indices = np.random.choice(len(X_sample), num_samples, replace=False)
        
        for i, idx in enumerate(sample_indices):
            # Create waterfall plot for individual prediction
            plt.figure(figsize=(12, 8))
            shap.waterfall_plot(
                shap.Explanation(
                    values=self.shap_values[idx],
                    base_values=self.explainer.expected_value,
                    data=X_sample.iloc[idx],
                    feature_names=feature_cols
                ),
                show=False
            )
            plt.title(f'Local SHAP Explanation - Sample {i+1}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.reports_dir / f'shap_waterfall_sample_{i+1}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create force plot
            plt.figure(figsize=(12, 6))
            shap.force_plot(
                self.explainer.expected_value,
                self.shap_values[idx],
                X_sample.iloc[idx],
                feature_names=feature_cols,
                show=False
            )
            plt.title(f'SHAP Force Plot - Sample {i+1}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.reports_dir / f'shap_force_sample_{i+1}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Local SHAP explanations created for {num_samples} samples")
    
    def create_interaction_plots(self, X_sample, feature_cols):
        """Create SHAP interaction plots"""
        logger.info("Creating SHAP interaction plots...")
        
        # Get top features for interaction analysis
        top_features = X_sample.columns[:10].tolist()
        
        # Create interaction plots for top feature pairs
        for i, feature1 in enumerate(top_features[:5]):
            for feature2 in top_features[i+1:6]:
                try:
                    plt.figure(figsize=(10, 6))
                    shap.dependence_plot(
                        feature1, 
                        self.shap_values, 
                        X_sample,
                        interaction_index=feature2,
                        show=False
                    )
                    plt.title(f'SHAP Dependence Plot: {feature1} vs {feature2}', fontsize=12, fontweight='bold')
                    plt.tight_layout()
                    plt.savefig(self.reports_dir / f'shap_dependence_{feature1}_vs_{feature2}.png', dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    logger.warning(f"Could not create interaction plot for {feature1} vs {feature2}: {e}")
        
        logger.info("SHAP interaction plots created")
    
    def create_interactive_shap_plots(self, X_sample, feature_cols):
        """Create interactive SHAP plots using Plotly"""
        logger.info("Creating interactive SHAP plots...")
        
        # 1. Interactive Summary Plot
        feature_importance = np.abs(self.shap_values).mean(0)
        top_features = feature_cols[:15]  # Top 15 features
        
        fig_summary = go.Figure()
        
        for i, feature in enumerate(top_features):
            feature_idx = feature_cols.index(feature)
            shap_values_feature = self.shap_values[:, feature_idx]
            
            fig_summary.add_trace(go.Scatter(
                x=X_sample[feature],
                y=shap_values_feature,
                mode='markers',
                name=feature,
                opacity=0.6,
                marker=dict(size=4)
            ))
        
        fig_summary.update_layout(
            title='Interactive SHAP Summary Plot',
            xaxis_title='Feature Values',
            yaxis_title='SHAP Values',
            height=600
        )
        
        fig_summary.write_html(self.reports_dir / 'interactive_shap_summary.html')
        
        # 2. Feature Importance Bar Chart
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': feature_importance
        }).sort_values('importance', ascending=True).tail(15)
        
        fig_importance = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title='SHAP Feature Importance (Interactive)'
        )
        
        fig_importance.update_layout(
            xaxis_title='Mean |SHAP Value|',
            yaxis_title='Features'
        )
        
        fig_importance.write_html(self.reports_dir / 'interactive_shap_importance.html')
        
        # 3. SHAP Values Distribution
        fig_dist = go.Figure()
        
        for feature in top_features[:10]:
            feature_idx = feature_cols.index(feature)
            shap_values_feature = self.shap_values[:, feature_idx]
            
            fig_dist.add_trace(go.Box(
                y=shap_values_feature,
                name=feature,
                boxpoints='outliers'
            ))
        
        fig_dist.update_layout(
            title='SHAP Values Distribution by Feature',
            yaxis_title='SHAP Values',
            xaxis_title='Features',
            height=600
        )
        
        fig_dist.write_html(self.reports_dir / 'interactive_shap_distribution.html')
        
        logger.info("Interactive SHAP plots created")
    
    def generate_business_insights(self, importance_df, X_sample, feature_cols):
        """Generate business insights from SHAP analysis"""
        logger.info("Generating business insights...")
        
        insights = {
            'top_drivers': [],
            'feature_analysis': {},
            'recommendations': []
        }
        
        # Top drivers analysis
        top_features = importance_df.head(10)
        insights['top_drivers'] = top_features['feature'].tolist()
        
        # Feature analysis
        for _, row in top_features.iterrows():
            feature = row['feature']
            importance = row['importance']
            
            feature_values = X_sample[feature]
            feature_shap = self.shap_values[:, feature_cols.index(feature)]
            
            # Analyze feature impact
            positive_impact = feature_shap[feature_shap > 0].mean()
            negative_impact = feature_shap[feature_shap < 0].mean()
            
            insights['feature_analysis'][feature] = {
                'importance': float(importance),
                'positive_impact': float(positive_impact),
                'negative_impact': float(negative_impact),
                'mean_value': float(feature_values.mean()),
                'std_value': float(feature_values.std())
            }
        
        # Generate recommendations
        recommendations = []
        
        # Price-related insights
        price_features = [f for f in top_features['feature'] if 'price' in f.lower()]
        if price_features:
            recommendations.append("Price optimization is crucial for sales performance")
        
        # Lag features insights
        lag_features = [f for f in top_features['feature'] if 'lag' in f.lower()]
        if lag_features:
            recommendations.append("Historical sales patterns significantly influence future sales")
        
        # Promotion insights
        promo_features = [f for f in top_features['feature'] if 'promotion' in f.lower()]
        if promo_features:
            recommendations.append("Promotional activities have strong impact on sales")
        
        # Seasonal insights
        seasonal_features = [f for f in top_features['feature'] if any(x in f.lower() for x in ['month', 'quarter', 'season'])]
        if seasonal_features:
            recommendations.append("Seasonal patterns are important for sales forecasting")
        
        insights['recommendations'] = recommendations
        
        # Save insights
        with open(self.reports_dir / 'business_insights.json', 'w') as f:
            import json
            json.dump(insights, f, indent=2, default=str)
        
        logger.info("Business insights generated")
        return insights
    
    def create_explainability_report(self, importance_df, insights):
        """Create comprehensive explainability report"""
        logger.info("Creating explainability report...")
        
        report = {
            'explainability_summary': {
                'analysis_date': pd.Timestamp.now().isoformat(),
                'total_features_analyzed': len(importance_df),
                'top_features_count': 10,
                'shap_values_calculated': len(self.shap_values)
            },
            'top_features': importance_df.head(10).to_dict('records'),
            'business_insights': insights,
            'model_interpretability': {
                'global_explanations': 'SHAP summary and bar plots generated',
                'local_explanations': 'Waterfall and force plots for individual predictions',
                'interaction_analysis': 'Feature interaction plots created',
                'business_recommendations': len(insights['recommendations'])
            }
        }
        
        # Save report
        with open(self.reports_dir / 'explainability_report.json', 'w') as f:
            import json
            json.dump(report, f, indent=2, default=str)
        
        logger.info("Explainability report saved")
        return report
    
    def run_explainability_analysis(self):
        """Run the complete SHAP explainability pipeline"""
        logger.info("Starting FMCG SHAP explainability pipeline...")
        
        try:
            # Load model and data
            test_df = self.load_model_and_data()
            
            # Prepare data for SHAP
            X, y, feature_cols = self.prepare_data_for_shap(test_df)
            
            # Create SHAP explainer
            X_sample = self.create_shap_explainer(X)
            
            # Create global explanations
            importance_df = self.create_global_explanations(X_sample, feature_cols)
            
            # Create local explanations
            self.create_local_explanations(X_sample, feature_cols)
            
            # Create interaction plots
            self.create_interaction_plots(X_sample, feature_cols)
            
            # Create interactive plots
            self.create_interactive_shap_plots(X_sample, feature_cols)
            
            # Generate business insights
            insights = self.generate_business_insights(importance_df, X_sample, feature_cols)
            
            # Create explainability report
            report = self.create_explainability_report(importance_df, insights)
            
            logger.info("SHAP explainability analysis completed successfully!")
            return {
                'importance_df': importance_df,
                'insights': insights,
                'report': report
            }
            
        except Exception as e:
            logger.error(f"Error in explainability pipeline: {e}")
            raise

def main():
    """Main function to run SHAP explainability analysis"""
    explainer = FMCGExplainability()
    results = explainer.run_explainability_analysis()
    
    print("\n" + "="*50)
    print("SHAP EXPLAINABILITY ANALYSIS COMPLETED")
    print("="*50)
    print("Top 10 Most Important Features:")
    for i, row in results['importance_df'].head(10).iterrows():
        print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    print(f"\nBusiness Insights Generated: {len(results['insights']['recommendations'])}")
    for i, rec in enumerate(results['insights']['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print(f"\nReports Generated:")
    print("  - SHAP Summary Plot")
    print("  - SHAP Bar Plot")
    print("  - Local Explanations (Waterfall & Force Plots)")
    print("  - Interactive Plots")
    print("  - Business Insights Report")
    print("="*50)

if __name__ == "__main__":
    main() 