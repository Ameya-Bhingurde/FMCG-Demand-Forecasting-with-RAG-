"""
Model Evaluation Module for FMCG Sales Data
Evaluates Prophet and XGBoost models with comprehensive metrics
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FMCGModelEvaluator:
    def __init__(self, config_path="config.yaml"):
        """Initialize the model evaluator with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.processed_dir = Path(self.config['data']['processed_dir'])
        self.models_dir = Path(self.config['models']['prophet_model']).parent
        self.reports_dir = Path(self.config['reports']['evaluation']).parent
        self.prophet_model_path = self.config['models']['prophet_model']
        self.xgboost_model_path = self.config['models']['xgboost_model']
        self.target_column = self.config['features']['target_column']
        
        # Create reports directory if it doesn't exist
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def load_models_and_data(self):
        """Load trained models and test data"""
        logger.info("Loading models and test data...")
        
        # Load models
        with open(self.prophet_model_path, 'rb') as f:
            prophet_model = joblib.load(f)
        
        with open(self.xgboost_model_path, 'rb') as f:
            xgboost_model = joblib.load(f)
        
        # Load test data
        test_df = pd.read_csv(self.processed_dir / "test_features.csv")
        test_df['date'] = pd.to_datetime(test_df['date'])
        
        logger.info(f"Loaded test data: {test_df.shape}")
        return prophet_model, xgboost_model, test_df
    
    def evaluate_prophet_model(self, model, test_data):
        """Comprehensive evaluation of Prophet model"""
        logger.info("Evaluating Prophet model...")
        
        # Prepare test data for Prophet
        daily_sales = test_data.groupby('date')[self.target_column].sum().reset_index()
        daily_sales.columns = ['ds', 'y']
        
        # Add regressors if available
        if 'price_unit' in test_data.columns:
            daily_price = test_data.groupby('date')['price_unit'].mean().reset_index()
            daily_price.columns = ['ds', 'price']
            daily_sales = daily_sales.merge(daily_price, on='ds', how='left')
        
        if 'promotion_flag' in test_data.columns:
            daily_promotion = test_data.groupby('date')['promotion_flag'].mean().reset_index()
            daily_promotion.columns = ['ds', 'promotion']
            daily_sales = daily_sales.merge(daily_promotion, on='ds', how='left')
        
        # Make predictions
        future = model.make_future_dataframe(periods=len(daily_sales))
        
        # Add regressors to future if they exist
        if 'price' in daily_sales.columns:
            future = future.merge(daily_sales[['ds', 'price']], on='ds', how='left')
        if 'promotion' in daily_sales.columns:
            future = future.merge(daily_sales[['ds', 'promotion']], on='ds', how='left')
        
        forecast = model.predict(future)
        
        # Extract predictions for test period
        test_forecast = forecast.tail(len(daily_sales))
        y_true = daily_sales['y'].values
        y_pred = test_forecast['yhat'].values
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, model_name="Prophet")
        
        # Create evaluation results
        results = {
            'metrics': metrics,
            'y_true': y_true,
            'y_pred': y_pred,
            'dates': daily_sales['ds'].values,
            'forecast': test_forecast
        }
        
        return results
    
    def evaluate_xgboost_model(self, model, test_data):
        """Comprehensive evaluation of XGBoost model"""
        logger.info("Evaluating XGBoost model...")
        
        # Prepare test data for XGBoost
        exclude_cols = ['date', self.target_column]
        feature_cols = [col for col in test_data.columns if col not in exclude_cols and not col.startswith('holiday_name')]
        
        # Handle categorical columns
        categorical_cols = [col for col in feature_cols if test_data[col].dtype == 'object']
        for col in categorical_cols:
            if col in test_data.columns:
                test_data[col] = test_data[col].astype('category').cat.codes
        
        X_test = test_data[feature_cols].fillna(0)
        y_test = test_data[self.target_column]
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test.values, y_pred, model_name="XGBoost")
        
        # Create evaluation results
        results = {
            'metrics': metrics,
            'y_true': y_test.values,
            'y_pred': y_pred,
            'dates': test_data['date'].values,
            'feature_importance': self.get_feature_importance(model, feature_cols)
        }
        
        return results
    
    def calculate_metrics(self, y_true, y_pred, model_name):
        """Calculate comprehensive evaluation metrics"""
        logger.info(f"Calculating metrics for {model_name}...")
        
        # Basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # Additional metrics
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        
        # Directional accuracy
        directional_accuracy = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100
        
        # Bias
        bias = np.mean(y_pred - y_true)
        
        # Variance explained
        variance_explained = 1 - (np.var(y_true - y_pred) / np.var(y_true))
        
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2),
            'mse': float(mse),
            'directional_accuracy': float(directional_accuracy),
            'bias': float(bias),
            'variance_explained': float(variance_explained),
            'mean_true': float(np.mean(y_true)),
            'mean_pred': float(np.mean(y_pred)),
            'std_true': float(np.std(y_true)),
            'std_pred': float(np.std(y_pred))
        }
        
        logger.info(f"{model_name} Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}, MAPE: {mape:.2f}%")
        
        return metrics
    
    def get_feature_importance(self, model, feature_names):
        """Get feature importance for XGBoost model"""
        try:
            importance = model.feature_importances_
            feature_importance = dict(zip(feature_names, importance))
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        except:
            return {}
    
    def create_evaluation_plots(self, prophet_results, xgboost_results):
        """Create comprehensive evaluation plots"""
        logger.info("Creating evaluation plots...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('FMCG Sales Model Evaluation', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted - Prophet
        axes[0, 0].scatter(prophet_results['y_true'], prophet_results['y_pred'], alpha=0.6)
        axes[0, 0].plot([prophet_results['y_true'].min(), prophet_results['y_true'].max()], 
                       [prophet_results['y_true'].min(), prophet_results['y_true'].max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Sales')
        axes[0, 0].set_ylabel('Predicted Sales')
        axes[0, 0].set_title('Prophet: Actual vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Actual vs Predicted - XGBoost
        axes[0, 1].scatter(xgboost_results['y_true'], xgboost_results['y_pred'], alpha=0.6)
        axes[0, 1].plot([xgboost_results['y_true'].min(), xgboost_results['y_true'].max()], 
                       [xgboost_results['y_true'].min(), xgboost_results['y_true'].max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Sales')
        axes[0, 1].set_ylabel('Predicted Sales')
        axes[0, 1].set_title('XGBoost: Actual vs Predicted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Time Series Plot - Prophet
        dates = pd.to_datetime(prophet_results['dates'])
        axes[0, 2].plot(dates, prophet_results['y_true'], label='Actual', alpha=0.7)
        axes[0, 2].plot(dates, prophet_results['y_pred'], label='Predicted', alpha=0.7)
        axes[0, 2].set_xlabel('Date')
        axes[0, 2].set_ylabel('Sales')
        axes[0, 2].set_title('Prophet: Time Series Forecast')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Residuals Plot - Prophet
        residuals = prophet_results['y_pred'] - prophet_results['y_true']
        axes[1, 0].scatter(prophet_results['y_pred'], residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Predicted Sales')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Prophet: Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Residuals Plot - XGBoost
        residuals = xgboost_results['y_pred'] - xgboost_results['y_true']
        axes[1, 1].scatter(xgboost_results['y_pred'], residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Sales')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('XGBoost: Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Feature Importance - XGBoost
        if xgboost_results['feature_importance']:
            top_features = dict(list(xgboost_results['feature_importance'].items())[:10])
            features = list(top_features.keys())
            importance = list(top_features.values())
            axes[1, 2].barh(range(len(features)), importance)
            axes[1, 2].set_yticks(range(len(features)))
            axes[1, 2].set_yticklabels(features)
            axes[1, 2].set_xlabel('Feature Importance')
            axes[1, 2].set_title('XGBoost: Top 10 Features')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.reports_dir / 'model_evaluation_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Evaluation plots saved")
    
    def create_interactive_plots(self, prophet_results, xgboost_results):
        """Create interactive Plotly plots"""
        logger.info("Creating interactive plots...")
        
        # 1. Prophet Time Series
        dates = pd.to_datetime(prophet_results['dates'])
        fig_prophet = go.Figure()
        
        fig_prophet.add_trace(go.Scatter(
            x=dates, y=prophet_results['y_true'],
            mode='lines', name='Actual',
            line=dict(color='blue', width=2)
        ))
        
        fig_prophet.add_trace(go.Scatter(
            x=dates, y=prophet_results['y_pred'],
            mode='lines', name='Predicted',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig_prophet.update_layout(
            title='Prophet: Sales Forecast vs Actual',
            xaxis_title='Date',
            yaxis_title='Sales',
            hovermode='x unified'
        )
        
        fig_prophet.write_html(self.reports_dir / 'prophet_forecast.html')
        
        # 2. XGBoost Feature Importance
        if xgboost_results['feature_importance']:
            top_features = dict(list(xgboost_results['feature_importance'].items())[:15])
            
            fig_importance = px.bar(
                x=list(top_features.values()),
                y=list(top_features.keys()),
                orientation='h',
                title='XGBoost: Feature Importance'
            )
            
            fig_importance.update_layout(
                xaxis_title='Importance',
                yaxis_title='Features'
            )
            
            fig_importance.write_html(self.reports_dir / 'xgboost_importance.html')
        
        # 3. Metrics Comparison
        metrics_comparison = pd.DataFrame({
            'Metric': ['MAE', 'RMSE', 'MAPE', 'R²', 'Directional Accuracy'],
            'Prophet': [
                prophet_results['metrics']['mae'],
                prophet_results['metrics']['rmse'],
                prophet_results['metrics']['mape'],
                prophet_results['metrics']['r2'],
                prophet_results['metrics']['directional_accuracy']
            ],
            'XGBoost': [
                xgboost_results['metrics']['mae'],
                xgboost_results['metrics']['rmse'],
                xgboost_results['metrics']['mape'],
                xgboost_results['metrics']['r2'],
                xgboost_results['metrics']['directional_accuracy']
            ]
        })
        
        fig_comparison = px.bar(
            metrics_comparison,
            x='Metric',
            y=['Prophet', 'XGBoost'],
            title='Model Performance Comparison',
            barmode='group'
        )
        
        fig_comparison.write_html(self.reports_dir / 'model_comparison.html')
        
        logger.info("Interactive plots saved")
    
    def save_evaluation_report(self, prophet_results, xgboost_results):
        """Save comprehensive evaluation report"""
        logger.info("Saving evaluation report...")
        
        # Create comprehensive report
        report = {
            'evaluation_summary': {
                'evaluation_date': pd.Timestamp.now().isoformat(),
                'test_data_size': len(prophet_results['y_true']),
                'date_range': f"{pd.to_datetime(prophet_results['dates']).min()} to {pd.to_datetime(prophet_results['dates']).max()}"
            },
            'prophet_results': prophet_results['metrics'],
            'xgboost_results': xgboost_results['metrics'],
            'model_comparison': {
                'best_mae': 'Prophet' if prophet_results['metrics']['mae'] < xgboost_results['metrics']['mae'] else 'XGBoost',
                'best_rmse': 'Prophet' if prophet_results['metrics']['rmse'] < xgboost_results['metrics']['rmse'] else 'XGBoost',
                'best_r2': 'XGBoost' if xgboost_results['metrics']['r2'] > prophet_results['metrics']['r2'] else 'Prophet',
                'best_mape': 'Prophet' if prophet_results['metrics']['mape'] < xgboost_results['metrics']['mape'] else 'XGBoost'
            },
            'feature_importance': xgboost_results['feature_importance']
        }
        
        # Save as JSON
        with open(self.reports_dir / "evaluation.json", 'w') as f:
            import json
            json.dump(report, f, indent=2, default=str)
        
        # Save detailed metrics as CSV
        detailed_metrics = pd.DataFrame({
            'Metric': list(prophet_results['metrics'].keys()),
            'Prophet': list(prophet_results['metrics'].values()),
            'XGBoost': list(xgboost_results['metrics'].values())
        })
        detailed_metrics.to_csv(self.reports_dir / "detailed_metrics.csv", index=False)
        
        logger.info("Evaluation report saved")
        return report
    
    def run_evaluation(self):
        """Run the complete model evaluation pipeline"""
        logger.info("Starting FMCG model evaluation pipeline...")
        
        try:
            # Load models and data
            prophet_model, xgboost_model, test_data = self.load_models_and_data()
            
            # Evaluate models
            prophet_results = self.evaluate_prophet_model(prophet_model, test_data)
            xgboost_results = self.evaluate_xgboost_model(xgboost_model, test_data)
            
            # Create plots
            self.create_evaluation_plots(prophet_results, xgboost_results)
            self.create_interactive_plots(prophet_results, xgboost_results)
            
            # Save evaluation report
            report = self.save_evaluation_report(prophet_results, xgboost_results)
            
            logger.info("Model evaluation completed successfully!")
            return {
                'prophet_results': prophet_results,
                'xgboost_results': xgboost_results,
                'report': report
            }
            
        except Exception as e:
            logger.error(f"Error in evaluation pipeline: {e}")
            raise

def main():
    """Main function to run model evaluation"""
    evaluator = FMCGModelEvaluator()
    results = evaluator.run_evaluation()
    
    print("\n" + "="*50)
    print("MODEL EVALUATION COMPLETED")
    print("="*50)
    print("Prophet Model Performance:")
    print(f"  MAE: {results['prophet_results']['metrics']['mae']:.2f}")
    print(f"  RMSE: {results['prophet_results']['metrics']['rmse']:.2f}")
    print(f"  R²: {results['prophet_results']['metrics']['r2']:.3f}")
    print(f"  MAPE: {results['prophet_results']['metrics']['mape']:.2f}%")
    print("\nXGBoost Model Performance:")
    print(f"  MAE: {results['xgboost_results']['metrics']['mae']:.2f}")
    print(f"  RMSE: {results['xgboost_results']['metrics']['rmse']:.2f}")
    print(f"  R²: {results['xgboost_results']['metrics']['r2']:.3f}")
    print(f"  MAPE: {results['xgboost_results']['metrics']['mape']:.2f}%")
    print("\nBest Model by Metric:")
    for metric, best_model in results['report']['model_comparison'].items():
        print(f"  {metric}: {best_model}")
    print("="*50)

if __name__ == "__main__":
    main() 