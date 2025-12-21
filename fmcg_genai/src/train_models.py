"""
Model Training Module for FMCG Sales Data
Trains Prophet and XGBoost models for sales forecasting
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML imports
from prophet import Prophet
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FMCGModelTrainer:
    def __init__(self, config_path="config.yaml"):
        """Initialize the model trainer with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.processed_dir = Path(self.config['data']['processed_dir'])
        self.models_dir = Path(self.config['models']['prophet_model']).parent
        self.prophet_model_path = self.config['models']['prophet_model']
        self.xgboost_model_path = self.config['models']['xgboost_model']
        self.target_column = self.config['features']['target_column']
        
        # Model parameters
        self.prophet_params = self.config['models_config']['prophet']
        self.xgboost_params = self.config['models_config']['xgboost']
        
        # Create models directory if it doesn't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def load_engineered_data(self):
        """Load engineered data splits"""
        logger.info("Loading engineered data...")
        
        train_df = pd.read_csv(self.processed_dir / "train_features.csv")
        val_df = pd.read_csv(self.processed_dir / "validation_features.csv")
        test_df = pd.read_csv(self.processed_dir / "test_features.csv")
        
        # Convert date columns
        for df in [train_df, val_df, test_df]:
            df['date'] = pd.to_datetime(df['date'])
        
        logger.info(f"Loaded train: {train_df.shape}, validation: {val_df.shape}, test: {test_df.shape}")
        return train_df, val_df, test_df
    
    def prepare_prophet_data(self, df):
        """Prepare data for Prophet model"""
        logger.info("Preparing data for Prophet model...")
        
        # Prophet requires 'ds' (date) and 'y' (target) columns
        # We'll aggregate by date for overall sales forecasting
        daily_sales = df.groupby('date')[self.target_column].sum().reset_index()
        daily_sales.columns = ['ds', 'y']
        
        # Add additional regressors if available
        # Temporarily disabled due to NaN issues
        # if 'price_unit' in df.columns:
        #     daily_price = df.groupby('date')['price_unit'].mean().reset_index()
        #     daily_price.columns = ['ds', 'price']
        #     daily_sales = daily_sales.merge(daily_price, on='ds', how='left')
        
        # if 'promotion_flag' in df.columns:
        #     daily_promotion = df.groupby('date')['promotion_flag'].mean().reset_index()
        #     daily_promotion.columns = ['ds', 'promotion']
        #     daily_sales = daily_sales.merge(daily_promotion, on='ds', how='left')
        
        # Handle NaN values
        daily_sales = daily_sales.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Prophet data shape: {daily_sales.shape}")
        return daily_sales
    
    def train_prophet_model(self, train_data, val_data=None):
        """Train Prophet model for time series forecasting"""
        logger.info("Training Prophet model...")
        
        # Prepare training data
        prophet_train = self.prepare_prophet_data(train_data)
        
        # Initialize Prophet model with parameters
        model = Prophet(
            changepoint_prior_scale=self.prophet_params['changepoint_prior_scale'],
            seasonality_prior_scale=self.prophet_params['seasonality_prior_scale'],
            holidays_prior_scale=self.prophet_params['holidays_prior_scale'],
            seasonality_mode=self.prophet_params['seasonality_mode'],
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        
        # Add additional regressors if available
        # Temporarily disabled due to NaN issues
        # if 'price' in prophet_train.columns:
        #     model.add_regressor('price')
        # if 'promotion' in prophet_train.columns:
        #     model.add_regressor('promotion')
        
        # Fit the model
        model.fit(prophet_train)
        
        logger.info("Prophet model training completed")
        return model
    
    def prepare_xgboost_data(self, df):
        """Prepare data for XGBoost model"""
        logger.info("Preparing data for XGBoost model...")
        
        # Select features for XGBoost (exclude date and target)
        exclude_cols = ['date', self.target_column]
        feature_cols = [col for col in df.columns if col not in exclude_cols and not col.startswith('holiday_name')]
        
        # Handle categorical columns that might not be encoded
        categorical_cols = [col for col in feature_cols if df[col].dtype == 'object']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category').cat.codes
        
        # Prepare X and y
        X = df[feature_cols].fillna(0)
        y = df[self.target_column]
        
        logger.info(f"XGBoost data shape: X={X.shape}, y={y.shape}")
        logger.info(f"Feature columns: {len(feature_cols)}")
        
        return X, y, feature_cols
    
    def train_xgboost_model(self, train_data, val_data=None):
        """Train XGBoost model for sales prediction"""
        logger.info("Training XGBoost model...")
        
        # Prepare training data
        X_train, y_train, feature_cols = self.prepare_xgboost_data(train_data)
        
        # Initialize XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=self.xgboost_params['n_estimators'],
            max_depth=self.xgboost_params['max_depth'],
            learning_rate=self.xgboost_params['learning_rate'],
            subsample=self.xgboost_params['subsample'],
            colsample_bytree=self.xgboost_params['colsample_bytree'],
            random_state=self.xgboost_params['random_state'],
            n_jobs=-1,
            early_stopping_rounds=50
        )
        
        # Prepare validation data if available
        if val_data is not None:
            X_val, y_val, _ = self.prepare_xgboost_data(val_data)
            eval_set = [(X_train, y_train), (X_val, y_val)]
            model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        else:
            model.fit(X_train, y_train, verbose=False)
        
        # Store feature names
        model.feature_names = feature_cols
        
        logger.info("XGBoost model training completed")
        return model
    
    def evaluate_prophet_model(self, model, test_data):
        """Evaluate Prophet model"""
        logger.info("Evaluating Prophet model...")
        
        # Prepare test data
        prophet_test = self.prepare_prophet_data(test_data)
        
        # Handle NaN values in future dataframe
        if 'price' in prophet_test.columns:
            prophet_test['price'] = prophet_test['price'].fillna(method='ffill').fillna(method='bfill')
        if 'promotion' in prophet_test.columns:
            prophet_test['promotion'] = prophet_test['promotion'].fillna(method='ffill').fillna(method='bfill')
        
        # Make predictions
        future = model.make_future_dataframe(periods=len(prophet_test))
        
        # Add regressors to future if they exist
        # Temporarily disabled due to NaN issues
        # if 'price' in prophet_test.columns:
        #     future = future.merge(prophet_test[['ds', 'price']], on='ds', how='left')
        # if 'promotion' in prophet_test.columns:
        #     future = future.merge(prophet_test[['ds', 'promotion']], on='ds', how='left')
        
        forecast = model.predict(future)
        
        # Calculate metrics
        y_true = prophet_test['y'].values
        y_pred = forecast['yhat'].tail(len(y_true)).values
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
        
        logger.info(f"Prophet Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        
        return metrics, forecast
    
    def evaluate_xgboost_model(self, model, test_data):
        """Evaluate XGBoost model"""
        logger.info("Evaluating XGBoost model...")
        
        # Prepare test data
        X_test, y_test, _ = self.prepare_xgboost_data(test_data)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
        
        logger.info(f"XGBoost Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}, MAPE: {mape:.2f}%")
        
        return metrics, y_pred
    
    def save_models(self, prophet_model, xgboost_model):
        """Save trained models"""
        logger.info("Saving trained models...")
        
        # Save Prophet model
        with open(self.prophet_model_path, 'wb') as f:
            joblib.dump(prophet_model, f)
        logger.info(f"Prophet model saved to {self.prophet_model_path}")
        
        # Save XGBoost model
        with open(self.xgboost_model_path, 'wb') as f:
            joblib.dump(xgboost_model, f)
        logger.info(f"XGBoost model saved to {self.xgboost_model_path}")
        
        # Save model metadata
        metadata = {
            'prophet_params': self.prophet_params,
            'xgboost_params': self.xgboost_params,
            'target_column': self.target_column,
            'training_date': pd.Timestamp.now().isoformat()
        }
        
        with open(self.models_dir / "model_metadata.json", 'w') as f:
            import json
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info("Model metadata saved")
    
    def generate_training_report(self, prophet_metrics, xgboost_metrics):
        """Generate training report"""
        logger.info("Generating training report...")
        
        report = {
            'prophet_metrics': prophet_metrics,
            'xgboost_metrics': xgboost_metrics,
            'training_summary': {
                'prophet_model_path': self.prophet_model_path,
                'xgboost_model_path': self.xgboost_model_path,
                'training_date': pd.Timestamp.now().isoformat()
            }
        }
        
        # Save report
        with open(self.models_dir / "training_report.json", 'w') as f:
            import json
            json.dump(report, f, indent=2, default=str)
        
        logger.info("Training report saved")
        return report
    
    def run_training(self):
        """Run the complete model training pipeline"""
        logger.info("Starting FMCG model training pipeline...")
        
        try:
            # Load engineered data
            train_df, val_df, test_df = self.load_engineered_data()
            
            # Train Prophet model
            prophet_model = self.train_prophet_model(train_df, val_df)
            
            # Train XGBoost model
            xgboost_model = self.train_xgboost_model(train_df, val_df)
            
            # Evaluate models
            prophet_metrics, prophet_forecast = self.evaluate_prophet_model(prophet_model, test_df)
            xgboost_metrics, xgboost_predictions = self.evaluate_xgboost_model(xgboost_model, test_df)
            
            # Save models
            self.save_models(prophet_model, xgboost_model)
            
            # Generate training report
            report = self.generate_training_report(prophet_metrics, xgboost_metrics)
            
            logger.info("Model training completed successfully!")
            return {
                'prophet_model': prophet_model,
                'xgboost_model': xgboost_model,
                'prophet_metrics': prophet_metrics,
                'xgboost_metrics': xgboost_metrics,
                'report': report
            }
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            raise

def main():
    """Main function to run model training"""
    trainer = FMCGModelTrainer()
    results = trainer.run_training()
    
    print("\n" + "="*50)
    print("MODEL TRAINING COMPLETED")
    print("="*50)
    print("Prophet Model Metrics:")
    print(f"  MAE: {results['prophet_metrics']['mae']:.2f}")
    print(f"  RMSE: {results['prophet_metrics']['rmse']:.2f}")
    print(f"  MAPE: {results['prophet_metrics']['mape']:.2f}%")
    print("\nXGBoost Model Metrics:")
    print(f"  MAE: {results['xgboost_metrics']['mae']:.2f}")
    print(f"  RMSE: {results['xgboost_metrics']['rmse']:.2f}")
    print(f"  R²: {results['xgboost_metrics']['r2']:.3f}")
    print(f"  MAPE: {results['xgboost_metrics']['mape']:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main() 