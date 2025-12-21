"""
Feature Engineering Module for FMCG Sales Data
Creates lag features, rolling averages, time-based features, and categorical encodings
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from datetime import datetime, timedelta
import holidays
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FMCGFeatureEngineer:
    def __init__(self, config_path="config.yaml"):
        """Initialize the feature engineer with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.processed_dir = Path(self.config['data']['processed_dir'])
        self.features_file = self.config['data']['features_file']
        self.target_column = self.config['features']['target_column']
        self.lag_features = self.config['features']['lag_features']
        self.rolling_windows = self.config['features']['rolling_windows']
        self.categorical_columns = self.config['features']['categorical_columns']
        
        # Initialize encoders
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # Create processed directory if it doesn't exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def load_processed_data(self):
        """Load processed data splits"""
        logger.info("Loading processed data...")
        
        train_df = pd.read_csv(self.processed_dir / "train.csv")
        val_df = pd.read_csv(self.processed_dir / "validation.csv")
        test_df = pd.read_csv(self.processed_dir / "test.csv")
        
        # Convert date columns
        for df in [train_df, val_df, test_df]:
            df['date'] = pd.to_datetime(df['date'])
        
        logger.info(f"Loaded train: {train_df.shape}, validation: {val_df.shape}, test: {test_df.shape}")
        return train_df, val_df, test_df
    
    def create_time_features(self, df):
        """Create time-based features"""
        logger.info("Creating time-based features...")
        
        # Basic time features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
        
        # Seasonality features
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        df['sin_day_of_week'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['cos_day_of_week'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def create_holiday_features(self, df):
        """Create holiday-related features"""
        logger.info("Creating holiday features...")
        
        # Create holiday calendar (using US holidays as example)
        # You can modify this for your specific country/region
        us_holidays = holidays.US()
        
        df['is_holiday'] = df['date'].dt.date.apply(lambda x: x in us_holidays).astype(int)
        df['holiday_name'] = df['date'].apply(lambda x: us_holidays.get(x, 'No Holiday'))
        
        # Create holiday proximity features
        df['days_to_holiday'] = 0
        df['days_after_holiday'] = 0
        
        for idx, row in df.iterrows():
            date = row['date']
            date_date = date.date()
            # Find nearest holiday before and after
            holiday_dates = list(us_holidays.keys())
            holiday_dates = [d for d in holiday_dates if d.year == date.year]
            
            if holiday_dates:
                # Days to next holiday
                future_holidays = [d for d in holiday_dates if d >= date_date]
                if future_holidays:
                    next_holiday = min(future_holidays)
                    df.loc[idx, 'days_to_holiday'] = (next_holiday - date_date).days
                
                # Days after previous holiday
                past_holidays = [d for d in holiday_dates if d <= date_date]
                if past_holidays:
                    prev_holiday = max(past_holidays)
                    df.loc[idx, 'days_after_holiday'] = (date_date - prev_holiday).days
        
        return df
    
    def create_lag_features(self, df):
        """Create lag features for time series"""
        logger.info("Creating lag features...")
        
        # Sort by date and product/region combination
        df = df.sort_values(['date', 'sku', 'region']).reset_index(drop=True)
        
        # Create lag features for different combinations
        for lag in self.lag_features:
            # Overall lag
            df[f'lag_{lag}_sales'] = df.groupby(['sku', 'region'])['units_sold'].shift(lag)
            
            # Price lag
            df[f'lag_{lag}_price'] = df.groupby(['sku', 'region'])['price_unit'].shift(lag)
            
            # Stock lag
            df[f'lag_{lag}_stock'] = df.groupby(['sku', 'region'])['stock_available'].shift(lag)
            
            # Delivery days lag
            df[f'lag_{lag}_delivery'] = df.groupby(['sku', 'region'])['delivery_days'].shift(lag)
        
        return df
    
    def create_rolling_features(self, df):
        """Create rolling average features"""
        logger.info("Creating rolling features...")
        
        # Sort by date and product/region combination
        df = df.sort_values(['date', 'sku', 'region']).reset_index(drop=True)
        
        for window in self.rolling_windows:
            # Rolling sales averages
            rolling_mean = df.groupby(['sku', 'region'])['units_sold'].rolling(
                window=window, min_periods=1).mean().reset_index(0, drop=True)
            df[f'rolling_{window}_sales_mean'] = rolling_mean
            
            rolling_std = df.groupby(['sku', 'region'])['units_sold'].rolling(
                window=window, min_periods=1).std().reset_index(0, drop=True)
            df[f'rolling_{window}_sales_std'] = rolling_std
            
            rolling_min = df.groupby(['sku', 'region'])['units_sold'].rolling(
                window=window, min_periods=1).min().reset_index(0, drop=True)
            df[f'rolling_{window}_sales_min'] = rolling_min
            
            rolling_max = df.groupby(['sku', 'region'])['units_sold'].rolling(
                window=window, min_periods=1).max().reset_index(0, drop=True)
            df[f'rolling_{window}_sales_max'] = rolling_max
            
            # Rolling price averages
            rolling_price = df.groupby(['sku', 'region'])['price_unit'].rolling(
                window=window, min_periods=1).mean().reset_index(0, drop=True)
            df[f'rolling_{window}_price_mean'] = rolling_price
            
            # Rolling stock averages
            rolling_stock = df.groupby(['sku', 'region'])['stock_available'].rolling(
                window=window, min_periods=1).mean().reset_index(0, drop=True)
            df[f'rolling_{window}_stock_mean'] = rolling_stock
        
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features"""
        logger.info("Creating interaction features...")
        
        # Price-volume interactions
        df['price_volume_ratio'] = df['price_unit'] * df['units_sold']
        df['price_stock_ratio'] = df['price_unit'] / (df['stock_available'] + 1)
        
        # Promotion interactions
        df['promotion_effect'] = df['promotion_flag'] * df['units_sold']
        df['promotion_price_effect'] = df['promotion_flag'] * df['price_unit']
        
        # Delivery interactions
        df['delivery_stock_ratio'] = df['delivery_days'] / (df['stock_available'] + 1)
        df['delivery_sales_ratio'] = df['delivery_days'] * df['units_sold']
        
        # Seasonal interactions
        df['monthly_sales_trend'] = df['month'] * df['units_sold']
        df['weekend_sales_boost'] = df['is_weekend'] * df['units_sold']
        
        return df
    
    def encode_categorical_features(self, df, is_training=True):
        """Encode categorical features"""
        logger.info("Encoding categorical features...")
        
        for col in self.categorical_columns:
            if col in df.columns:
                if is_training:
                    # Create new encoder for training
                    le = LabelEncoder()
                    df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    # Use existing encoder for test/validation
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        # Handle unseen categories
                        df[f'{col}_encoded'] = df[col].astype(str).map(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
                    else:
                        df[f'{col}_encoded'] = 0
        
        return df
    
    def scale_numeric_features(self, df, is_training=True):
        """Scale numeric features"""
        logger.info("Scaling numeric features...")
        
        # Select numeric features for scaling (excluding date, target, and encoded features)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['date', self.target_column] + [f'{col}_encoded' for col in self.categorical_columns]
        scale_cols = [col for col in numeric_cols if col not in exclude_cols and not col.startswith('lag_') and not col.startswith('rolling_')]
        
        if is_training:
            df[scale_cols] = self.scaler.fit_transform(df[scale_cols])
        else:
            df[scale_cols] = self.scaler.transform(df[scale_cols])
        
        return df
    
    def handle_missing_features(self, df):
        """Handle missing values in engineered features"""
        logger.info("Handling missing values in engineered features...")
        
        # Fill lag features with 0 or forward fill
        lag_cols = [col for col in df.columns if col.startswith('lag_')]
        for col in lag_cols:
            df[col] = df[col].fillna(0)
        
        # Fill rolling features with forward fill
        rolling_cols = [col for col in df.columns if col.startswith('rolling_')]
        for col in rolling_cols:
            df[col] = df[col].fillna(method='ffill').fillna(0)
        
        # Fill other numeric features with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def engineer_features(self, df, is_training=True):
        """Apply all feature engineering steps"""
        logger.info(f"Engineering features for {'training' if is_training else 'test'} data...")
        
        # Create time features
        df = self.create_time_features(df)
        
        # Create holiday features
        df = self.create_holiday_features(df)
        
        # Create lag features
        df = self.create_lag_features(df)
        
        # Create rolling features
        # df = self.create_rolling_features(df)  # Temporarily disabled due to index issues
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, is_training)
        
        # Scale numeric features
        df = self.scale_numeric_features(df, is_training)
        
        # Handle missing values
        df = self.handle_missing_features(df)
        
        return df
    
    def save_engineered_data(self, train_df, val_df, test_df):
        """Save engineered data"""
        logger.info("Saving engineered data...")
        
        # Save engineered datasets
        train_df.to_csv(self.processed_dir / "train_features.csv", index=False)
        val_df.to_csv(self.processed_dir / "validation_features.csv", index=False)
        test_df.to_csv(self.processed_dir / "test_features.csv", index=False)
        
        # Save combined features file
        combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        combined_df.to_csv(self.processed_dir / self.features_file, index=False)
        
        # Save encoders
        import joblib
        joblib.dump(self.label_encoders, self.processed_dir / "label_encoders.pkl")
        joblib.dump(self.scaler, self.processed_dir / "scaler.pkl")
        
        logger.info("Engineered data saved successfully")
        
        return {
            'train': train_df,
            'validation': val_df,
            'test': test_df,
            'combined': combined_df
        }
    
    def generate_feature_summary(self, df):
        """Generate feature summary"""
        logger.info("Generating feature summary...")
        
        feature_summary = {
            'total_features': len(df.columns),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object']).columns),
            'lag_features': len([col for col in df.columns if col.startswith('lag_')]),
            'rolling_features': len([col for col in df.columns if col.startswith('rolling_')]),
            'time_features': len([col for col in df.columns if col in ['year', 'month', 'quarter', 'day_of_week', 'is_weekend']]),
            'feature_list': list(df.columns)
        }
        
        # Save summary
        import json
        with open(self.processed_dir / "feature_summary.json", 'w') as f:
            json.dump(feature_summary, f, indent=2)
        
        logger.info("Feature summary saved")
        return feature_summary
    
    def run_feature_engineering(self):
        """Run the complete feature engineering pipeline"""
        logger.info("Starting FMCG feature engineering pipeline...")
        
        try:
            # Load processed data
            train_df, val_df, test_df = self.load_processed_data()
            
            # Engineer features for each dataset
            train_features = self.engineer_features(train_df, is_training=True)
            val_features = self.engineer_features(val_df, is_training=False)
            test_features = self.engineer_features(test_df, is_training=False)
            
            # Save engineered data
            data_splits = self.save_engineered_data(train_features, val_features, test_features)
            
            # Generate feature summary
            summary = self.generate_feature_summary(data_splits['combined'])
            
            logger.info("Feature engineering completed successfully!")
            return data_splits, summary
            
        except Exception as e:
            logger.error(f"Error in feature engineering pipeline: {e}")
            raise

def main():
    """Main function to run feature engineering"""
    engineer = FMCGFeatureEngineer()
    data_splits, summary = engineer.run_feature_engineering()
    
    print("\n" + "="*50)
    print("FEATURE ENGINEERING COMPLETED")
    print("="*50)
    print(f"Total features: {summary['total_features']}")
    print(f"Numeric features: {summary['numeric_features']}")
    print(f"Lag features: {summary['lag_features']}")
    print(f"Rolling features: {summary['rolling_features']}")
    print(f"Time features: {summary['time_features']}")
    print("="*50)

if __name__ == "__main__":
    main() 