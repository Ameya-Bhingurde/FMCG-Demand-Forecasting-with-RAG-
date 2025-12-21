"""
Data Preprocessing Module for FMCG Sales Data
Handles data cleaning, outlier detection, and time-series train/test split
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FMCGDataPreprocessor:
    def __init__(self, config_path="config.yaml"):
        """Initialize the preprocessor with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.raw_dir = Path(self.config['data']['raw_dir'])
        self.processed_dir = Path(self.config['data']['processed_dir'])
        self.main_file = self.config['data']['main_file']
        self.cleaned_file = self.config['data']['cleaned_file']
        
        # Create processed directory if it doesn't exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load all FMCG data files"""
        logger.info("Loading FMCG data files...")
        
        # Load main CSV file
        main_data_path = self.raw_dir / self.main_file
        if main_data_path.exists():
            df = pd.read_csv(main_data_path)
            logger.info(f"Loaded main data: {df.shape}")
        else:
            raise FileNotFoundError(f"Main data file not found: {main_data_path}")
        
        # Load additional CSV files if they exist
        additional_files = list(self.raw_dir.glob("*.csv"))
        additional_files = [f for f in additional_files if f.name != self.main_file]
        
        for file_path in additional_files:
            try:
                additional_df = pd.read_csv(file_path)
                logger.info(f"Loaded additional file {file_path.name}: {additional_df.shape}")
                # Append if columns match
                if set(additional_df.columns) == set(df.columns):
                    df = pd.concat([df, additional_df], ignore_index=True)
                    logger.info(f"Appended {file_path.name}")
            except Exception as e:
                logger.warning(f"Could not load {file_path.name}: {e}")
        
        return df
    
    def clean_data(self, df):
        """Clean the dataset"""
        logger.info("Starting data cleaning...")
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Handle missing values
        logger.info("Handling missing values...")
        
        # Fill missing values based on column type
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # For numeric columns, fill with median
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
                logger.info(f"Filled missing values in {col} with median")
        
        # For categorical columns, fill with mode
        for col in categorical_columns:
            if df[col].isnull().sum() > 0:
                mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col] = df[col].fillna(mode_value)
                logger.info(f"Filled missing values in {col} with mode: {mode_value}")
        
        # Remove outliers using IQR method for numeric columns
        logger.info("Removing outliers...")
        outlier_threshold = self.config['preprocessing']['outlier_threshold']
        
        for col in numeric_columns:
            if col != 'date':  # Skip date column
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - outlier_threshold * IQR
                upper_bound = Q3 + outlier_threshold * IQR
                
                outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outliers_count = outliers_mask.sum()
                
                if outliers_count > 0:
                    df = df[~outliers_mask]
                    logger.info(f"Removed {outliers_count} outliers from {col}")
        
        # Filter by date range
        min_date = pd.to_datetime(self.config['preprocessing']['min_date'])
        max_date = pd.to_datetime(self.config['preprocessing']['max_date'])
        df = df[(df['date'] >= min_date) & (df['date'] <= max_date)]
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"Cleaned data shape: {df.shape}")
        return df
    
    def create_time_series_split(self, df):
        """Create time-series train/test split"""
        logger.info("Creating time-series split...")
        
        train_split_date = pd.to_datetime(self.config['preprocessing']['train_split_date'])
        test_split_date = pd.to_datetime(self.config['preprocessing']['test_split_date'])
        
        # Create train/test split
        train_df = df[df['date'] <= train_split_date].copy()
        test_df = df[df['date'] >= test_split_date].copy()
        
        # Create validation set (last 3 months of training data)
        val_start_date = train_split_date - pd.DateOffset(months=3)
        val_df = df[(df['date'] > val_start_date) & (df['date'] <= train_split_date)].copy()
        
        logger.info(f"Train set: {train_df.shape} ({train_df['date'].min()} to {train_df['date'].max()})")
        logger.info(f"Validation set: {val_df.shape} ({val_df['date'].min()} to {val_df['date'].max()})")
        logger.info(f"Test set: {test_df.shape} ({test_df['date'].min()} to {test_df['date'].max()})")
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, train_df, val_df, test_df, full_df):
        """Save processed data to files"""
        logger.info("Saving processed data...")
        
        # Save full cleaned dataset
        full_df.to_csv(self.processed_dir / self.cleaned_file, index=False)
        logger.info(f"Saved full cleaned data to {self.processed_dir / self.cleaned_file}")
        
        # Save train/val/test splits
        train_df.to_csv(self.processed_dir / "train.csv", index=False)
        val_df.to_csv(self.processed_dir / "validation.csv", index=False)
        test_df.to_csv(self.processed_dir / "test.csv", index=False)
        
        logger.info("Saved train/validation/test splits")
        
        return {
            'train': train_df,
            'validation': val_df,
            'test': test_df,
            'full': full_df
        }
    
    def generate_data_summary(self, df):
        """Generate data summary statistics"""
        logger.info("Generating data summary...")
        
        summary = {
            'total_records': len(df),
            'date_range': f"{df['date'].min()} to {df['date'].max()}",
            'unique_products': df['sku'].nunique(),
            'unique_brands': df['brand'].nunique(),
            'unique_regions': df['region'].nunique(),
            'total_sales': df['units_sold'].sum(),
            'avg_sales_per_day': df.groupby('date')['units_sold'].sum().mean(),
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict()
        }
        
        # Save summary
        import json
        with open(self.processed_dir / "data_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("Data summary saved")
        return summary
    
    def run_preprocessing(self):
        """Run the complete preprocessing pipeline"""
        logger.info("Starting FMCG data preprocessing pipeline...")
        
        try:
            # Load data
            df = self.load_data()
            
            # Clean data
            df_cleaned = self.clean_data(df)
            
            # Create time-series split
            train_df, val_df, test_df = self.create_time_series_split(df_cleaned)
            
            # Save processed data
            data_splits = self.save_processed_data(train_df, val_df, test_df, df_cleaned)
            
            # Generate summary
            summary = self.generate_data_summary(df_cleaned)
            
            logger.info("Data preprocessing completed successfully!")
            return data_splits, summary
            
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {e}")
            raise

def main():
    """Main function to run preprocessing"""
    preprocessor = FMCGDataPreprocessor()
    data_splits, summary = preprocessor.run_preprocessing()
    
    print("\n" + "="*50)
    print("PREPROCESSING COMPLETED")
    print("="*50)
    print(f"Total records: {summary['total_records']}")
    print(f"Date range: {summary['date_range']}")
    print(f"Unique products: {summary['unique_products']}")
    print(f"Total sales: {summary['total_sales']:,.0f}")
    print(f"Average daily sales: {summary['avg_sales_per_day']:,.0f}")
    print("="*50)

if __name__ == "__main__":
    main() 