"""
Main Orchestration Script for FMCG Sales Analytics & Forecasting Pipeline
Runs the complete end-to-end workflow: preprocessing → feature engineering → training → evaluation → SHAP → RAG
"""

import os
import sys
import yaml
import logging
import time
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import pipeline modules
from data_preprocessing import FMCGDataPreprocessor
from feature_engineering import FMCGFeatureEngineer
from train_models import FMCGModelTrainer
from evaluate_models import FMCGModelEvaluator
from explainability import FMCGExplainability
from rag_pipeline import FMCGRAGPipeline

# Setup logging
def setup_logging():
    """Setup comprehensive logging"""
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "fmcg_pipeline.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

class FMCGPipelineOrchestrator:
    def __init__(self, config_path="config.yaml"):
        """Initialize the orchestrator"""
        self.logger = setup_logging()
        
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.start_time = time.time()
        self.pipeline_status = {
            'preprocessing': False,
            'feature_engineering': False,
            'training': False,
            'evaluation': False,
            'explainability': False,
            'rag': False
        }
        
    def print_banner(self):
        """Print pipeline banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    FMCG Sales Analytics & Forecasting Pipeline               ║
║                                                                              ║
║  🚀 End-to-End ML + Generative AI Workflow                                  ║
║  📊 Data Preprocessing → Feature Engineering → Model Training               ║
║  🔍 Model Evaluation → SHAP Explainability → RAG Pipeline                   ║
║  📈 Dashboard → Business Insights                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def print_step_header(self, step_name, step_number, total_steps):
        """Print step header"""
        print(f"\n{'='*80}")
        print(f"STEP {step_number}/{total_steps}: {step_name.upper()}")
        print(f"{'='*80}")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
    
    def print_step_footer(self, step_name, success=True):
        """Print step footer"""
        status = "✅ COMPLETED" if success else "❌ FAILED"
        duration = time.time() - self.start_time
        print(f"\n{'='*80}")
        print(f"{step_name.upper()}: {status}")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        print(f"{'='*80}")
    
    def run_preprocessing(self):
        """Run data preprocessing step"""
        self.print_step_header("Data Preprocessing", 1, 6)
        
        try:
            self.logger.info("Starting data preprocessing...")
            
            preprocessor = FMCGDataPreprocessor()
            data_splits, summary = preprocessor.run_preprocessing()
            
            self.pipeline_status['preprocessing'] = True
            self.print_step_footer("Data Preprocessing", True)
            
            return data_splits, summary
            
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {e}")
            self.print_step_footer("Data Preprocessing", False)
            raise
    
    def run_feature_engineering(self):
        """Run feature engineering step"""
        self.print_step_header("Feature Engineering", 2, 6)
        
        try:
            self.logger.info("Starting feature engineering...")
            
            engineer = FMCGFeatureEngineer()
            data_splits, summary = engineer.run_feature_engineering()
            
            self.pipeline_status['feature_engineering'] = True
            self.print_step_footer("Feature Engineering", True)
            
            return data_splits, summary
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            self.print_step_footer("Feature Engineering", False)
            raise
    
    def run_model_training(self):
        """Run model training step"""
        self.print_step_header("Model Training", 3, 6)
        
        try:
            self.logger.info("Starting model training...")
            
            trainer = FMCGModelTrainer()
            results = trainer.run_training()
            
            self.pipeline_status['training'] = True
            self.print_step_footer("Model Training", True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            self.print_step_footer("Model Training", False)
            raise
    
    def run_model_evaluation(self):
        """Run model evaluation step"""
        self.print_step_header("Model Evaluation", 4, 6)
        
        try:
            self.logger.info("Starting model evaluation...")
            
            evaluator = FMCGModelEvaluator()
            results = evaluator.run_evaluation()
            
            self.pipeline_status['evaluation'] = True
            self.print_step_footer("Model Evaluation", True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            self.print_step_footer("Model Evaluation", False)
            raise
    
    def run_explainability(self):
        """Run SHAP explainability step"""
        self.print_step_header("SHAP Explainability", 5, 6)
        
        try:
            self.logger.info("Starting SHAP explainability analysis...")
            
            explainer = FMCGExplainability()
            results = explainer.run_explainability_analysis()
            
            self.pipeline_status['explainability'] = True
            self.print_step_footer("SHAP Explainability", True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"SHAP explainability failed: {e}")
            self.print_step_footer("SHAP Explainability", False)
            raise
    
    def run_rag_pipeline(self):
        """Run RAG pipeline step"""
        self.print_step_header("RAG Pipeline", 6, 6)
        
        try:
            self.logger.info("Starting RAG pipeline...")
            
            rag_pipeline = FMCGRAGPipeline()
            results = rag_pipeline.run_rag_pipeline()
            
            self.pipeline_status['rag'] = True
            self.print_step_footer("RAG Pipeline", True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"RAG pipeline failed: {e}")
            self.print_step_footer("RAG Pipeline", False)
            raise
    
    def generate_pipeline_report(self, results):
        """Generate comprehensive pipeline report"""
        self.logger.info("Generating pipeline report...")
        
        total_duration = time.time() - self.start_time
        
        report = {
            'pipeline_summary': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration_seconds': total_duration,
                'total_duration_minutes': total_duration / 60,
                'pipeline_status': self.pipeline_status,
                'successful_steps': sum(self.pipeline_status.values()),
                'total_steps': len(self.pipeline_status)
            },
            'step_results': results,
            'next_steps': [
                "Run the Streamlit dashboard: streamlit run src/dashboard_app.py",
                "Explore the generated reports in the reports/ directory",
                "Check model performance in models/ directory",
                "Test RAG queries using the dashboard"
            ]
        }
        
        # Save report
        report_path = Path("reports/pipeline_report.json")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            import json
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def print_final_summary(self, report):
        """Print final pipeline summary"""
        summary = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              PIPELINE COMPLETED                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 Pipeline Summary:
   • Total Duration: {report['pipeline_summary']['total_duration_minutes']:.2f} minutes
   • Successful Steps: {report['pipeline_summary']['successful_steps']}/{report['pipeline_summary']['total_steps']}
   • Start Time: {report['pipeline_summary']['start_time']}
   • End Time: {report['pipeline_summary']['end_time']}

✅ Completed Steps:
"""
        
        for step, status in self.pipeline_status.items():
            status_icon = "✅" if status else "❌"
            summary += f"   {status_icon} {step.replace('_', ' ').title()}\n"
        
        summary += f"""
🚀 Next Steps:
"""
        
        for i, step in enumerate(report['next_steps'], 1):
            summary += f"   {i}. {step}\n"
        
        summary += f"""
📁 Generated Files:
   • Models: models/
   • Reports: reports/
   • Processed Data: data/processed/
   • Vector Store: vector_store/
   • Logs: logs/

🎯 Ready to Use:
   • Streamlit Dashboard: streamlit run src/dashboard_app.py
   • RAG Queries: Available in dashboard
   • Model Predictions: Prophet & XGBoost models trained
   • Business Insights: SHAP analysis completed

╔══════════════════════════════════════════════════════════════════════════════╗
║                    🎉 FMCG Analytics Pipeline Successfully Completed! 🎉     ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        
        print(summary)
    
    def run_pipeline(self, skip_steps=None):
        """Run the complete pipeline"""
        if skip_steps is None:
            skip_steps = []
        
        self.print_banner()
        
        results = {}
        
        try:
            # Step 1: Data Preprocessing
            if 'preprocessing' not in skip_steps:
                data_splits, summary = self.run_preprocessing()
                results['preprocessing'] = {'data_splits': 'success', 'summary': summary}
            else:
                self.logger.info("Skipping data preprocessing")
            
            # Step 2: Feature Engineering
            if 'feature_engineering' not in skip_steps:
                data_splits, summary = self.run_feature_engineering()
                results['feature_engineering'] = {'data_splits': 'success', 'summary': summary}
            else:
                self.logger.info("Skipping feature engineering")
            
            # Step 3: Model Training
            if 'training' not in skip_steps:
                training_results = self.run_model_training()
                results['training'] = training_results
            else:
                self.logger.info("Skipping model training")
            
            # Step 4: Model Evaluation
            if 'evaluation' not in skip_steps:
                eval_results = self.run_model_evaluation()
                results['evaluation'] = eval_results
            else:
                self.logger.info("Skipping model evaluation")
            
            # Step 5: SHAP Explainability
            if 'explainability' not in skip_steps:
                shap_results = self.run_explainability()
                results['explainability'] = shap_results
            else:
                self.logger.info("Skipping SHAP explainability")
            
            # Step 6: RAG Pipeline
            if 'rag' not in skip_steps:
                rag_results = self.run_rag_pipeline()
                results['rag'] = rag_results
            else:
                self.logger.info("Skipping RAG pipeline")
            
            # Generate final report
            report = self.generate_pipeline_report(results)
            
            # Print final summary
            self.print_final_summary(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            print(f"\n❌ Pipeline failed: {e}")
            return None

def main():
    """Main function to run the pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="FMCG Sales Analytics Pipeline")
    parser.add_argument(
        "--skip", 
        nargs="+", 
        choices=['preprocessing', 'feature_engineering', 'training', 'evaluation', 'explainability', 'rag'],
        help="Steps to skip"
    )
    parser.add_argument(
        "--config", 
        default="config.yaml",
        help="Configuration file path"
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    orchestrator = FMCGPipelineOrchestrator(args.config)
    report = orchestrator.run_pipeline(skip_steps=args.skip or [])
    
    if report:
        print("\n🎉 Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 