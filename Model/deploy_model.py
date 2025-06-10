#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CenanoInk Project: Model Deployment Pipeline

This script provides an automated pipeline for deploying updated models
to the CenanoInk dashboard. It handles:

1. Training the ensemble model on the latest data
2. Evaluating model performance
3. Running comparison against previous models
4. Generating model documentation
5. Deploying the model to the dashboard

Usage:
    python deploy_model.py --data_path /path/to/data.csv --target_col relevance_category
"""

import argparse
import os
import sys
import logging
import json
import shutil
from datetime import datetime
from pathlib import Path
import subprocess
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'logs' / f'model_deployment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def run_command(command, description, check=True):
    """
    Run a shell command and log the output.
    
    Parameters:
    -----------
    command : str or list
        The command to run
    description : str
        Description of the command for logging
    check : bool, default=True
        Whether to check the return code
        
    Returns:
    --------
    returncode : int
        The return code of the command
    """
    logger.info(f"Running {description}...")
    
    try:
        result = subprocess.run(
            command,
            shell=True if isinstance(command, str) else False,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"Command completed with return code {result.returncode}")
        
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.debug(f"STDOUT: {line}")
                
        if result.stderr:
            for line in result.stderr.splitlines():
                logger.warning(f"STDERR: {line}")
                
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with return code {e.returncode}")
        logger.error(f"STDERR: {e.stderr}")
        
        if not check:
            return e.returncode
        
        raise


def train_model(data_path, target_col, output_dir=None):
    """
    Train the ensemble model.
    
    Parameters:
    -----------
    data_path : str
        Path to the data CSV file
    target_col : str
        Name of the target column
    output_dir : str, optional
        Directory to save output files
        
    Returns:
    --------
    model_path : str
        Path to the trained model
    """
    logger.info("Training ensemble model...")
    
    # Build command
    cmd = [
        sys.executable,
        str(project_root / 'scripts' / 'train_ensemble_model.py'),
        '--data_path', data_path,
        '--target_col', target_col
    ]
    
    if output_dir:
        cmd.extend(['--output_dir', output_dir])
    
    # Run command
    returncode = run_command(cmd, "model training")
    
    if returncode != 0:
        logger.error("Model training failed")
        raise RuntimeError("Model training failed")
    
    # Find the model file (assumes latest model is in the specified output_dir)
    if output_dir:
        model_dir = Path(output_dir)
    else:
        model_dir = project_root / 'models' / 'trained'
    
    model_files = list(model_dir.glob('*.pkl'))
    if not model_files:
        logger.error(f"No model files found in {model_dir}")
        raise FileNotFoundError(f"No model files found in {model_dir}")
    
    # Get the latest model
    latest_model = max(model_files, key=os.path.getctime)
    
    logger.info(f"Model trained successfully: {latest_model}")
    return str(latest_model)


def optimize_hyperparameters(data_path, target_col, n_trials=100):
    """
    Optimize hyperparameters using Optuna.
    
    Parameters:
    -----------
    data_path : str
        Path to the data CSV file
    target_col : str
        Name of the target column
    n_trials : int, default=100
        Number of optimization trials
        
    Returns:
    --------
    params_path : str
        Path to the optimal parameters file
    """
    logger.info(f"Optimizing hyperparameters with {n_trials} trials...")
    
    # Build command
    cmd = [
        sys.executable,
        str(project_root / 'scripts' / 'optimize_hyperparameters.py'),
        '--data_path', data_path,
        '--target_col', target_col,
        '--n_trials', str(n_trials)
    ]
    
    # Run command
    returncode = run_command(cmd, "hyperparameter optimization")
    
    if returncode != 0:
        logger.error("Hyperparameter optimization failed")
        raise RuntimeError("Hyperparameter optimization failed")
    
    # Find the parameters file
    params_path = project_root / 'models' / 'configs' / 'optimal_params.json'
    
    if not params_path.exists():
        logger.error(f"Optimal parameters file not found: {params_path}")
        raise FileNotFoundError(f"Optimal parameters file not found: {params_path}")
    
    logger.info(f"Hyperparameters optimized successfully: {params_path}")
    return str(params_path)


def run_model_comparison(data_path, target_col, output_dir=None):
    """
    Run model comparison between LightGBM, CatBoost, XGBoost and the ensemble.
    
    Parameters:
    -----------
    data_path : str
        Path to the data CSV file
    target_col : str
        Name of the target column
    output_dir : str, optional
        Directory to save output files
        
    Returns:
    --------
    comparison_path : str
        Path to the comparison results
    """
    logger.info("Running model comparison...")
    
    # Build command
    cmd = [
        sys.executable,
        str(project_root / 'scripts' / 'model_comparison.py'),
        '--data_path', data_path,
        '--target_col', target_col
    ]
    
    if output_dir:
        cmd.extend(['--output_dir', output_dir])
    
    # Run command
    returncode = run_command(cmd, "model comparison")
    
    if returncode != 0:
        logger.error("Model comparison failed")
        raise RuntimeError("Model comparison failed")
    
    # Find the comparison results
    if output_dir:
        comparison_path = Path(output_dir) / 'model_comparison_metrics.csv'
    else:
        # Find the latest comparison results
        results_dir = project_root / 'results'
        comparison_dirs = [d for d in results_dir.glob('model_comparison_*') if d.is_dir()]
        
        if not comparison_dirs:
            logger.error(f"No comparison results found in {results_dir}")
            raise FileNotFoundError(f"No comparison results found in {results_dir}")
        
        latest_dir = max(comparison_dirs, key=os.path.getctime)
        comparison_path = latest_dir / 'model_comparison_metrics.csv'
    
    if not comparison_path.exists():
        logger.error(f"Comparison results not found: {comparison_path}")
        raise FileNotFoundError(f"Comparison results not found: {comparison_path}")
    
    logger.info(f"Model comparison completed successfully: {comparison_path}")
    return str(comparison_path)


def update_documentation(model_path, comparison_path):
    """
    Update model documentation with latest performance metrics.
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model
    comparison_path : str
        Path to the comparison results
        
    Returns:
    --------
    docs_path : str
        Path to the updated documentation
    """
    logger.info("Updating model documentation...")
    
    # Load comparison results
    comparison_df = pd.read_csv(comparison_path)
    
    # Load model metadata
    model_path = Path(model_path)
    metadata_path = model_path.with_suffix('.json')
    
    if not metadata_path.exists():
        logger.error(f"Model metadata not found: {metadata_path}")
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Extract performance metrics
    ensemble_metrics = comparison_df[comparison_df['model_type'].str.contains('Ensemble')].iloc[0].to_dict()
    lightgbm_metrics = comparison_df[comparison_df['model_type'] == 'LightGBM'].iloc[0].to_dict() if 'LightGBM' in comparison_df['model_type'].values else None
    catboost_metrics = comparison_df[comparison_df['model_type'] == 'CatBoost'].iloc[0].to_dict() if 'CatBoost' in comparison_df['model_type'].values else None
    xgboost_metrics = comparison_df[comparison_df['model_type'] == 'XGBoost'].iloc[0].to_dict() if 'XGBoost' in comparison_df['model_type'].values else None
    
    # Format documentation
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    docs = f"""# CenanoInk Ensemble Model Documentation

## Model Overview

This document provides details about the CenanoInk LightGBM-CatBoost ensemble model
for predicting relevance of scientific literature in the nanomaterials domain.

**Last Updated:** {timestamp}

## Model Architecture

The model is an ensemble combining LightGBM and CatBoost models, optimized for the
technical fields extracted from scientific literature. It handles a mix of categorical
and numerical features with specialized preprocessing.

### Components

1. **Data Preprocessor**: Handles missing values and encoding of categorical features
2. **Feature Selection**: Uses Statistically Equivalent Signatures to select optimal features
3. **LightGBM Model**: Fast gradient boosting framework with high performance
4. **CatBoost Model**: Gradient boosting optimized for categorical features
5. **Weighted Ensemble**: Combines predictions using optimized weights

## Performance Metrics

### Ensemble Model Performance

- **Accuracy**: {ensemble_metrics.get('accuracy', 'N/A'):.4f}
- **Precision**: {ensemble_metrics.get('precision', 'N/A'):.4f}
- **Recall**: {ensemble_metrics.get('recall', 'N/A'):.4f}
- **F1 Score**: {ensemble_metrics.get('f1', 'N/A'):.4f}
- **AUC**: {ensemble_metrics.get('auc', 'N/A'):.4f}

### Component Models Performance

#### LightGBM Model

- **Accuracy**: {lightgbm_metrics.get('accuracy', 'N/A') if lightgbm_metrics else 'N/A'}
- **Precision**: {lightgbm_metrics.get('precision', 'N/A') if lightgbm_metrics else 'N/A'}
- **Recall**: {lightgbm_metrics.get('recall', 'N/A') if lightgbm_metrics else 'N/A'}
- **F1 Score**: {lightgbm_metrics.get('f1', 'N/A') if lightgbm_metrics else 'N/A'}
- **AUC**: {lightgbm_metrics.get('auc', 'N/A') if lightgbm_metrics else 'N/A'}

#### CatBoost Model

- **Accuracy**: {catboost_metrics.get('accuracy', 'N/A') if catboost_metrics else 'N/A'}
- **Precision**: {catboost_metrics.get('precision', 'N/A') if catboost_metrics else 'N/A'}
- **Recall**: {catboost_metrics.get('recall', 'N/A') if catboost_metrics else 'N/A'}
- **F1 Score**: {catboost_metrics.get('f1', 'N/A') if catboost_metrics else 'N/A'}
- **AUC**: {catboost_metrics.get('auc', 'N/A') if catboost_metrics else 'N/A'}

## Model Configuration

### Preprocessing

- **Categorical Features**: {metadata.get('categorical_features', [])}
- **Numerical Features**: {metadata.get('numerical_features', [])}

### LightGBM Configuration

```json
{metadata.get('lgbm_params', '{}')}
```

### CatBoost Configuration

```json
{metadata.get('catboost_params', '{}')}
```

### Ensemble Weights

```json
{metadata.get('model_weights', '{}')}
```

## Usage Examples

### Loading the Model

```python
from src.dashboard.model_integration import ModelIntegration

# Initialize and load model
model_integration = ModelIntegration()
model_integration.load_latest_model()

# Or load a specific model
model_integration.load_model('/path/to/model.pkl', '/path/to/metadata.json')
```

### Making Predictions

```python
import pandas as pd

# Single prediction
sample = pd.DataFrame({
    'feature1': ['value1'],
    'feature2': [0.5],
    # ... other features ...
})

prediction = model_integration.predict(sample)
print(f"Prediction: {prediction['prediction']}")
print(f"Probability: {prediction['probability']}")

# Batch predictions
batch = pd.DataFrame({
    'feature1': ['value1', 'value2', 'value3'],
    'feature2': [0.5, 0.7, 0.2],
    # ... other features ...
})

predictions = model_integration.batch_predict(batch)
```

### Explanation Features

```python
from src.dashboard.model_visualization import ModelExplainer

# Initialize explainer
explainer = ModelExplainer(model_integration)

# Explain a prediction
explanation = explainer.explain_prediction(sample)

# Visualize feature importance
fig = explainer.plot_feature_importance(sample)
```

## Deployment Information

- **Model File**: {model_path.name}
- **Training Date**: {metadata.get('training_date', 'Unknown')}
- **Records Used**: {metadata.get('n_samples', 'Unknown')}
"""
    
    # Save documentation
    docs_path = project_root / 'docs' / 'ENSEMBLE_MODEL_DOCUMENTATION.md'
    with open(docs_path, 'w') as f:
        f.write(docs)
    
    logger.info(f"Documentation updated successfully: {docs_path}")
    return str(docs_path)


def deploy_model(model_path, target_dir=None):
    """
    Deploy the model to the target directory.
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model
    target_dir : str, optional
        Target directory for deployment
        
    Returns:
    --------
    deploy_path : str
        Path to the deployed model
    """
    logger.info("Deploying model...")
    
    model_path = Path(model_path)
    metadata_path = model_path.with_suffix('.json')
    
    if not metadata_path.exists():
        logger.error(f"Model metadata not found: {metadata_path}")
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
    
    # Determine target directory
    if target_dir is None:
        target_dir = project_root / 'models' / 'deployed'
    else:
        target_dir = Path(target_dir)
    
    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy model and metadata
    target_model_path = target_dir / model_path.name
    target_metadata_path = target_dir / metadata_path.name
    
    shutil.copy2(model_path, target_model_path)
    shutil.copy2(metadata_path, target_metadata_path)
    
    # Create a symlink to the latest model
    latest_model_path = target_dir / 'latest_model.pkl'
    latest_metadata_path = target_dir / 'latest_model.json'
    
    # Remove existing symlinks if they exist
    if latest_model_path.exists():
        latest_model_path.unlink()
    if latest_metadata_path.exists():
        latest_metadata_path.unlink()
    
    # Create new symlinks
    latest_model_path.symlink_to(target_model_path)
    latest_metadata_path.symlink_to(target_metadata_path)
    
    logger.info(f"Model deployed successfully to {target_dir}")
    return str(target_model_path)


def restart_dashboard():
    """
    Restart the dashboard to use the new model.
    
    Returns:
    --------
    success : bool
        Whether the restart was successful
    """
    logger.info("Restarting dashboard...")
    
    # Check if the dashboard is running
    try:
        # Find dashboard process
        cmd = "ps aux | grep 'streamlit run' | grep -v grep"
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.stdout.strip():
            # Dashboard is running, kill it
            logger.info("Dashboard is running, restarting...")
            
            # Extract PID
            pid = result.stdout.split()[1]
            kill_cmd = f"kill -9 {pid}"
            
            # Kill process
            kill_result = subprocess.run(kill_cmd, shell=True, check=False)
            
            if kill_result.returncode != 0:
                logger.warning("Failed to kill dashboard process")
            
        # Start dashboard
        dashboard_cmd = f"nohup {sys.executable} -m streamlit run {project_root}/src/dashboard/app.py > /dev/null 2>&1 &"
        
        # Run command
        subprocess.run(dashboard_cmd, shell=True, check=False)
        
        logger.info("Dashboard restarted successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to restart dashboard: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Deploy model pipeline for CenanoInk project')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data CSV file')
    parser.add_argument('--target_col', type=str, required=True, help='Name of the target column')
    parser.add_argument('--optimize', action='store_true', help='Run hyperparameter optimization')
    parser.add_argument('--trials', type=int, default=100, help='Number of optimization trials')
    parser.add_argument('--skip_training', action='store_true', help='Skip model training')
    parser.add_argument('--skip_comparison', action='store_true', help='Skip model comparison')
    parser.add_argument('--skip_docs', action='store_true', help='Skip documentation update')
    parser.add_argument('--skip_restart', action='store_true', help='Skip dashboard restart')
    parser.add_argument('--model_path', type=str, help='Path to existing model (when skipping training)')
    parser.add_argument('--output_dir', type=str, help='Directory for output files')
    parser.add_argument('--deploy_dir', type=str, help='Directory for model deployment')
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting model deployment pipeline")
        
        # Create output directory if specified
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
        
        # Step 1: Hyperparameter optimization (if requested)
        if args.optimize:
            params_path = optimize_hyperparameters(args.data_path, args.target_col, args.trials)
            logger.info(f"Hyperparameter optimization completed: {params_path}")
        
        # Step 2: Train model (if not skipped)
        if not args.skip_training:
            model_path = train_model(args.data_path, args.target_col, args.output_dir)
            logger.info(f"Model training completed: {model_path}")
        else:
            if not args.model_path:
                logger.error("Model path must be specified when skipping training")
                sys.exit(1)
            model_path = args.model_path
            logger.info(f"Using existing model: {model_path}")
        
        # Step 3: Run model comparison (if not skipped)
        if not args.skip_comparison:
            comparison_path = run_model_comparison(args.data_path, args.target_col, args.output_dir)
            logger.info(f"Model comparison completed: {comparison_path}")
        else:
            # Try to find latest comparison
            results_dir = project_root / 'results'
            comparison_dirs = [d for d in results_dir.glob('model_comparison_*') if d.is_dir()]
            
            if comparison_dirs:
                latest_dir = max(comparison_dirs, key=os.path.getctime)
                comparison_path = str(latest_dir / 'model_comparison_metrics.csv')
                logger.info(f"Using existing comparison results: {comparison_path}")
            else:
                comparison_path = None
                logger.warning("No comparison results found, skipping documentation update")
        
        # Step 4: Update documentation (if not skipped)
        if not args.skip_docs and comparison_path:
            docs_path = update_documentation(model_path, comparison_path)
            logger.info(f"Documentation updated: {docs_path}")
        
        # Step 5: Deploy model
        deploy_path = deploy_model(model_path, args.deploy_dir)
        logger.info(f"Model deployed: {deploy_path}")
        
        # Step 6: Restart dashboard (if not skipped)
        if not args.skip_restart:
            success = restart_dashboard()
            if success:
                logger.info("Dashboard restarted successfully")
            else:
                logger.warning("Failed to restart dashboard, manual restart required")
        
        logger.info("Model deployment pipeline completed successfully")
        
    except Exception as e:
        logger.exception(f"Error in deployment pipeline: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
