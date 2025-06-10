# Ensemble Model Component Documentation

## Overview

This document describes the implementation of the LightGBM-CatBoost ensemble model designed for the CenanoInk project's filtered Open Access dataset (9,046 records, 19 technical fields).

## Key Implementation Components

### 1. Data Preprocessing Pipeline

The `TechnicalFieldsPreprocessor` class provides specialized preprocessing for the 19 technical fields:

```python
class TechnicalFieldsPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_cols=None, numerical_cols=None):
        self.categorical_cols = categorical_cols or []
        self.numerical_cols = numerical_cols or []
        self.categorical_encoders = {}
        self.numerical_scaler = RobustScaler()
        self.numerical_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
```

Key preprocessing steps:
- Categorical fields encoding with special handling for list-type fields (common in extracted technical data)
- Numerical fields scaled with `RobustScaler` to handle outliers
- Missing value imputation using appropriate strategies for each field type
- Conversion of string representations of lists to actual Python lists

### 2. Feature Selection Strategy

The `StatisticallyEquivalentSignatures` class implements a multivariate feature selection approach:

```python
class StatisticallyEquivalentSignatures:
    def __init__(self, n_features=None, n_signatures=5, random_state=42):
        self.n_features = n_features
        self.n_signatures = n_signatures
        self.random_state = random_state
        self.signatures = []
```

Key aspects:
- Uses mutual information as the base selection metric
- Creates multiple feature signatures to reduce selection bias
- Maintains a balance between stability and performance
- Particularly effective for the scientific literature domain with its specialized terminology

### 3. Model Implementation

The `CenanoinkEnsembleModel` class implements the weighted ensemble of LightGBM and CatBoost:

```python
class CenanoinkEnsembleModel:
    def __init__(self, model_params=None):
        self.model_params = model_params or {}
        self.lightgbm_model = None
        self.catboost_model = None
        self.preprocessor = None
        self.feature_selector = None
        self.label_encoder = None
        self.ensemble_weights = [0.6, 0.4]  # Default weights [LightGBM, CatBoost]
        self.best_features = None
```

Key model features:
- LightGBM as primary component for speed and efficiency
- CatBoost for superior handling of categorical features and missing values
- Weighted ensemble approach that can be dynamically optimized
- Extensive evaluation metrics with proper handling of class imbalance

### 4. Cross-Validation Framework

Stratified cross-validation is implemented to handle class imbalance:

```python
def cross_validate(self, X, y, cv=5):
    # Encode target if needed
    if np.issubdtype(y.dtype, np.object_):
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    else:
        y_encoded = y
    
    # Setup stratified cross-validation
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Perform cross-validation
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        
        # Train and evaluate model on each fold
        ...
```

### 5. Overfitting Prevention

Multiple strategies implemented to prevent overfitting:

1. **Monitoring**: The `check_for_overfitting` method compares train and validation performance
2. **Regularization**: Appropriate regularization parameters in both models
3. **Feature selection**: Reduces dimensionality to prevent overfitting
4. **Cross-validation**: Ensures generalizability of the model

### 6. Dashboard Integration

The `model_integration.py` module facilitates seamless integration with the dashboard:

```python
class ModelIntegration:
    def __init__(self):
        self.model = None
        self.metadata = None
        self.model_path = None
    
    def load_latest_model(self, models_dir=None):
        ...
    
    def predict(self, data):
        ...
    
    def explain_prediction(self, data):
        ...
```

Key integration features:
- Simple API for loading models
- Prediction methods tailored for dashboard use
- Explanation functionality for interpretability
- Performance metrics accessible for visualization

## Model Configuration Details

### LightGBM Configuration

```python
lightgbm_params = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'verbose': -1,
    'n_jobs': -1,
    'n_estimators': 500
}
```

### CatBoost Configuration

```python
catboost_params = {
    'iterations': 500,
    'learning_rate': 0.05,
    'depth': 6,
    'loss_function': 'MultiClass',
    'random_seed': 42,
    'verbose': False,
    'allow_writing_files': False,
    'early_stopping_rounds': 50
}
```

## Performance Evaluation Metrics

The implementation includes comprehensive evaluation metrics:

1. **Accuracy**: Overall classification accuracy
2. **Precision, Recall, F1 (weighted)**: Accounts for class imbalance
3. **Confusion Matrix**: Detailed class-level performance analysis
4. **Cross-Validation Results**: Ensures generalizability of the model
5. **Overfitting Analysis**: Detects and alerts on model overfitting

## Usage Example

```python
# Load and prepare filtered Open Access dataset
df, categorical_cols, numerical_cols = load_and_prepare_data("filtered_open_access.csv")

# Initialize model
model = CenanoinkEnsembleModel()

# Train model
model.fit(X_train, y_train, 
          categorical_cols=categorical_cols,
          numerical_cols=numerical_cols,
          optimize_weights=True)

# Evaluate model
results = model.evaluate(X_test, y_test)

# Save model for dashboard integration
model.save_model("models/trained/")
```

## Integration with Dashboard

```python
# In dashboard code:
from src.dashboard.model_integration import model_integration

# Load model
model_integration.load_latest_model()

# Make prediction on new data
prediction = model_integration.predict(new_data)

# Get model information for visualization
model_info = model_integration.get_model_info()
```

## Conclusion

This implementation provides an optimized ensemble model specifically designed for the CenanoInk project's filtered Open Access dataset, balancing accuracy and computational efficiency while maintaining interpretability for scientific applications.
