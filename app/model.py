import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import io

def prepare_data(df, target_col, classification=False):
    """
    Prepare data for modeling by handling missing values and scaling features
    """
    # Create feature matrix X and target vector y
    X = df.drop(['MR VC mm', 'MR area cm2', 'Patient ID', 'Cycle', 'Frame', 'Time (moment)'], axis=1)
    y = df[target_col]
    
    if classification:
        # For classification, create binary classes based on median value
        y = (y > y.median()).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names, target_col, classification=False):
    """
    Train and evaluate Random Forest and Gradient Boosting models with hyperparameter tuning
    """
    if classification:
        # Random Forest Classifier parameters
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10]
        }
        rf_model = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5)
        
    else:
        # Random Forest Regressor parameters
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10]
        }
        rf_model = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=5)
        
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    rf_pred = rf_model.predict(X_test)
    
    plots = {}

    if classification:
        accuracy = accuracy_score(y_test, rf_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, rf_pred, average='binary')
        metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1 Score: {f1:.3f}'
    else:
        mse = mean_squared_error(y_test, rf_pred)
        r2 = r2_score(y_test, rf_pred)
        metrics_text = f'MSE: {mse:.3f}\nR2 Score: {r2:.3f}'

    # Feature importance plot
    plt.figure(figsize=(12, 6))
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.best_estimator_.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title(f'Feature Importance for {target_col}\n({rf_model.__class__.__name__})')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    plots[f'{target_col}_feature_importance'] = buffer.getvalue()
    
    # Actual vs Predicted plot with metrics
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test, alpha=0.5, color='green', label='Actual Values')
    plt.scatter(y_test, rf_pred, alpha=0.5, color='blue', label='Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    
    # Add metrics text box
    plt.text(0.05, 0.95, metrics_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted Values for {target_col}', pad=20)
    plt.legend()
    plt.tight_layout();
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    plots[f'{target_col}_actual_vs_predicted'] = buffer.getvalue()
    
    return plots
