import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, KFold, learning_curve
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Define the Result class to store evaluation results
class Result:
    def __init__(self, name, params, scoremean, stdresult, timespent, precision, recall, f1):
        self.name = name
        self.params = params
        self.scoremean = scoremean
        self.stdresult = stdresult
        self.timespent = timespent
        self.precision = precision
        self.recall = recall
        self.f1 = f1

def test_xgboost_hyperparameters(X, y, param_grid, text_column=None):
    """
    Test different hyperparameters for XGBoost classifier and determine the best configuration.
    
    Parameters:
    -----------
    X : array-like or pd.Series
        Features for classification. If text_column is provided, this should be a pandas Series or list of text.
        Otherwise, it should be preprocessed feature vectors.
    y : array-like
        Target labels for classification.
    param_grid : dict
        Dictionary of XGBoost parameters to test. Each key is a parameter name, and each value is a list of
        values to test for that parameter. Example:
        {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 200]
        }
    text_column : bool, optional
        If True, X is treated as text data and TF-IDF vectorization is applied. Default is None.
    
    Returns:
    --------
    allresults : list
        List of Result objects containing evaluation metrics for each parameter combination,
        sorted by accuracy score in descending order.
    """
    seed = 7
    allresults = []
    
    # Prepare scoring metrics
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1': make_scorer(f1_score, average='weighted')
    }
    
    # Generate all parameter combinations
    param_combinations = []
    param_names = list(param_grid.keys())
    
    def generate_combinations(current_combo, index):
        if index == len(param_names):
            param_combinations.append(current_combo.copy())
            return
        
        param_name = param_names[index]
        param_values = param_grid[param_name]
        
        for value in param_values:
            current_combo[param_name] = value
            generate_combinations(current_combo, index + 1)
    
    generate_combinations({}, 0)
    
    # Create cross-validation object
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    
    # Test each parameter combination
    for i, params in enumerate(param_combinations):
        # Create model name based on parameters
        model_name = "XGBoost"
        for param, value in params.items():
            model_name += f"_{param}={value}"
        
        print(f"Evaluating {model_name} ({i+1}/{len(param_combinations)})")
        
        # Create the XGBoost classifier with current parameters
        if text_column is not None:
            # Create a pipeline with TF-IDF and XGBoost
            model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('xgb', XGBClassifier(**params, random_state=seed, use_label_encoder=False))
            ])
        else:
            # Use XGBoost directly on preprocessed features
            model = XGBClassifier(**params, random_state=seed, use_label_encoder=False)
        
        # Perform cross-validation
        start_time = time.time()
        cv_results = cross_validate(model, X, y, cv=kfold, scoring=scoring)
        thetime = time.time() - start_time
        
        # Store results
        result = Result(
            name=model_name,
            params=str(params),
            scoremean=cv_results['test_accuracy'].mean(),
            stdresult=cv_results['test_accuracy'].std(),
            timespent=thetime,
            precision=cv_results['test_precision'].mean(),
            recall=cv_results['test_recall'].mean(),
            f1=cv_results['test_f1'].mean()
        )
        
        allresults.append(result)
        
        # Print current results
        print(f"  Accuracy: {result.scoremean:.3f} (±{result.stdresult:.3f})")
        print(f"  Precision: {result.precision:.3f}, Recall: {result.recall:.3f}, F1: {result.f1:.3f}")
        print(f"  Time: {thetime:.3f}s\n")
    
    # Sort results by accuracy score
    allresults = sorted(allresults, key=lambda result: result.scoremean, reverse=True)
    
    # Print all results
    print('\nAll Results (sorted by accuracy):')
    print('=' * 80)
    for i, result in enumerate(allresults):
        print(f"{i+1}. {result.name}")
        print(f"   Parameters: {result.params}")
        print(f"   Accuracy: {result.scoremean:.3f} (±{result.stdresult:.3f})")
        print(f"   Precision: {result.precision:.3f}, Recall: {result.recall:.3f}, F1: {result.f1:.3f}")
        print(f"   Time: {result.timespent:.3f}s")
        print('-' * 80)
    
    # Print best result
    print('\nBest Result:')
    print('=' * 80)
    print(f"Model: {allresults[0].name}")
    print(f"Parameters: {allresults[0].params}")
    print(f"Accuracy: {allresults[0].scoremean:.3f} (±{allresults[0].stdresult:.3f})")
    print(f"Precision: {allresults[0].precision:.3f}, Recall: {allresults[0].recall:.3f}, F1: {allresults[0].f1:.3f}")
    print(f"Time: {allresults[0].timespent:.3f}s")
    
    # Save results to CSV
    save_results_to_csv(allresults, 'xgboost_hyperparameter_results.csv')
    
    # Plot feature importance for the best model if not using text data
    if text_column is None:
        plot_feature_importance(model, X)
    
    return allresults

def save_results_to_csv(allresults, filename='xgboost_hyperparameter_results.csv'):
    """Save the results to a CSV file."""
    data = {
        'Model Name': [result.name for result in allresults],
        'Parameters': [result.params for result in allresults],
        'Accuracy': [result.scoremean for result in allresults],
        'Std Dev': [result.stdresult for result in allresults],
        'Precision': [result.precision for result in allresults],
        'Recall': [result.recall for result in allresults],
        'F1 Score': [result.f1 for result in allresults],
        'Execution Time (s)': [result.timespent for result in allresults]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f'\nResults saved to {filename}')

def plot_feature_importance(model, X):
    """Plot feature importance for XGBoost model."""
    if hasattr(model, 'named_steps'):
        # If using pipeline, get the XGBoost model
        xgb_model = model.named_steps['xgb']
    else:
        xgb_model = model
    
    # Get feature importance
    importance = xgb_model.feature_importances_
    
    # Create feature names if not available
    feature_names = getattr(X, 'columns', [f'feature_{i}' for i in range(len(importance))])
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance_df.head(20))
    plt.title('XGBoost Feature Importance (Top 20)')
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance.png')
    plt.close()

def plot_learning_curves(model, X, y, cv=5):
    """Plot learning curves for the model."""
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    # Calculate learning curves
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=cv,
        n_jobs=-1,
        scoring='accuracy'
    )
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.plot(train_sizes, val_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    
    plt.xlabel('Training Examples')
    plt.ylabel('Accuracy Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('xgboost_learning_curves.png')
    plt.close()

# Example usage
def main():
    # Load data
    file_path = "scitweets_export.tsv"
    df = pd.read_csv(file_path, sep='\t')
    
    # Assuming 'text' is the column with tweet content and 'label' is the target
    X = df['text']  # Replace with your actual text column name
    y = df['label']  # Replace with your actual label column name
    
    # Define parameter grid to test
    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 3]
    }
    
    # Run hyperparameter testing
    results = test_xgboost_hyperparameters(X, y, param_grid, text_column=True)
    
    return results

if __name__ == "__main__":
    main()
