import pandas as pd
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, KFold
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

def test_svm_hyperparameters(X, y, param_grid, text_column=None):
    """
    Test different hyperparameters for SVM classifier and determine the best configuration.
    
    Parameters:
    -----------
    X : array-like or pd.Series
        Features for classification. If text_column is provided, this should be a pandas Series or list of text.
        Otherwise, it should be preprocessed feature vectors.
    y : array-like
        Target labels for classification.
    param_grid : dict
        Dictionary of SVM parameters to test. Each key is a parameter name, and each value is a list of
        values to test for that parameter. Example:
        {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto', 0.1, 0.01]
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
        model_name = "SVM"
        for param, value in params.items():
            model_name += f"_{param}={value}"
        
        print(f"Evaluating {model_name} ({i+1}/{len(param_combinations)})")
        
        # Create the SVM classifier with current parameters
        if text_column is not None:
            # Create a pipeline with TF-IDF and SVM
            model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('svm', SVC(**params))
            ])
        else:
            # Use SVM directly on preprocessed features
            model = SVC(**params)
        
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
    save_results_to_csv(allresults, 'svm_hyperparameter_results.csv')
    
    return allresults

def save_results_to_csv(allresults, filename='svm_hyperparameter_results.csv'):
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
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    
    # Run hyperparameter testing
    results = test_svm_hyperparameters(X, y, param_grid, text_column=True)
    
    return results

if __name__ == "__main__":
    main()
