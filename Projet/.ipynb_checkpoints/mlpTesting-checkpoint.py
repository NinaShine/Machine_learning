import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate, KFold, learning_curve
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

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

def plot_neural_network(hidden_layer_sizes):
    """Plot the neural network architecture."""
    plt.figure(figsize=(10, 8))
    
    # Calculate the maximum number of nodes in any layer
    max_nodes = max(max(hidden_layer_sizes), 10)  # 10 is placeholder for input/output
    
    # Create positions for each layer
    n_layers = len(hidden_layer_sizes) + 2  # +2 for input and output layers
    layer_positions = np.linspace(0, 1, n_layers)
    
    # Plot input layer (placeholder with 10 nodes)
    y_positions = np.linspace(0, 1, 10)
    plt.scatter([0]*10, y_positions, c='b', s=100, label='Input Layer')
    
    # Plot hidden layers
    for i, n_nodes in enumerate(hidden_layer_sizes):
        y_positions = np.linspace(0, 1, n_nodes)
        plt.scatter([layer_positions[i+1]]*n_nodes, y_positions, c='r', s=100)
        
        # Draw connections from previous layer
        if i == 0:
            prev_nodes = 10  # Input layer nodes
        else:
            prev_nodes = hidden_layer_sizes[i-1]
        
        # Draw a subset of connections to avoid cluttering
        for j in range(n_nodes):
            for k in range(prev_nodes):
                if (j + k) % 2 == 0:  # Draw only some connections
                    plt.plot([layer_positions[i], layer_positions[i+1]], 
                           [y_positions[k], y_positions[j]], 
                           'gray', alpha=0.1)
    
    # Plot output layer (placeholder with 2 nodes for binary classification)
    plt.scatter([1]*2, [0.3, 0.7], c='g', s=100, label='Output Layer')
    
    # Draw connections to output layer
    for i in range(2):
        for j in range(hidden_layer_sizes[-1]):
            if (i + j) % 2 == 0:
                plt.plot([layer_positions[-2], 1], 
                       [y_positions[j], [0.3, 0.7][i]], 
                       'gray', alpha=0.1)
    
    plt.title('Neural Network Architecture')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('mlp_architecture.png')
    plt.close()

def test_mlp_hyperparameters(X, y, param_grid, text_column=None):
    """
    Test different hyperparameters for MLP classifier and determine the best configuration.
    
    Parameters:
    -----------
    X : array-like or pd.Series
        Features for classification. If text_column is provided, this should be a pandas Series or list of text.
        Otherwise, it should be preprocessed feature vectors.
    y : array-like
        Target labels for classification.
    param_grid : dict
        Dictionary of MLP parameters to test. Each key is a parameter name, and each value is a list of
        values to test for that parameter. Example:
        {
            'hidden_layer_sizes': [(100,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001]
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
    
    # Test each parameter combination with progress bar
    for i, params in enumerate(tqdm(param_combinations)):
        # Create model name based on parameters
        model_name = "MLP"
        for param, value in params.items():
            model_name += f"_{param}={value}"
        
        print(f"\nEvaluating {model_name} ({i+1}/{len(param_combinations)})")
        
        # Plot neural network architecture for this configuration
        if 'hidden_layer_sizes' in params:
            plot_neural_network(params['hidden_layer_sizes'])
        
        # Create the MLP classifier with current parameters
        if text_column is not None:
            # Create a pipeline with TF-IDF and MLP
            model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('mlp', MLPClassifier(random_state=seed, **params, early_stopping=True, validation_fraction=0.1))
            ])
        else:
            # Use MLP directly on preprocessed features
            model = MLPClassifier(random_state=seed, **params, early_stopping=True, validation_fraction=0.1)
        
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
    
    # Plot learning curves for the best model
    plot_learning_curves(model, X, y)
    
    # Save results to CSV
    save_results_to_csv(allresults, 'mlp_hyperparameter_results.csv')
    
    return allresults

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
    
    # Plot learning curves
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
    plt.savefig('mlp_learning_curves.png')
    plt.close()

def save_results_to_csv(allresults, filename='mlp_hyperparameter_results.csv'):
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
        'hidden_layer_sizes': [(100,), (50, 50), (100, 50), (100, 100)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [300],
        'alpha': [0.0001, 0.001, 0.01]
    }
    
    # Run hyperparameter testing
    results = test_mlp_hyperparameters(X, y, param_grid, text_column=True)
    
    return results

if __name__ == "__main__":
    main()
