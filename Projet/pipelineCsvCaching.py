import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, cross_validate
import time

class Result:
    def __init__(self, name, scoremean, stdresult, timespent, params=None, precision=None, recall=None, f1=None):
        self.name = name
        self.scoremean = scoremean
        self.stdresult = stdresult
        self.timespent = timespent
        self.params = params
        self.precision = precision
        self.recall = recall
        self.f1 = f1



def MyTestPipelines(models, X, y, score='accuracy', save_csv=True, filename='model_results.csv'):
    seed = 7
    allresults = []
    results = []
    names = []
    
    # Define the scoring metrics to calculate
    scoring = {'accuracy': 'accuracy', 
               'precision': 'precision_weighted',
               'recall': 'recall_weighted', 
               'f1': 'f1_weighted'}
    
    for name, model in models:
        print(f"Evaluation de {name}")
        
        # Cross validation
        kfold = KFold(n_splits=10, random_state=seed)
        start_time = time.time()
        
        # Use cross_validate to get multiple metrics
        cv_results = cross_validate(model, X, y, cv=kfold, scoring=scoring)
        
        thetime = time.time() - start_time
        
        # Create result object with all metrics
        result = Result(
            name=name,
            scoremean=cv_results['test_accuracy'].mean(),
            stdresult=cv_results['test_accuracy'].std(),
            timespent=thetime,
            params=str(model.get_params()),
            precision=cv_results['test_precision'].mean(),
            recall=cv_results['test_recall'].mean(),
            f1=cv_results['test_f1'].mean()
        )
        
        allresults.append(result)
        results.append(cv_results['test_accuracy'])
        names.append(name)
        
        print(f"{name} : {cv_results['test_accuracy'].mean():.3f} ({cv_results['test_accuracy'].std():.3f}) in {thetime:.3f} s")
    
    # Sort by the main score (accuracy by default)
    allresults = sorted(allresults, key=lambda result: result.scoremean, reverse=True)
    
    # Display the best result
    print('\nLe meilleur resultat : ')
    print(f'Classifier : {allresults[0].name} {score} : {allresults[0].scoremean:.3f} ({allresults[0].stdresult:.3f}) en {allresults[0].timespent:.3f} s\n')
    
    # Display all results
    print('Tous les résultats : \n')
    for result in allresults:
        print(f'Classifier : {result.name} {score} : {result.scoremean:.3f} ({result.stdresult:.3f}) en {result.timespent:.3f} s')
    
    # Save results to CSV if requested
    if save_csv:
        save_results_to_csv(allresults, filename)
    
    return allresults

def save_results_to_csv(allresults, filename='model_results.csv'):
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
