# Machine Learning Tweet Classification Pipeline

This project implements a machine learning pipeline for classifying tweets as science-related or not, with two distinct phases of preprocessing.

## Features

- **Two-phase testing pipeline:**
  1. Without preprocessing (basic TF-IDF)
  2. With advanced preprocessing (multilingual lemmatization, stopwords)
- **Automated model evaluation** with multiple metrics
- **Results export** to CSV files
- **Visualization** of confusion matrices
- **Cross-validation** for robust evaluation

## Project Structure

- `ml_pipeline_combined.py`: Main pipeline implementation
- `model_results_without_preprocessing.csv`: Results from Phase 1
- `model_results_with_preprocessing.csv`: Results from Phase 2
- `confusion_matrix_*.png`: Generated confusion matrix plots

## Models Tested

1. Naive Bayes
2. Random Forest
3. SVM (Linear kernel)
4. K-Nearest Neighbors
5. AdaBoost
6. XGBoost
7. Neural Network (MLP)

## Metrics Tracked

- Accuracy
- Precision
- Recall
- F1 Score
- Cross-validation scores (mean and std) for all metrics

## Prerequisites

```bash
pip install pandas numpy scikit-learn seaborn matplotlib emoji spacy langdetect xgboost optuna
python -m spacy download fr_core_news_sm
python -m spacy download en_core_web_sm
```

## Usage

1. Ensure all required data files are present:
   - `scitweets_balanced.tsv`
   - `StopWordsFrench.csv`

2. Run the pipeline:
```bash
python ml_pipeline_combined.py
```

3. Check the results in:
   - Generated CSV files
   - Confusion matrix plots

## Pipeline Phases

### Phase 1: Without Preprocessing
- Basic TF-IDF vectorization
- Binary features (URLs, mentions, hashtags, emojis)
- No text cleaning or lemmatization

### Phase 2: With Preprocessing
- Advanced text cleaning
- Multilingual lemmatization (French/English)
- Stopwords removal
- N-gram features
- Enhanced TF-IDF parameters
