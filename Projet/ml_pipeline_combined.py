import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, make_scorer
)
import optuna
import emoji
import re
import scipy.sparse
import spacy
from langdetect import detect

class ModelPipeline:
    def __init__(self, name="ModelPipeline"):
        self.name = name
        self.models = {
            "Naïve Bayes": MultinomialNB(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(kernel='linear'),
            "KNN": KNeighborsClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            "Neural Network - MLP": MLPClassifier(max_iter=300),
        }
        self.scorers = {
            "Accuracy": make_scorer(accuracy_score),
            "Precision": make_scorer(precision_score, average='weighted'),
            "Recall": make_scorer(recall_score, average='weighted'),
            "F1 Score": make_scorer(f1_score, average='weighted')
        }
        self.results = {}

    def preprocess_data(self, df, with_preprocessing=False):
        """Preprocess data based on the phase (with or without preprocessing)"""
        # Convert emojis
        df['text'] = df['text'].apply(lambda x: emoji.demojize(x))
        
        # Add binary features
        df['has_url'] = df['text'].str.contains(r'http[s]?://', regex=True)
        df['has_mention'] = df['text'].str.contains(r'@\w+', regex=True)
        df['has_hashtag'] = df['text'].str.contains(r'#\w+', regex=True)
        df['has_emoji'] = df['text'].str.contains(r':[^:\s]+:')
        
        if with_preprocessing:
            # Advanced preprocessing for Phase 2
            try:
                nlp_fr = spacy.load("fr_core_news_sm")
                nlp_en = spacy.load("en_core_web_sm")
            except OSError:
                print("Installing spaCy models...")
                import os
                os.system("python -m spacy download fr_core_news_sm")
                os.system("python -m spacy download en_core_web_sm")
                nlp_fr = spacy.load("fr_core_news_sm")
                nlp_en = spacy.load("en_core_web_sm")
            
            def clean_and_lemmatize_multilang(text):
                text = text.lower()
                text = re.sub(r"http\S+", "URL", text)
                text = re.sub(r"@\w+", "MENTION", text)
                text = re.sub(r"#(\w+)", r"\1", text)
                text = re.sub(r"[^\w\s:]", "", text)
                try:
                    lang = detect(text)
                except:
                    lang = "en"
                doc = nlp_fr(text) if lang == "fr" else nlp_en(text)
                return " ".join([token.lemma_ for token in doc if not token.is_stop])
            
            df['text_clean'] = df['text'].apply(clean_and_lemmatize_multilang)
            text_column = 'text_clean'
        else:
            # Simple preprocessing for Phase 1
            text_column = 'text'

        return df, text_column

    def prepare_features(self, df, text_column, stopwords=None):
        """Prepare feature matrix from preprocessed data"""
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(
            min_df=5,
            max_df=0.95,
            stop_words=stopwords,
            lowercase=True,
            ngram_range=(1, 2)
        ) if stopwords else TfidfVectorizer()
        
        X_text = df[text_column]
        X_tfidf = vectorizer.fit_transform(X_text)
        
        # Binary features
        extra_features = df[['has_url', 'has_mention', 'has_hashtag', 'has_emoji']].astype(int)
        X_extra = scipy.sparse.csr_matrix(extra_features.values)
        
        # Combine features
        X_final = scipy.sparse.hstack([X_tfidf, X_extra])
        
        return X_final, vectorizer

    def train_and_evaluate(self, X, y, phase_name):
        """Train and evaluate models, saving results to CSV"""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = []
        
        for name, model in self.models.items():
            print(f"\n🔍 Training model: {name}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                "Model": name,
                "Phase": phase_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average='weighted'),
                "Recall": recall_score(y_test, y_pred, average='weighted'),
                "F1": f1_score(y_test, y_pred, average='weighted')
            }
            
            # Cross-validation
            for metric_name, scorer in self.scorers.items():
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
                metrics[f"CV_{metric_name}_Mean"] = cv_scores.mean()
                metrics[f"CV_{metric_name}_Std"] = cv_scores.std()
            
            results.append(metrics)
            
            # Plot confusion matrix
            plt.figure(figsize=(6, 5))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {name} ({phase_name})')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(f'confusion_matrix_{phase_name}_{name.lower().replace(" ", "_")}.png')
            plt.close()
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'model_results_{phase_name.lower().replace(" ", "_")}.csv', index=False)
        return results_df

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('scitweets_balanced.tsv', sep='\t')
    df.dropna(subset=['text'], inplace=True)
    y = df['science_related']
    
    # Initialize pipeline
    pipeline = ModelPipeline()
    
    # Phase 1: Without preprocessing
    print("\n=== Phase 1: Without Preprocessing ===")
    df_phase1, text_column = pipeline.preprocess_data(df.copy(), with_preprocessing=False)
    X_phase1, _ = pipeline.prepare_features(df_phase1, text_column)
    results_phase1 = pipeline.train_and_evaluate(X_phase1, y, "Without_Preprocessing")
    
    # Phase 2: With preprocessing
    print("\n=== Phase 2: With Preprocessing ===")
    df_phase2, text_column = pipeline.preprocess_data(df.copy(), with_preprocessing=True)
    # Load French stopwords
    stopwords_df = pd.read_csv("StopWordsFrench.csv", sep=',', index_col=0)
    french_stopwords = stopwords_df.index.tolist()
    X_phase2, _ = pipeline.prepare_features(df_phase2, text_column, french_stopwords)
    results_phase2 = pipeline.train_and_evaluate(X_phase2, y, "With_Preprocessing")
    
    print("\nResults have been saved to CSV files and confusion matrices have been plotted.")

if __name__ == "__main__":
    main()
