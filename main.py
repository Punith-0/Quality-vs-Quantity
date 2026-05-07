import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


imdb_df = pd.read_csv("imdb_dataset.csv")
twitter_df = pd.read_csv("twitter_dataset.csv")

datasets = {
    "imdb": imdb_df,
    "twitter": twitter_df
}

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text, method):
    text = re.sub(r'<.*?>', '', str(text))
    words = word_tokenize(text.lower())
    
    processed_words = []
    
    for w in words:
        if not w.isalpha():
            continue
        if len(w) < 3:
            continue
        
        if method in ["stopwords", "both"]:
            if w in stop_words:
                continue
        
        if method in ["lemmatization", "both"]:
            w = lemmatizer.lemmatize(w)
        
        processed_words.append(w)
    
    return " ".join(processed_words)


methods = ["raw", "stopwords", "lemmatization", "both"]
sizes = [0.2, 0.5, 0.8, 0.99]
seeds = [1, 2, 3, 4, 5]

models = {
    "logreg": LogisticRegression(max_iter=200),
    "svm": LinearSVC(),
    "nb": MultinomialNB()
}

results = []

for dataset_name, df in datasets.items():
    
    texts = df["text"].astype(str).tolist()
    labels = df["sentiment"]
    labels = labels.astype(str).str.lower().str.strip()

    labels = labels.map({"positive": 1, "negative": 0, "1": 1, "0": 0})
    valid= labels.notna()
    texts = [t for t, v in zip(texts, valid) if v]
    labels = labels[valid].astype(int).tolist()
    
    for method in methods:
        print(f"Dataset: {dataset_name} | Method: {method}")
        
        processed_texts = [preprocess(t, method) for t in texts]
        
        for size in sizes:
            for seed in seeds:
                
                X_subset, _, y_subset, _ = train_test_split(
                    processed_texts,
                    labels,
                    train_size=size,
                    random_state=seed
                )
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_subset,
                    y_subset,
                    test_size=0.2,
                    random_state=seed
                )
                
                vectorizer = TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1,2)
                )
                
                X_train_vec = vectorizer.fit_transform(X_train)
                X_test_vec = vectorizer.transform(X_test)
                
                for model_name, model in models.items():
                    
                    model.fit(X_train_vec, y_train)
                    preds = model.predict(X_test_vec)
                    
                    acc = accuracy_score(y_test, preds)
                    f1 = f1_score(y_test, preds)
                    
                    results.append({
                        "dataset": dataset_name,
                        "model": model_name,
                        "method": method,
                        "data_size": size,
                        "seed": seed,
                        "accuracy": acc,
                        "f1": f1
                    })

df_results = pd.DataFrame(results)

final_df = df_results.groupby(
    ["dataset", "model", "method", "data_size"]
).agg({
    "accuracy": ["mean", "std"],
    "f1": ["mean", "std"]
}).reset_index()

print("\nFinal Aggregated Results:\n")
print(final_df)

final_df.to_csv("final_results.csv", index=False)

summary = final_df.copy()

summary.columns = ['dataset','model','method','data_size',
                   'acc_mean','acc_std','f1_mean','f1_std']

summary = summary.groupby(['dataset','method','data_size'])['acc_mean'].mean().reset_index()

summary = summary.sort_values(['dataset','data_size','acc_mean'], ascending=[True, True, False])

summary = summary.groupby(['dataset','data_size']).first().reset_index()

print("\nBest Method per Data Size:\n")
print(summary)
summary.to_csv("results_summary.csv", index=False)

import matplotlib.pyplot as plt

final_df.columns = ['dataset', 'model', 'method', 'data_size',
                    'acc_mean', 'acc_std', 'f1_mean', 'f1_std']

for dataset_name in final_df["dataset"].unique():
    
    for method in final_df["method"].unique():
        
        plt.figure(figsize=(7,5))
        
        subset = final_df[
            (final_df["dataset"] == dataset_name) &
            (final_df["method"] == method)
        ]
        
        for model in subset["model"].unique():
            model_data = subset[subset["model"] == model].sort_values("data_size")
            
            plt.plot(
                model_data["data_size"],
                model_data["acc_mean"],
                marker='o',
                linewidth=2,
                label=model
            )
            
            # optional: error shading
            plt.fill_between(
                model_data["data_size"],
                model_data["acc_mean"] - model_data["acc_std"],
                model_data["acc_mean"] + model_data["acc_std"],
                alpha=0.15
            )
        
        plt.xlabel("Training Data Size")
        plt.ylabel("Accuracy")
        plt.title(f"{dataset_name.upper()} - {method} preprocessing")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.show()