# 🧠 To Remove or Not to Remove

### The Real Impact of Text Cleaning and Data Scale in Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---
## 📌 Overview

This project investigates whether common text preprocessing techniques like stopword removal and lemmatization actually improve classification performance.

Instead of assuming preprocessing is always beneficial, this study evaluates:

* different text cleaning strategies
* varying training data sizes
* multiple machine learning models
* across multiple datasets

---

## 🎯 Objectives

* Analyze the impact of preprocessing on model accuracy
* Study how training data size affects performance
* Compare multiple machine learning models
* Determine whether preprocessing is necessary or optional

---

## 📊 Datasets

* IMDB Movie Reviews (10,000 samples)
* Twitter Dataset (10,000 samples)

Each dataset contains:

* text data
* binary labels

---

## ⚙️ Methodology

### 🔹 Preprocessing Methods

* Raw (no preprocessing)
* Stopword Removal
* Lemmatization
* Stopwords + Lemmatization

### 🔹 Feature Extraction

* TF-IDF Vectorization
* n-grams (1,2)

### 🔹 Models Used

* Logistic Regression
* Support Vector Machine (SVM)
* Naive Bayes

### 🔹 Experiment Setup

* Training sizes: 20%, 50%, 80%, 99%
* Test size: 20%
* 5 random seeds for consistency

### 🔹 Evaluation Metrics

* Accuracy
* F1 Score

---

## 📈 Results

* SVM consistently performed best across datasets
* Preprocessing had minimal impact on performance
* Raw text often performed equally or better than cleaned text
* Increasing training data improved accuracy significantly
* Performance plateaued at higher data sizes

---

## 📊 Key Insight

> Text cleaning is not always beneficial.
> Data quantity and model choice have a stronger impact on performance.

---

## 🖼️ Visualizations

* Accuracy vs Training Data Size
* Model Comparison Graphs
* Consolidated Preprocessing Comparison

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python main.py
```

---

## 📂 Project Structure

```
├── main.py
├── imdb_dataset.csv
├── twitter_dataset.csv
├── final_results.csv
```

---

## 📚 References

* scikit-learn documentation
* NLTK documentation
* Public datasets (IMDB, Twitter)

---

## 👨‍💻 Author

* Punith Dewangan 

---

## ⭐ Notes

This project challenges the common assumption that preprocessing is always necessary and highlights the importance of evaluating each step empirically.
