# 🧠 More Data or More Cleaning 

### The Real Impact of Text Cleaning and Data Scale in Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Research](https://img.shields.io/badge/Type-Research-blueviolet)
![Project](https://img.shields.io/badge/Project-ML-green)
![Academic](https://img.shields.io/badge/Use-Academic-orange)


## 📌 Overview

This project investigates whether common text preprocessing techniques such as stopword removal and lemmatization consistently improve sentiment-classification performance.

Instead of assuming that text preprocessing is always beneficial, the study empirically evaluates the effect of preprocessing under different experimental conditions.

The experiments examine:

* Different text preprocessing strategies
* Different training-data sizes
* Multiple machine-learning models
* Multiple sentiment datasets

The goal is to determine whether additional text cleaning provides meaningful performance improvements and whether its computational cost is justified.

---

## 🎯 Objectives

The main objectives of the project are:

* Analyze the impact of text preprocessing on classification performance
* Study how training-data size affects model performance
* Compare multiple machine-learning models
* Evaluate preprocessing across different datasets
* Determine whether preprocessing consistently provides measurable benefits
* Examine the interaction between preprocessing, model choice, and training-data scale

---

## 📊 Datasets

The experiments use two sentiment-classification datasets.

### IMDb Movie Reviews

* 10,000 samples used in the experiments
* Movie-review text
* Binary sentiment labels

### Twitter Dataset

* 10,000 samples used in the experiments
* Short-form social-media text
* Binary sentiment labels

Using both datasets allows the study to compare preprocessing behavior across different types of text.

---

## ⚙️ Methodology

### 🔹 Preprocessing Methods

Four preprocessing configurations are evaluated:

1. Raw Text
2. Stopword Removal
3. Lemmatization
4. Stopword Removal + Lemmatization

The raw-text configuration provides a baseline against which the preprocessing methods can be compared.

---

### 🔹 Feature Extraction

Text is converted into numerical features using:

* TF-IDF Vectorization
* Unigrams
* Bigrams

The resulting sparse feature representation is used as input to the machine-learning models.

---

### 🔹 Models Used

The study evaluates three machine-learning classifiers:

* Logistic Regression
* Linear Support Vector Machine (SVM)
* Multinomial Naive Bayes

Using multiple classifiers allows the effect of preprocessing to be examined across different learning algorithms.

---

### 🔹 Experiment Setup

The experiments evaluate different training-data conditions:

* 20%
* 50%
* 80%
* 99%

A test size of **20%** is used.

Multiple random seeds are used to reduce the influence of a single random split and improve the consistency of the experimental comparison.

---

### 🔹 Evaluation Metrics

The models are evaluated using:

* Accuracy
* F1 Score

Both metrics are considered when comparing different experimental configurations.

---

## 📈 Results

The experiments examine the relationship between preprocessing, training-data scale, model choice, and classification performance.

The main observations from the experiments include:

* SVM achieved strong performance across the evaluated datasets and configurations.
* Preprocessing generally produced relatively small changes in classification performance.
* Raw text frequently achieved performance comparable to or better than additional preprocessing.
* Increasing the amount of training data generally improved classification performance.
* Performance improvements became smaller at higher training-data levels.

The detailed numerical results and visualizations are documented in the project repository and Wiki.

---

## 📊 Key Insight

> Text preprocessing is not automatically beneficial in every sentiment-classification setting.

The study highlights the importance of evaluating preprocessing empirically rather than treating text cleaning as a universally necessary step.

Training-data scale and model selection can also have a substantial influence on classification performance.

---

## 🖼️ Visualizations

The project includes visual analysis of the experimental results, including:

* Accuracy vs. Training Data Size
* Model Comparison
* Preprocessing Comparison
* Consolidated Experimental Results

These visualizations help compare the effect of preprocessing and training-data scale across models and datasets.

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Punith-0/Quality-vs-Quantity.git
cd Quality-vs-Quantity
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the experiment

```bash
python main.py
```

The exact execution flow may depend on the current version of the repository.

Refer to the project Wiki for detailed experimental and reproducibility instructions.

---

## 📂 Project Structure

The repository contains the main components required for:

```text
Dataset
   ↓
Preprocessing
   ↓
TF-IDF Feature Extraction
   ↓
Model Training
   ↓
Evaluation
   ↓
Results
```

Key project resources include:

* Dataset resources
* Experiment scripts
* Preprocessing implementations
* Machine-learning models
* Evaluation code
* Experimental results
* Visualization resources

For a detailed explanation of the repository organization, see the **Repository Structure** page in the Wiki.

---

## 📚 Documentation

The project Wiki provides detailed documentation for the research and implementation.

The main documentation sections include:

* Research Methodology
* Datasets
* Text Preprocessing
* Models
* Experimental Setup
* Results
* Reproducibility
* Repository Structure
* References

The Wiki serves as the technical documentation for the project and its experimental workflow.

---

## 📖 Research Paper

The research paper associated with this project is currently being prepared for publication.

Publication-specific information such as:

* Journal or conference
* DOI
* Volume and issue
* Page numbers
* Publication date

will be added after the paper is officially published.

---

## 📚 References

The project uses established machine-learning and NLP resources, including:

* scikit-learn
* NLTK
* NumPy
* Pandas
* Matplotlib
* IMDb sentiment dataset
* Twitter sentiment dataset

Detailed references and resources are documented in the project Wiki.

---

## 👨‍💻 Authors

* **Punith Dewangan**
* **Mir Shaad Ali**

Department of Artificial Intelligence and Data Science
Global Academy of Technology, Bengaluru, India

---

## ⭐ Project Status

This repository contains the implementation and experimental resources for an academic research project.

The research paper is **not yet published**.

The repository and Wiki may continue to be updated as the research progresses and additional documentation or publication information becomes available.

---

## 🔬 Research Focus

The central question of the project is:

> Does text preprocessing always provide enough benefit to justify its additional computational cost?

Rather than assuming that more preprocessing leads to better results, this project evaluates the question experimentally across datasets, models, preprocessing configurations, and training-data scales.
