# NLP Sentiment Analysis with TF-IDF & Logistic Regression

This repository contains an end-to-end natural language processing (NLP) pipeline designed to perform **Binary Sentiment Analysis** (e.g., classifying text as positive or negative) on large-scale text data, such as the IMDB Movie Reviews dataset.

The project leverages a robust statistical approach using **TF-IDF (Term Frequency-Inverse Document Frequency)** for feature extraction and a **Logistic Regression** classifier for scalable, lightweight, and highly interpretable modeling.

## 📌 Project Overview
Text data requires strict preprocessing before it can be effectively processed by mathematical algorithms. This pipeline implements:
- **Text Cleaning:** Utilizing Regular Expressions (Regex) to strip out numerical digits, punctuation markers, and unneeded structural symbols.
- **Formatting Regularization:** Stripping newline breaks (`\n`) and tab characters (`\t`) to ensure document string uniformness.
- **Feature Extraction:** Converting unstructured natural texts into highly informative numerical matrices via **TF-IDF Vectorization**.
- **Target Encoding:** Automatically mapping categorical labels into binary integers (`0` and `1`) using `LabelEncoder`.
- **Classification:** Training a highly effective **Logistic Regression** model capable of generalizing across large document datasets (~50,000 samples).
- **Inference Pipeline:** Saving the serialization artifacts (`joblib`) to safely stream and classify custom, out-of-sample raw text strings.

## 🛠️ Tech Stack & Dependencies
The notebook is built using **Python 3** and relies on the following open-source frameworks:
- **Pandas & NumPy:** For tabular operations and numeric matrix handling.
- **Scikit-Learn (sklearn):** For modeling (`LogisticRegression`), data splitting (`train_test_split`), tokenization (`TfidfVectorizer`), and diagnostic validation metrics (`classification_report`).
- **Joblib:** For serializing and reloading pipeline objects (`b.joblibs`).

## 📊 Evaluation & Performance
The model is evaluated using foundational classification metrics: Precision, Recall, Accuracy, and the $F_1\text{-score}$. Thanks to the mathematical properties of TF-IDF word weighting, the Logistic Regression model achieves outstanding classification scores on text balanced-sentiment sets without requiring heavy neural network hardware.
