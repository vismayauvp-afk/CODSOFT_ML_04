# SMS Spam Detection

This project is an **SMS Spam Detection system** that classifies SMS messages as **spam** or **ham (legitimate)** using machine learning. It is trained on the [SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset) and uses **Naive Bayes**, **Logistic Regression**, and **SVM** classifiers with **TF-IDF** feature extraction.

---

## Features

- Preprocesses SMS messages by:
  - Lowercasing text  
  - Removing punctuation  
  - Removing stopwords  

- Supports **interactive message testing**:
  - Users can type any message and receive a prediction.  
  - Detects spam messages based on patterns learned from the dataset.  

- Trains and evaluates **three machine learning models**:
  - Naive Bayes  
  - Logistic Regression  
  - Support Vector Machine (SVM)  

---

## Dataset

- Dataset: [SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset)  
- Format: CSV (`spam.csv`) with two columns:
  - `v1` → label (`spam` or `ham`)  
  - `v2` → message text  

---

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd SMS_Spam_Detection
