# 📰 Fake News Detection System

A machine learning-based system to classify news as **real or fake**, exploring trust issues in AI-generated and online content.

---

## 🚀 Overview

With the rapid growth of AI-generated content, verifying the authenticity of information has become critical.

This project builds a simple yet effective model to detect whether a news article is real or fake using Natural Language Processing (NLP) techniques.

---

## 🧠 Key Idea

The system uses:

- **TF-IDF Vectorization** → Convert text into numerical features  
- **Logistic Regression** → Classify news as real or fake  

---

## ⚙️ Features

- ✅ Text preprocessing and cleaning  
- ✅ TF-IDF based feature extraction  
- ✅ Logistic Regression model  
- ✅ Accuracy evaluation  
- ✅ Real-time prediction using user input  

---

## 📂 Dataset

The dataset contains the following columns:

- title
- text ✅ (used as input)
- date
- source
- author
- category
- label ✅ (target: real / fake)

---

## 🛠️ How It Works

1. Load dataset
2. Extract `text` and `label`
3. Convert text → TF-IDF features
4. Train model
5. Predict new inputs

---

## 🔍 Learning Outcomes
1. Understanding NLP pipelines
2. Applying machine learning to real-world problems
3. Exploring trust and misinformation in AI systems

---

## 🚧 Limitations
1. Accuracy depends on dataset quality
2. Cannot fully understand context or sarcasm
3. Limited to binary classification

---

## 🔮 Future Improvements
1. Deep learning models (LSTM, BERT)
2. Better preprocessing (stemming, lemmatization)
3. Web interface for user interaction
4. Integration with fact-checking APIs

---

## 🎯 Motivation

This project explores the challenge of trust in AI-generated and online information, aligning with research in reliable and trustworthy AI systems.
