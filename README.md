# recipe-cuisine-detection
Predict cuisine type from recipe ingredients using TF-IDF and classical ML classifiers (LinearSVC, LR, Naive Bayes) — 82% accuracy on 13 cuisines.
# 🍽️ What's Cooking? — Cuisine Classifier

> Predict the cuisine of a recipe just from its ingredients using classical NLP techniques.

```
Input  → ["chicken", "turmeric", "garam masala", "yogurt", "onion"]
Output → 🇮🇳 Indian  (82.3% accuracy)
```

---

## 📌 Overview

This project trains a text classification model on a multi-cuisine recipe dataset. Given a list of ingredients, the model predicts which cuisine the dish belongs to — Indian, Italian, Chinese, French, and 9 more.

No deep learning. No fancy embeddings. Just clean NLP fundamentals that work.

---

## 📊 Dataset

- **Source:** [Multi-Cuisine Recipe Dataset](https://www.kaggle.com/datasets/sonalshinde123/multi-cuisine-recipe-dataset) — Kaggle
- **Size:** 620 recipes × 5 columns
- **Label:** `area` — 13 cuisine classes
- **Input feature:** `ingredients` column (free text)
- **Challenge:** Imbalanced classes (Indian: 200 vs Italian: 21)

| Cuisine | Count |
|---------|-------|
| Indian | 200 |
| American | 77 |
| British | 59 |
| Spanish | 48 |
| Turkish | 30 |
| French | 28 |
| Chinese | 27 |
| Vietnamese | 27 |
| Polish | 27 |
| Jamaican | 27 |
| Thai | 27 |
| Canadian | 22 |
| Italian | 21 |

---

## ⚙️ Pipeline

```
Raw ingredients text
        ↓
Text Cleaning
  - lowercase
  - remove numbers & fractions (1/4, 2, ...)
  - remove units (cup, tbsp, teaspoon, ...)
  - remove descriptions (chopped, diced, ...)
        ↓
TF-IDF Vectorizer
  - ngram_range = (1, 2)
  - max_features = 8000
  - sublinear_tf = True
        ↓
Classifier (3 models compared)
  - LinearSVC ✅ best
  - Logistic Regression
  - Naive Bayes
        ↓
Cuisine Prediction
```

---

## 🏆 Results

| Model | Test Accuracy |
|-------|--------------|
| **LinearSVC** | **82.3%** |
| Logistic Regression | ~78% |
| Naive Bayes | ~72% |

> `class_weight='balanced'` used to handle class imbalance.
```

---


## 📁 Project Structure

```
whats-cooking-nlp/
│
├── cuisine_classifier.ipynb   # main notebook (EDA + training + evaluation)
├── cuisine_classifier.pkl     # saved best model (LinearSVC)
├── recipes.csv                # dataset (download from Kaggle)
├── requirements.txt
└── README.md
```

---


## 📦 Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```
## ⚠️ Note

If GitHub fails to render the notebook and shows "An error occurred", just paste the notebook URL into [nbviewer.org] to view it properly.
