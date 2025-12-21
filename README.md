# 🎬 IMDB Sentiment Analysis

Binary sentiment classification of IMDB movie reviews using NLP feature engineering and multiple ML models. This repository contains notebooks, dataset utilities, a model registry, and a compact summary of results.

---

## 📌 Overview

- Clean and preprocess raw reviews (tokenization, normalization)
- Feature engineering with TF‑IDF (1–2 n‑grams, 5k features) and optional Word2Vec embeddings
- Train and compare several classifiers
- Persist trained artifacts and a model registry for reproducibility

---

## 🧠 Models Trained

- Logistic Regression (best performer)
- Multinomial Naive Bayes
- Random Forest
- Gradient Boosting
- Decision Tree

Cross-validation and GridSearchCV were applied to top models for stable performance.

---

## ⚙️ Setup

```powershell
# From the project root
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

If you prefer conda, create and activate an environment, then install from `requirements.txt`.

---

## 🧪 Run the Notebook

You can explore end-to-end preprocessing, training, and evaluation directly in VS Code or Jupyter:

- Open `NLP_SENTIMENT_ANALYSIS.ipynb` and run cells sequentially
- Or open `NLP_Sentiment_Analysis_Complete.ipynb` for the full pipeline view

> Tip: In VS Code, use the “Run All” button in the notebook toolbar.

---

## 📊 Results (from notebook)

Summary below is derived from `sentiment_analysis_summary.txt` (full details inside the file):

- Dataset: 49,582 samples (Train: 39,665, Test: 9,917)
- Sentiment distribution: ~50/50 (negative: 24,698, positive: 24,884)
- Features: TF‑IDF (5,000 features, 1–2 n‑grams, min_df=2, max_df=0.8). Word2Vec 100‑dim available.

Top models by F1‑weighted on the test set:

- Logistic Regression: Accuracy 0.8872, F1 0.8871, ROC‑AUC 0.9565
- Multinomial NB: Accuracy 0.8565, F1 0.8565, ROC‑AUC 0.9314
- Random Forest: Accuracy 0.8441, F1 0.8441, ROC‑AUC 0.9252

Additional artifacts produced in the notebook:

- Confusion matrices, ROC and PR curves (top models)
- Cross‑validation score distributions

> Key insight: Logistic Regression is the best trade‑off of accuracy, AUC, and stability across folds.

---

## 📦 Model Registry & Artifacts

- Trained models and metadata live under `models/`
- Registry files (JSON) provide details and paths, e.g. `model_registry_YYYYMMDD_HHMMSS.json`

Loading a saved model (example):

```python
import json, joblib, os

registry_path = os.path.join('models', 'model_registry_20251209_004604.json')
with open(registry_path, 'r', encoding='utf-8') as f:
	registry = json.load(f)

best_entry = max(registry['models'], key=lambda m: m.get('f1_weighted', 0))
model_path = best_entry['model_path']
vectorizer_path = registry.get('vectorizer_path')

clf = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path) if vectorizer_path else None

def predict_review(text: str):
	X = vectorizer.transform([text]) if vectorizer else [text]
	return clf.predict(X)[0]

print(predict_review("This movie was amazing! Great acting and story."))
```

---

## 🧰 Utilities

- `test_dataset.py`: quick dataset sanity check (shape, columns, sentiment distribution, sample review)

Run it:

```powershell
python .\test_dataset.py
```

---

## 📂 Project Structure

```
Sentiment-Analysis-IMBD/
├── dataset/
│   └── IMDBDataset.csv
├── models/
│   ├── model_registry_20251209_004604.json
│   ├── registre_20251210_050945.json
│   └── registre_20251210_051134.json
├── NLP_SENTIMENT_ANALYSIS.ipynb
├── NLP_Sentiment_Analysis_Complete.ipynb
├── requirements.txt
├── sentiment_analysis_summary.txt
├── test_dataset.py
└── README.md
```

---

## 🚀 Next Steps

- Add a small CLI or FastAPI service for inference
- Track prediction confidence and set thresholds for business use
- Expand preprocessing (handling sarcasm, negations, domain adaptation)

---

## 🙌 Acknowledgments

Thanks to the open IMDB dataset and the Python ecosystem (scikit‑learn, pandas, numpy, matplotlib).

