# Fake News Detection — NLP Classifier

A machine-learning pipeline that classifies news articles as **real** or **fake** using classical NLP techniques. The project experiments with two text-vectorization methods (Bag-of-Words and TF-IDF) and five classifiers, sweeping across a range of hyperparameter configurations to identify the best-performing setup.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Pipeline](#pipeline)
4. [Models & Vectorizers](#models--vectorizers)
5. [Experiments & Results](#experiments--results)
6. [Getting Started](#getting-started)
7. [License](#license)

---

## Project Overview

| | |
|---|---|
| **Task** | Binary text classification — *Fake (0)* vs *Real (1)* |
| **Vectorizers** | Bag-of-Words (CountVectorizer), TF-IDF |
| **Classifiers** | Logistic Regression, SVM, Naïve Bayes, Random Forest, XGBoost |
| **Key metric** | F1-score (test set) |
| **Feature sizes tested** | 5 000 and 10 000 max features |

Best observed test F1-score: **≈ 93%** (Logistic Regression with BOW, max\_features = 5 000)

---

## Repository Structure

```
fake-news-detection-nlp/
├── notebooks/
│   ├── clean.ipynb            # Data loading & cleaning walkthrough
│   ├── experimentsBOW.ipynb   # BOW hyperparameter sweep
│   └── experimentsTFIDF.ipynb # TF-IDF hyperparameter sweep
├── src/
│   ├── data_preprocess.py     # TextPreprocessor class (cleaning + lemmatization)
│   ├── vectorize.py           # Unified get_vectorize() helper (BOW / TF-IDF)
│   └── model_evaluate.py      # evaluate_model() and append_result() helpers
├── utils/
│   ├── vectorizeBOW.py        # run_bow_experiment() — BOW sweep utility
│   ├── vectorizeTFIDF.py      # run_tfidf_experiment() — TF-IDF sweep utility
│   └── visualize.py           # plot_best_model_results() visualization helper
├── results/
│   ├── BOW_lr_svm_nb_rf_xgb_results_5000.csv
│   ├── TFIDF_lr_svm_nb_rf_xgb_results_5000.csv
│   └── bow_lr_svm_nb_rf_xgb_results_10000.csv
├── reports/                   # Model-comparison bar charts (PNG)
├── .gitignore
├── LICENSE
└── README.md
```

> **Note:** The dataset directory is excluded from version control via `.gitignore`. See [Getting Started](#getting-started) for download instructions.

---

## Pipeline

```
Raw text
   │
   ▼
TextPreprocessor (src/data_preprocess.py)
   ├─ Lowercase
   ├─ URL extraction → slug text appended
   ├─ Remove URLs, emails, non-alpha characters
   ├─ Tokenize (NLTK word_tokenize)
   ├─ Remove stopwords
   └─ POS-aware lemmatization (WordNetLemmatizer)
   │
   ▼
Vectorizer (src/vectorize.py)
   ├─ CountVectorizer  (BOW)
   └─ TfidfVectorizer  (TF-IDF)
   │
   ▼
Classifier  →  evaluate_model()  →  Results CSV + Plots
```

---

## Models & Vectorizers

### Vectorizers

| Vectorizer | Class | Key hyperparameters swept |
|---|---|---|
| Bag-of-Words | `CountVectorizer` | `ngram_range`, `min_df`, `max_df`, `max_features` |
| TF-IDF | `TfidfVectorizer` | `ngram_range`, `min_df`, `max_df`, `max_features` |

### Classifiers

| Model | sklearn class |
|---|---|
| Logistic Regression | `LogisticRegression` |
| Support Vector Machine | `LinearSVC` / `SVC` |
| Naïve Bayes | `MultinomialNB` |
| Random Forest | `RandomForestClassifier` |
| XGBoost | `XGBClassifier` |

---

## Experiments & Results

Results from every hyperparameter configuration are saved to CSV files under `results/`. Visualizations are saved under `reports/`.

### Sample Results (BOW, max\_features = 5 000)

| Model | Test Accuracy | Test F1 | Train–Test F1 Gap |
|---|---|---|---|
| Logistic Regression | 92.88% | 92.80% | 3.35% |
| SVM | — | — | — |
| Naïve Bayes | — | — | — |
| Random Forest | — | — | — |
| XGBoost | — | — | — |

> Full results are in `results/BOW_lr_svm_nb_rf_xgb_results_5000.csv` and `results/TFIDF_lr_svm_nb_rf_xgb_results_5000.csv`.

### Result Visualizations

Model-comparison bar charts are stored in `reports/`:

| File | Description |
|---|---|
| `BOW-best-f1-model-comparison_5000.png` | Best F1 per model — BOW (5 000 features) |
| `BOW-best_f1-score-configs_5000.png` | Best F1 + hyperparams — BOW (5 000 features) |
| `TFIDF-best-f1-model-comparison_5000.png` | Best F1 per model — TF-IDF (5 000 features) |
| `TFIDF-best-f1-score-configs_5000.png` | Best F1 + hyperparams — TF-IDF (5 000 features) |
| `BOW-best-f1-model-comparison_10000.png` | Best F1 per model — BOW (10 000 features) |
| `TFIDF-best-f1-model-comparison_10000.png` | Best F1 per model — TF-IDF (10 000 features) |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Recommended: a virtual environment (`venv` or `conda`)

### Installation

```bash
git clone https://github.com/harmandeep2993/fake-news-detection-nlp.git
cd fake-news-detection-nlp

pip install -r requirements.txt   # add your own requirements file if needed
```

Core dependencies:

```
nltk
scikit-learn
xgboost
pandas
matplotlib
```

Download the required NLTK resources once:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

### Dataset

Place your dataset (CSV with at least a text column and a label column) inside a `dataset/` folder at the project root. The folder is git-ignored. A commonly used public dataset is the [Fake and Real News Dataset on Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset).

### Running the Notebooks

```bash
jupyter notebook notebooks/clean.ipynb           # preprocessing walkthrough
jupyter notebook notebooks/experimentsBOW.ipynb  # BOW experiments
jupyter notebook notebooks/experimentsTFIDF.ipynb # TF-IDF experiments
```

### Using the Source Modules

```python
from src.data_preprocess import TextPreprocessor
from src.vectorize import get_vectorize
from src.model_evaluate import evaluate_model

preprocessor = TextPreprocessor()
X_train_clean = X_train.apply(preprocessor.preprocess)
X_test_clean  = X_test.apply(preprocessor.preprocess)

vectorizer, features, X_train_vec, X_test_vec = get_vectorize(
    X_train_clean, X_test_clean,
    method='tfidf',
    ngram_range=(1, 2),
    max_features=5000
)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_vec, y_train)

result = evaluate_model(model, X_train_vec, X_test_vec, y_train, y_test)
print(result)
```

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file included in this repository.