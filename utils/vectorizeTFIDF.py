import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

metrics_df = pd.DataFrame()

def run_tfidf_experiment(
    model: BaseEstimator,
    X_train,
    X_test,
    y_train,
    y_test,
    ngram_range=(1,1),
    min_df:int | float = 10,
    max_df: float =1.0,
    set_max_features=5000,
    comments=""
):
    
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=set_max_features
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    
    model.fit(X_train_vec, y_train)
    
    y_train_pred = model.predict(X_train_vec)
    y_test_pred  = model.predict(X_test_vec)

    train_f1 = f1_score(y_train, y_train_pred)
    test_f1  = f1_score(y_test, y_test_pred)
    
    row = {
            "model": model.__class__.__name__,
            "method": 'TFIDF',
            "ngram_range": ngram_range,
            "min_df": min_df,
            "max_df": max_df,
            "max_feature": len(vectorizer.get_feature_names_out()),
            "features_limit": set_max_features,
            "train_acc": round(accuracy_score(y_train, y_train_pred), 4),
            "test_acc":  round(accuracy_score(y_test, y_test_pred), 4),
            "f1_train": round(f1_score(y_train, y_train_pred), 4),
            "f1_test":  round(f1_score(y_test, y_test_pred),4),
            "gap_percent": round((train_f1 - test_f1) * 100, 2),
            "comments": comments
        }
        
    return row