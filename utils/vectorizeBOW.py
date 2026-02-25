import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

metrics_df = pd.DataFrame()

def run_bow_experiment(
    model: BaseEstimator,
    X_train,
    X_test,
    y_train,
    y_test,
    ngram_range=(1,1),
    min_df:int | float = 10,
    max_df: float =1.0,
    set_max_features=10000,
    comments=""
):
    
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features= set_max_features
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
        "ngram_range": ngram_range,
        "min_df": min_df,
        "max_df": max_df,
        "features_rec": len(vectorizer.get_feature_names_out()),
        "max_features": set_max_features,
        
        
        "test_accuracy":  accuracy_score(y_test, y_test_pred),
        
        "train_f1": f1_score(y_train, y_train_pred),
        "test_f1":  f1_score(y_test, y_test_pred),
        "gap_percent": round((train_f1 - test_f1) * 100, 2),
        
        "comments": comments
    }
    
    return row