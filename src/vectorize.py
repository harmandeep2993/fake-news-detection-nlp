import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def get_vectorize(
        X_train,
        X_test,
        method: str = 'bow',
        ngram_range: tuple=(1,1),
        min_df:int = 1, 
        max_df: float = 0.01,
        max_features:int | None = 10000):
    
    if method == 'bow':

        vectorizer = CountVectorizer(
            ngram_range=ngram_range, 
            min_df=min_df, 
            max_df=max_df, 
            max_features=max_features
        )
        print("CountVectorizer")
    
    elif method == 'tfidf':

        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range, 
            min_df=min_df, 
            max_df=max_df, 
            max_features=max_features
        )
        print("TF-IDF Vectorizer")
    
    else:
        raise ValueError("Vectorize method must be 'bow' or 'tfidf'")
    
    # Fit only on training data
    X_train_vec = vectorizer.fit_transform(X_train)

    # Transform test data
    X_test_vec = vectorizer.transform(X_test)

    features = vectorizer.get_feature_names_out()

    print("Vectorization completed")
    print("Total Features:", max_features) 

    return vectorizer, features, X_train_vec, X_test_vec