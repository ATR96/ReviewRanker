import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from scipy.sparse import vstack
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from sentence_transformers import SentenceTransformer

# Load model once at module level (saves time)
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_sbert_embeddings(text_series, batch_size=128, save_path=None):
    embeddings = []
    for i in tqdm(range(0, len(text_series), batch_size), desc="SBERT Encoding"):
        batch = text_series[i:i + batch_size]
        batch_embeddings = sbert_model.encode(batch, show_progress_bar=False)
        embeddings.append(batch_embeddings)

    embeddings = np.vstack(embeddings)

    if save_path:
        joblib.dump(embeddings, save_path)

    return embeddings

def generate_tfidf_features(train_text, test_text, save_path=None, max_features=10000, ngram_range=(1,2), chunk_size = 10000):
    # Initialize TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=6,
        max_df=0.8
    )

    # Fit on a sample to avoid RAM overload
    sample_size = int(len(train_text)/3)
    df_sample = train_text.sample(sample_size, random_state=42)
    tfidf.fit(df_sample)

    # Define the transform function
    def transform_in_chunks(text_series):
        chunks = []
        n_chunks = len(text_series) // chunk_size + 1
        for i in tqdm(range(0, len(text_series), chunk_size), desc="Transforming TF-IDF", total=n_chunks):
            batch = text_series.iloc[i:i + chunk_size]
            batch_tfidf = tfidf.transform(batch)
            chunks.append(batch_tfidf)
        return vstack(chunks)

    # Apply on train and test separately
    X_train_tfidf = transform_in_chunks(train_text)
    X_test_tfidf = transform_in_chunks(test_text)

    # Optionally save the vectorizer
    if save_path:
        joblib.dump(tfidf, save_path)

    return X_train_tfidf, X_test_tfidf, tfidf

def scale_structured_features(X_train_struct, X_test_struct):
    # Binary columns to keep
    binary_columns = ['hedged', 'verified_purchase', 'is_bad_reviewers']

    # Separate continuous and binary
    X_train_bin = X_train_struct[binary_columns].reset_index(drop=True)
    X_test_bin = X_test_struct[binary_columns].reset_index(drop=True)

    X_train_cont = X_train_struct.drop(columns=binary_columns).reset_index(drop=True)
    X_test_cont = X_test_struct.drop(columns=binary_columns).reset_index(drop=True)

    # Scale continuous only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_cont)
    X_test_scaled = scaler.transform(X_test_cont)

    # Recombine
    X_train_final = pd.DataFrame(X_train_scaled, columns=X_train_cont.columns)
    X_test_final = pd.DataFrame(X_test_scaled, columns=X_test_cont.columns)

    X_train_final[binary_columns] = X_train_bin
    X_test_final[binary_columns] = X_test_bin

    return X_train_final.values, X_test_final.values, scaler

