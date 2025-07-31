import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(filepath, test_size=0.2, random_state=42):
    # df = pd.read_csv(filepath)
    df = pd.read_parquet(filepath)

    # Select your structured feature columns
    structured_columns = [
        'rating_deviation', 'hedged', 'hedging_density', 'product_avg_rating',
        'verified_purchase', 'length_readability_score', 'headline_length',
        'review_age', 'rating_alignment', 'star_rating', 'readability',
        'avg_rating_amplified_length', 'headline_word_count',
        'sentiment_subjectivity', 'rating_sentiment_gap', 'review_word_count',
        'hedge_count', 'review_length', 'is_bad_reviewers', 'sentiment_polarity'
    ]

    X_text = df['review_text_full']
    y = df['helpful_ratio']
    X_struct = df[structured_columns]

    return train_test_split(X_text, X_struct, y, test_size=test_size, random_state=random_state)
