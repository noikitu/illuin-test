from pathlib import Path

import joblib
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from src.data import load_dataset, TARGET_TAGS

COLS = ["prob_desc_description", "source_code"]
MODEL_PATH = Path("models/model.joblib") # Save of the trained model in joblib format
TRAIN_PATH = Path("data/train.parquet") # Save of the train split in parquet format
TEST_PATH = Path("data/test.parquet") # Save of the test split in parquet format


def build_model():
    """Logistic Regression model with TF-IDF features for both description and source code"""
    return Pipeline([
        ("features", ColumnTransformer([
            ("desc", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
             "prob_desc_description"), # Sublinear TF scaling to reduce the impact of very frequent terms, min_df to avoid unique terms
            ("code", TfidfVectorizer(token_pattern=r"[A-Za-z_][A-Za-z0-9_]*", min_df=2, sublinear_tf=True),
             "source_code"), # Token pattern to exclude numbers 
        ])),
        ("clf", OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight="balanced"))), # Balanced weight class to handle class imbalance (probabilites for ex)
    ])


def split_dataset(df):
    """Train test split with stratification on multilabels using iterative-stratification package"""
    mlb = MultiLabelBinarizer(classes=TARGET_TAGS) # Encode labels as a binary array
    Y = mlb.fit_transform(df["tags"])
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state= 42)
    train_idx, test_idx = next(msss.split(df, Y)) # Only 1 split so we use next for indexes
    X = df.drop(columns="tags")
    return X.iloc[train_idx], X.iloc[test_idx], Y[train_idx], Y[test_idx]


def train():
    df = load_dataset()
    X_train, X_test, Y_train, Y_test = split_dataset(df)

    # Save train and test splits in parquet format for future use
    df.loc[X_train.index].to_parquet(TRAIN_PATH)
    df.loc[X_test.index].to_parquet(TEST_PATH)

    model = build_model()
    model.fit(X_train, Y_train) 

    # Save Model
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
