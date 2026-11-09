import argparse
import numpy as np

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer

from src.data import TARGET_TAGS
from src.train import MODEL_PATH, TEST_PATH


def predict(src_uid=None):
    """Predicts the tags of one exercise of the test split, or lists some test ids if no id is given"""
    test = pd.read_parquet(TEST_PATH) # Predicitons only on test set 

    if src_uid is None: # If ID is not given, list available test ids 
        print(test.index[:10].tolist())
        return

    model = joblib.load(MODEL_PATH)
    pred = model.predict(test.loc[[src_uid]].drop(columns="tags"))[0]

    print("Predicted tags :", np.array(TARGET_TAGS)[pred == 1].tolist())
    print("True tags :     ", list(test.loc[src_uid, "tags"]))


def evaluate():
    """Evaluates on the test split only, never seen during training"""

    model = joblib.load(MODEL_PATH)
    test = pd.read_parquet(TEST_PATH)
    Y_test = MultiLabelBinarizer(classes=TARGET_TAGS).fit_transform(test["tags"])
    pred = model.predict(test.drop(columns="tags"))

    print(classification_report(Y_test, pred, target_names=TARGET_TAGS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Codeforces tags pred")
    parser.add_argument("command", choices=["predict", "evaluate"])
    parser.add_argument("src_uid", nargs="?", help="id of the exercise to predict the tags for")
    args = parser.parse_args()

    if args.command == "predict":
        predict(args.src_uid)
    else:
        evaluate()
