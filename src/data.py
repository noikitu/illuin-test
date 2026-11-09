import json
from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/code_classification_dataset")
TARGET_TAGS = ["math", "graphs", "strings", "number theory", "trees", "geometry", "games", "probabilities"]


def load_dataset(data_dir: str | Path | None=None):
    """Reads every JSON in the data directory and returns a dataframe"""

    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(data_dir.glob("*.json"))]
    df = pd.DataFrame(records)[["src_uid", "prob_desc_description", "source_code", "tags"]]

    df["tags"] = df["tags"].apply(lambda tags: [t for t in (tags or []) if t in TARGET_TAGS])

    return df.set_index("src_uid").fillna("")