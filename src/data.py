import json
from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/code_classification_dataset")
TARGET_TAGS = ["math", "graphs", "strings", "number theory", "trees", "geometry", "games", "probabilities"]


def load_dataset(data_dir: str | Path = DATA_DIR):
    """Reads every JSON in the data directory and returns a dataframe"""

    records = []

    for p in sorted(Path(data_dir).glob("*.json")):
        records.append(json.loads(p.read_text(encoding="utf-8")))

    df = pd.DataFrame(records)[["src_uid", "prob_desc_description", "source_code", "tags"]] # Classification will be doine on description and source code

    df["tags"] = df["tags"].apply(lambda tags: [t for t in (tags or []) if t in TARGET_TAGS]) # Only keep tageted tags

    return df.set_index("src_uid").fillna("")