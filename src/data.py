import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

TARGET_TAGS = ["math", "graphs", "strings", "number theory", "trees", "geometry", "games", "probabilities"]


def load_dataset(data_dir: str | Path | None=None):
    """Reads every JSON in the data directory and returns a dataframe with every useful columns"""
    data_dir = Path(data_dir or os.environ["DATA_DIR"])

    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(data_dir.glob("*.json"))]
    df = pd.DataFrame(records)[["src_uid", "prob_desc_description", "source_code", "tags"]]

    df["tags"] = df["tags"].apply(lambda tags: [t for t in (tags or []) if t in TARGET_TAGS])

    return df.set_index("src_uid").fillna("")