import pandas as pd
import seaborn as sns
from pathlib import Path
from src.utils.logger import logger

def load_titanic_data(save_path="data/raw/titanic.csv") -> pd.DataFrame:
    """Loads Titanic dataset from seaborn and saves to raw data path."""
    logger.info("Loading Titanic dataset...")
    df = sns.load_dataset("titanic")
    p = Path(save_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    logger.info(f"Saved raw dataset to {p}")
    return df
