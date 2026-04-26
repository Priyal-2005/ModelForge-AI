import pandas as pd
import seaborn as sns
import os

def load_titanic_data(save_path="data/raw/titanic.csv"):
    """Loads Titanic dataset from seaborn and saves to raw data path."""
    print("Loading Titanic dataset...")
    df = sns.load_dataset("titanic")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    return df
