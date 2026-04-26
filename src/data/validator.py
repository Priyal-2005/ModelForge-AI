import pandas as pd

def validate_data(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validates the dataset for required conditions."""
    errors = []
    
    # Required columns
    required_cols = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        
    # Check target is binary
    if 'survived' in df.columns:
        unique_targets = df['survived'].dropna().unique()
        if set(unique_targets) - {0, 1}:
            errors.append("Target column 'survived' is not binary.")
            
    # Check for empty dataframe
    if df.empty:
        errors.append("Dataset is empty.")
        
    return len(errors) == 0, errors
