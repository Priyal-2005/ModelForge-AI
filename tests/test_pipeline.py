import pandas as pd
from src.data.validator import validate_data
from src.data.preprocessor import DataPreprocessor

def test_validator():
    df = pd.DataFrame({
        'survived': [0, 1],
        'pclass': [1, 3],
        'sex': ['male', 'female'],
        'age': [22, 38],
        'sibsp': [1, 0],
        'parch': [0, 0],
        'fare': [7.25, 71.28],
        'embarked': ['S', 'C']
    })
    is_valid, errors = validate_data(df)
    assert is_valid is True
    assert len(errors) == 0

def test_validator_missing_cols():
    df = pd.DataFrame({'age': [22]})
    is_valid, errors = validate_data(df)
    assert is_valid is False
    assert len(errors) > 0
