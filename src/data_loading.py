# data_loading.py
import pandas as pd

def load_csv_file(filepath):
    """
    Load a CSV file and return the DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        raise IOError(f"Error reading {filepath}: {e}")
