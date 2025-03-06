import pandas as pd

def load_csv_file(filepath):
    """
    Load a CSV file without headers and return the DataFrame.
    """
    try:
        df = pd.read_csv(filepath, header=None)
        return df
    except Exception as e:
        raise IOError(f"Error reading {filepath}: {e}")
