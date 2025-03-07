import pandas as pd


def load_csv_file(filepath):
    """
    Load a CSV file without headers and assign default column names.
    Assumes CSV has 10 columns:
      - Columns 0-7: EEG channels
      - Column 8: timestamp (which we may ignore if messy)
      - Column 9: counter
    """
    try:
        df = pd.read_csv(filepath, header=None)
        # Assign default names
        df.columns = ['eeg1', 'eeg2', 'eeg3', 'eeg4', 'eeg5', 'eeg6', 'eeg7', 'eeg8', 'timestamp', 'counter']
        return df
    except Exception as e:
        raise IOError(f"Error reading {filepath}: {e}")
