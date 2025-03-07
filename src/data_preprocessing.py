# preprocessing.py
import numpy as np

def discard_settling_period(df, settling_time_seconds, sample_rate=250, counter_column='counter'):
    num_samples_to_discard = int(settling_time_seconds * sample_rate)
    # Normalize the counter so that it starts at 0
    df['norm_counter'] = df[counter_column] - df[counter_column].min()
    
    # For debugging: print the range of normalized counter values
    print("Normalized counter range:", df['norm_counter'].min(), df['norm_counter'].max())
    
    # Discard samples until the normalized counter reaches the threshold
    df_clean = df[df['norm_counter'] >= num_samples_to_discard].reset_index(drop=True)
    return df_clean


def extract_eeg_channels(df, eeg_columns=None):
    """
    Extract EEG channels from the DataFrame. If eeg_columns is None, 
    default to using the first 8 columns.
    
    Parameters:
      df: DataFrame with the raw data.
      eeg_columns: List of column indices or names corresponding to EEG channels.
                   If None, defaults to the first 8 columns.
      
    Returns:
      NumPy array of EEG data.
    """
    if eeg_columns is None:
        eeg_columns = list(range(8))
    return df.iloc[:, eeg_columns].values


def segment_epochs(eeg_data, epoch_length_seconds, sample_rate=250):
    """
    Segment continuous EEG data into fixed-length epochs.
    
    Parameters:
      eeg_data: NumPy array of shape (n_samples, n_channels).
      epoch_length_seconds: Duration of each epoch in seconds.
      sample_rate: Sampling rate.
      
    Returns:
      NumPy array of shape (n_epochs, epoch_length_samples, n_channels).
    """
    epoch_length_samples = int(epoch_length_seconds * sample_rate)
    n_samples = eeg_data.shape[0]
    n_epochs = n_samples // epoch_length_samples
    # Only use full epochs
    eeg_data = eeg_data[:n_epochs * epoch_length_samples]
    epochs = np.reshape(eeg_data, (n_epochs, epoch_length_samples, eeg_data.shape[1]))
    return epochs

def label_epochs_alternating(epochs, movement_label, neutral_label=2):
    """
    Label epochs in an alternating pattern. Assumes the first epoch is a movement epoch,
    and then alternates with neutral epochs.
    
    Parameters:
      epochs: NumPy array of shape (n_epochs, ...).
      movement_label: Label for the movement (e.g. 0 for left, 1 for right).
      neutral_label: Label for neutral (default is 2).
      
    Returns:
      A NumPy array of labels for each epoch.
    """
    n_epochs = epochs.shape[0]
    labels = []
    for i in range(n_epochs):
        # Even index: movement, Odd index: neutral.
        if i % 2 == 0:
            labels.append(movement_label)
        else:
            labels.append(neutral_label)
    return np.array(labels)

