# main.py
import os
import numpy as np
from data_loading import load_csv_file
from data_preprocessing import discard_settling_period, extract_eeg_channels, segment_epochs, label_epochs
from model_training import build_eegnet_model, train_model
from sklearn.model_selection import train_test_split

def main():
    # List of file paths for your sessions.
    # Assume file names include a hint of movement type (e.g., 'left' or 'right').
    filepaths = [
        "data/session_left.csv",
        "data/session_right_1.csv",
        "data/session_right_2.csv",
        "data/session_left_2.csv"
    ]
    
    all_epochs = []
    all_labels = []
    
    # Configuration for preprocessing
    sample_rate = 250  # Hz
    # Define EEG channel names from your CSV file (assumed to be 8 channels)
    eeg_columns = ['eeg1', 'eeg2', 'eeg3', 'eeg4', 'eeg5', 'eeg6', 'eeg7', 'eeg8']
    
    # For each session, you might have different settling times.
    # You could either hard-code them or read from a config.
    settling_times = {
        "session_left.csv": 20,
        "session_right_1.csv": 15,
        "session_right_2.csv": 15,
        "session_left_2.csv": 18,
    }
    
    # Each epoch is 2 seconds (i.e., 500 samples)
    epoch_length_seconds = 2
    
    for filepath in filepaths:
        print(f"Processing {filepath}...")
        df = load_csv_file(filepath)
        
        # Determine settling time based on the file name
        filename = os.path.basename(filepath)
        settling_time = settling_times.get(filename, 20)  # default to 20 seconds
        
        # Discard settling period
        df_clean = discard_settling_period(df, settling_time, sample_rate=sample_rate, counter_column='counter')
        
        # Extract EEG data
        eeg_data = extract_eeg_channels(df_clean, eeg_columns)
        
        # Segment the data into epochs (each 2 seconds long)
        epochs = segment_epochs(eeg_data, epoch_length_seconds, sample_rate=sample_rate)
        
        # Decide on label based on session type (this is just an example)
        if "left" in filename.lower():
            label = 0  # For left-hand movement
        elif "right" in filename.lower():
            label = 1  # For right-hand movement
        else:
            label = 2  # Could use a separate label for neutral or undefined
        
        labels = label_epochs(epochs, label)
        
        all_epochs.append(epochs)
        all_labels.append(labels)
    
    # Combine data from all sessions
    X = np.concatenate(all_epochs, axis=0)  # shape: (n_epochs, epoch_length_samples, n_channels)
    y = np.concatenate(all_labels, axis=0)
    
    # Expand dimensions to match EEGNet's expected input shape: (epochs, channels, time, 1)
    # We need to transpose axes if necessary. Here we assume data is (n_epochs, samples, channels)
    # so we switch to (n_epochs, channels, samples, 1).
    X = np.transpose(X, (0, 2, 1))
    X = X[..., np.newaxis]
    
    print("Preprocessing complete.")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Optional: split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Build the EEGNet model
    input_shape = X_train.shape[1:]  # (channels, samples, 1)
    nb_classes = len(np.unique(y))
    model = build_eegnet_model(input_shape=input_shape, nb_classes=nb_classes)
    
    # Train the model
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    
    # Evaluate model suitability by checking performance and inspecting the training history
    loss, acc = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {acc * 100:.2f}%")
    
if __name__ == '__main__':
    main()
