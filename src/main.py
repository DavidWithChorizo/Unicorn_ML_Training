# main.py
import os
import numpy as np
from data_loading import load_csv_file
from data_preprocessing import discard_settling_period, extract_eeg_channels, segment_epochs, label_epochs_alternating
from model_training import create_eegnet_model, train_model_with_tuning
from sklearn.model_selection import train_test_split

def main():
    # List of file paths for your sessions.
    # Assume file names include a hint of movement type (e.g., 'left' or 'right').
    filepaths = [
        "Data_Gtec/UnicornRecorder_06_03_2025_15_23_220_left and neutral 1.csv",
        "Data_Gtec/UnicornRecorder_06_03_2025_15_30_400_left_and_neutral_2_0.20.csv",
        "Data_Gtec/UnicornRecorder_06_03_2025_16_25_570_right_1_0.05.csv",
        "Data_Gtec/UnicornRecorder_06_03_2025_16_32_090_right_2_1.15.csv"
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
        "UnicornRecorder_06_03_2025_15_23_220_left and neutral 1.csv": 30,
        "UnicornRecorder_06_03_2025_15_30_400_left_and_neutral_2_0.20.csv": 15,
        "UnicornRecorder_06_03_2025_16_25_570_right_1_0.05.csv": 10,
        "UnicornRecorder_06_03_2025_16_32_090_right_2_1.15.csv": 75,
    }
    
    # Each epoch is 2 seconds (i.e., 500 samples)
    epoch_length_seconds = 2
    
    # For each session file, process and label the data.
    for filepath in filepaths:
        print(f"Processing {filepath}...")
        df = load_csv_file(filepath)
        
        # Determine settling time based on the file name
        filename = os.path.basename(filepath)
        settling_time = settling_times.get(filename, 20)  # default to 20 seconds
        df_clean = discard_settling_period(df, settling_time, sample_rate=sample_rate, counter_column='counter')
        eeg_data = extract_eeg_channels(df_clean)  # defaults to first 8 columns
        epochs = segment_epochs(eeg_data, epoch_length_seconds, sample_rate=sample_rate)
        
        # Decide movement label based on session type.
        # Here, "left" or "right" in filename determines the movement label.
        if "left" in filename.lower():
            movement_label = 0  # left-hand movement
        elif "right" in filename.lower():
            movement_label = 1  # right-hand movement
        else:
            movement_label = None  # For undefined sessions
        
        # If the file is clearly a movement session, label alternating epochs.
        if movement_label is not None:
            labels = label_epochs_alternating(epochs, movement_label, neutral_label=2)
        else:
            # If movement_label is undefined, you might choose to label all epochs as neutral,
            # or handle it differently.
            labels = np.full((epochs.shape[0],), 2)
        
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
    

    '''
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
    '''
if __name__ == '__main__':
    main()
