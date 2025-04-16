#!/usr/bin/env python
"""
simple_train.py

This script loads EEG CSV files, preprocesses the data, discards the neutral epochs,
and trains an EEGNet model to classify left and right hand movements using an 80/20 split.
Hyperparameters are fixed and no validation/tuning is performed.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, DepthwiseConv2D,
                                     Activation, AveragePooling2D, Dropout, SeparableConv2D,
                                     Flatten, Dense)
from tensorflow.keras.constraints import max_norm
from sklearn.model_selection import train_test_split

# ==================== Data Loading Functions ====================

def load_csv_file(filepath):
    """Load a CSV file and assign default column names."""
    try:
        df = pd.read_csv(filepath, header=None)
        df.columns = ['eeg1', 'eeg2', 'eeg3', 'eeg4', 'eeg5', 'eeg6', 'eeg7', 'eeg8', 'counter', 'timestamp']
        return df
    except Exception as e:
        raise IOError(f"Error reading {filepath}: {e}")

# ==================== Data Preprocessing Functions ====================

def discard_settling_period(df, settling_time_seconds, sample_rate=250, counter_column='counter'):
    """Discard the initial settling period from the data."""
    num_samples_to_discard = int(settling_time_seconds * sample_rate)
    df['norm_counter'] = df[counter_column] - df[counter_column].min()
    return df[df['norm_counter'] >= num_samples_to_discard].reset_index(drop=True)

def extract_eeg_channels(df, eeg_columns=None):
    """Extract EEG channels (defaults to first 8 columns)."""
    if eeg_columns is None:
        eeg_columns = list(range(8))
    return df.iloc[:, eeg_columns].values

def segment_epochs(eeg_data, epoch_length_seconds, sample_rate=250):
    """Segment continuous EEG data into fixed-length epochs."""
    epoch_length_samples = int(epoch_length_seconds * sample_rate)
    n_samples = eeg_data.shape[0]
    n_epochs = n_samples // epoch_length_samples
    eeg_data = eeg_data[:n_epochs * epoch_length_samples]  # Discard leftover samples.
    epochs = np.reshape(eeg_data, (n_epochs, epoch_length_samples, eeg_data.shape[1]))
    return epochs

# ==================== EEGNet Model Definition ====================

def create_eegnet_model(dropoutRate, kernLength, F1, D, F2, input_shape, nb_classes):
    """
    Create and compile the EEGNet model.
    """
    input1 = Input(shape=input_shape)
    
    # First block: Temporal convolution then depthwise convolution.
    x = Conv2D(F1, (1, kernLength), padding='same', use_bias=False)(input1)
    x = BatchNormalization()(x)
    x = DepthwiseConv2D((input_shape[0], 1),
                        use_bias=False,
                        depth_multiplier=D,
                        depthwise_constraint=max_norm(1.))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)
    x = Dropout(dropoutRate)(x)
    
    # Second block: Separable convolution.
    x = SeparableConv2D(F2, (1, 16), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 8))(x)
    x = Dropout(dropoutRate)(x)
    
    # Classification block.
    x = Flatten()(x)
    x = Dense(nb_classes, kernel_constraint=max_norm(0.5))(x)
    output = Activation('softmax')(x)
    
    model = Model(inputs=input1, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ==================== Main Function ====================

def main():
    # Define file paths; adjust paths as necessary.
    filepaths = [
        "Data_Gtec/adc_1_left.csv",
        "Data_Gtec/adc_1_right.csv",
        "Data_Gtec/adc_2_left.csv",
        "Data_Gtec/adc_2_right.csv",
        "Data_Gtec/adc_3_left.csv",
    ]
    
    all_epochs = []
    all_labels = []
    
    # Configuration parameters.
    sample_rate = 250            # Hz
    settling_time_seconds = 10   # discard first 10 seconds
    epoch_length_seconds = 2     # Each epoch is 2 seconds long (2*250 = 500 samples)
    
    # Process each CSV file.
    for filepath in filepaths:
        print(f"Processing file: {filepath}")
        df = load_csv_file(filepath)
        df_clean = discard_settling_period(df, settling_time_seconds, sample_rate, counter_column='counter')
        eeg_data = extract_eeg_channels(df_clean)
        epochs = segment_epochs(eeg_data, epoch_length_seconds, sample_rate)
        # Discard neutral epochs: keep every other epoch (assumes first epoch is the movement).
        action_epochs = epochs[::2]
        
        # Determine label based on filename.
        filename_lower = filepath.lower()
        if "left" in filename_lower:
            label = 0
        elif "right" in filename_lower:
            label = 1
        else:
            print(f"Skipping file {filepath}: Could not determine label.")
            continue
        
        labels = np.full((action_epochs.shape[0],), label)
        all_epochs.append(action_epochs)
        all_labels.append(labels)
    
    if not all_epochs:
        raise ValueError("No valid data was processed. Check file names and data integrity.")
    
    # Concatenate data from all files.
    X = np.concatenate(all_epochs, axis=0)  # shape: (n_epochs, 500, n_channels)
    y = np.concatenate(all_labels, axis=0)
    print(f"Total epochs: {X.shape[0]}, Labels shape: {y.shape}")
    
    # Prepare data for EEGNet:
    # EEGNet expects input shape: (channels, samples, 1)
    X = np.transpose(X, (0, 2, 1))  # New shape: (n_epochs, n_channels, samples)
    X = X[..., np.newaxis]          # Final shape: (n_epochs, n_channels, samples, 1)
    print(f"Data shape for EEGNet: {X.shape}")
    
    # Perform an 80/20 train/test split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    
    dropoutRate = 0.15464
    kernLength = 190
    F1 = 4
    D = 2
    F2 = 10
    
    # Define model input shape and number of classes.
    input_shape = X_train.shape[1:]  # e.g., (n_channels, 500, 1)
    nb_classes = 2
    
    # Create and train the model.
    model = create_eegnet_model(dropoutRate, kernLength, F1, D, F2, input_shape, nb_classes)
    model.summary()
    
    # Train the model on the training set only.
    history = model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)
    
    # Evaluate the model on the test set.
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")
    
    # Save the trained model.
    model.save("eeg_model_simple_80_20.h5")
    print("Model saved as 'eeg_model_simple_80_20.h5'.")

if __name__ == '__main__':
    # Optional: Disable oneDNN optimizations if needed.
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    main()