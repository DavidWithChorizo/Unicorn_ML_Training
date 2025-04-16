#!/usr/bin/env python
"""
new_main.py

This script loads EEG CSV files, preprocesses the data, discards the neutral epochs,
and trains an EEGNet model to classify left and right hand movements.

Features:
    - Loads CSV files (expects 10 columns: 8 EEG channels, counter, timestamp).
    - Discards the first 10 seconds of data (settling period).
    - Segments the remaining data into 2-second epochs (500 samples per epoch).
    - Discards neutral epochs by keeping only the first epoch of each action–neutral pair.
    - Labels epochs based on file names ('left' -> 0, 'right' -> 1).
    - Splits the data into training, validation, and test sets.
    - Optionally uses Optuna for hyperparameter tuning of the EEGNet model.
    - Trains and evaluates the model, then saves it to disk.
    - Generates a classification report table and plots training learning curves.

Usage:
    Simply run the script in your Python environment. Make sure all required CSV files 
    are in the same folder or update the file paths accordingly.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  # For performance metrics table
# Import necessary Keras components for building the EEGNet model.
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, DepthwiseConv2D,
                                     Activation, AveragePooling2D, Dropout, SeparableConv2D,
                                     Flatten, Dense)
from tensorflow.keras.constraints import max_norm
import optuna


# ==================== Data Loading Functions ====================

def load_csv_file(filepath):
    """
    Load a CSV file with no headers and assign default column names.
    
    CSV file is expected to have 10 columns:
        - Columns 0-7: EEG channels.
        - Column 8: Counter.
        - Column 9: Timestamp.
    
    Parameters:
        filepath (str): Path to the CSV file.
    
    Returns:
        df (pandas.DataFrame): Loaded data with assigned column names.
    """
    try:
        df = pd.read_csv(filepath, header=None)
        # Assign default column names.
        df.columns = ['eeg1', 'eeg2', 'eeg3', 'eeg4', 'eeg5', 'eeg6', 'eeg7', 'eeg8', 'counter', 'timestamp']
        return df
    except Exception as e:
        raise IOError(f"Error reading {filepath}: {e}")


# ==================== Data Preprocessing Functions ====================

def discard_settling_period(df, settling_time_seconds, sample_rate=250, counter_column='counter'):
    """
    Discard an initial settling period from the data.
    
    This function normalizes the counter column and discards rows until
    the normalized counter reaches the number of samples corresponding 
    to the given settling time.
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing raw EEG data.
        settling_time_seconds (int or float): Duration (in seconds) to discard.
        sample_rate (int): Sampling frequency in Hz.
        counter_column (str): Name of the column representing the counter.
    
    Returns:
        df_clean (pandas.DataFrame): DataFrame after discarding the settling period.
    """
    num_samples_to_discard = int(settling_time_seconds * sample_rate)
    # Normalize the counter so that it starts at zero.
    df['norm_counter'] = df[counter_column] - df[counter_column].min()
    print("Normalized counter range:", df['norm_counter'].min(), df['norm_counter'].max())
    # Discard rows until we pass the settling period.
    df_clean = df[df['norm_counter'] >= num_samples_to_discard].reset_index(drop=True)
    return df_clean


def extract_eeg_channels(df, eeg_columns=None):
    """
    Extract EEG channels from the DataFrame.
    
    If eeg_columns is None, defaults to the first 8 columns.
    
    Parameters:
        df (pandas.DataFrame): DataFrame with raw data.
        eeg_columns (list or None): List of column indices or names for EEG channels.
    
    Returns:
        numpy.ndarray: Array of EEG channel data.
    """
    if eeg_columns is None:
        eeg_columns = list(range(8))
    return df.iloc[:, eeg_columns].values


def segment_epochs(eeg_data, epoch_length_seconds, sample_rate=250):
    """
    Segment continuous EEG data into fixed-length epochs.
    
    If the total number of samples is not a multiple of the epoch length, discard the leftover samples.
    
    Parameters:
        eeg_data (numpy.ndarray): Array with shape (n_samples, n_channels).
        epoch_length_seconds (int or float): Duration in seconds for each epoch.
        sample_rate (int): Sampling frequency in Hz.
    
    Returns:
        numpy.ndarray: Array of shape (n_epochs, epoch_length_samples, n_channels).
    """
    epoch_length_samples = int(epoch_length_seconds * sample_rate)
    n_samples = eeg_data.shape[0]
    n_epochs = n_samples // epoch_length_samples  # Only complete epochs are used.
    # Discard extra samples that do not form a complete epoch.
    eeg_data = eeg_data[:n_epochs * epoch_length_samples]
    # Reshape into epochs.
    epochs = np.reshape(eeg_data, (n_epochs, epoch_length_samples, eeg_data.shape[1]))
    return epochs


# ==================== EEGNet Model Definition ====================

def create_eegnet_model(dropoutRate, kernLength, F1, D, F2, input_shape, nb_classes):
    """
    Create and compile the EEGNet model.
    
    Parameters:
        dropoutRate (float): Dropout rate for dropout layers.
        kernLength (int): Kernel length for the first temporal convolution.
        F1 (int): Number of filters for the initial convolution.
        D (int): Depth multiplier for the DepthwiseConv2D layer.
        F2 (int): Number of filters for the SeparableConv2D layer.
        input_shape (tuple): Shape of the input data (channels, samples, 1).
        nb_classes (int): Number of output classes.
    
    Returns:
        model (tf.keras.Model): Compiled EEGNet model.
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
    
    # Compile the model with Adam optimizer and sparse categorical crossentropy loss.
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# ==================== Optuna Hyperparameter Tuning Functions ====================

def objective(trial, X_train, y_train, X_val, y_val, input_shape, nb_classes, epochs=20, batch_size=32):
    """
    Objective function for Optuna hyperparameter optimization.
    
    This function defines the search space and returns 1 - validation accuracy 
    (because Optuna minimizes the objective).
    
    Parameters:
        trial (optuna.trial.Trial): A trial object for hyperparameter suggestions.
        X_train, y_train: Training data and labels.
        X_val, y_val: Validation data and labels.
        input_shape (tuple): Input shape for the model.
        nb_classes (int): Number of output classes.
        epochs (int): Number of epochs for training during tuning.
        batch_size (int): Batch size.
    
    Returns:
        float: Objective value (1 - validation accuracy).
    """
    # Define hyperparameter search space.
    dropoutRate = trial.suggest_float('dropoutRate', 0.1, 0.3)
    kernLength = trial.suggest_int('kernLength', 150, 200, step=10)
    F1 = trial.suggest_int('F1', 2, 8, step=2)
    D = trial.suggest_int('D', 1, 2)
    F2 = trial.suggest_int('F2', 4, 12, step=2)
    
    # Create model with the suggested hyperparameters.
    model = create_eegnet_model(dropoutRate, kernLength, F1, D, F2, input_shape, nb_classes)
    
    # Train the model silently (verbose=0) for a few epochs.
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0)
    # Use the last epoch's validation accuracy.
    val_accuracy = history.history['val_accuracy'][-1]
    return 1.0 - val_accuracy

def run_optuna_tuning(X_train, y_train, X_val, y_val, input_shape, nb_classes, n_trials=50):
    """
    Execute an Optuna study to optimize model hyperparameters.
    
    Parameters:
        X_train, y_train: Training data and labels.
        X_val, y_val: Validation data and labels.
        input_shape (tuple): Input shape for the model.
        nb_classes (int): Number of output classes.
        n_trials (int): Number of trials to run.
    
    Returns:
        dict: Best hyperparameters found.
    """
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, input_shape, nb_classes),
                   n_trials=n_trials)
    best_params = study.best_trial.params
    print("Best hyperparameters from Optuna:", best_params)
    return best_params


# ==================== Main Function ====================

def main():
    """
    Main function to load data, preprocess, tune hyperparameters (optional),
    train the EEGNet model, evaluate, save the model, and produce performance reports and plots.
    """
    # List of CSV file paths (make sure these files exist in the same directory or provide full paths)
    '''
    filepaths = [
        "Data_Gtec/adc_dos_left.csv",
        "Data_Gtec/adc_dos_right.csv",
        "Data_Gtec/adc_tres_left.csv",
        "Data_Gtec/adc_uno_left.csv",
        "Data_Gtec/adc_uno_right.csv"
    ]
    '''
    filepaths = [
        "Data_Gtec/adc_1_left.csv",
        "Data_Gtec/adc_1_right.csv",
        "Data_Gtec/adc_2_left.csv",
        "Data_Gtec/adc_2_right.csv",
        "Data_Gtec/adc_3_left.csv",
    ]
    # Lists to hold processed epochs and corresponding labels.
    all_epochs = []
    all_labels = []
    
    # Configuration parameters.
    sample_rate = 250            # Sampling frequency in Hz.
    settling_time_seconds = 10   # Discard first 10 seconds (10 * 250 = 2500 samples).
    epoch_length_seconds = 2     # Each epoch is 2 seconds long (2 * 250 = 500 samples).
    
    # Process each file.
    for filepath in filepaths:
        print(f"Processing file: {filepath}")
        # Load data from CSV.
        df = load_csv_file(filepath)
        # Discard the settling period.
        df_clean = discard_settling_period(df, settling_time_seconds, sample_rate, counter_column='counter')
        # Extract EEG channels (first 8 columns).
        eeg_data = extract_eeg_channels(df_clean)
        # Segment the continuous data into fixed-length epochs.
        epochs = segment_epochs(eeg_data, epoch_length_seconds, sample_rate)
        # Discard neutral epochs: keep every other epoch (assumes first epoch is the movement).
        action_epochs = epochs[::2]
        
        # Determine the label based on the filename.
        filename_lower = filepath.lower()
        if "left" in filename_lower:
            label = 0  # Label 0 for left-hand movement.
        elif "right" in filename_lower:
            label = 1  # Label 1 for right-hand movement.
        else:
            print(f"Skipping file {filepath}: Could not determine label.")
            continue
        
        # Create a label array for all action epochs in this file.
        labels = np.full((action_epochs.shape[0],), label)
        
        # Append processed epochs and labels to the overall list.
        all_epochs.append(action_epochs)
        all_labels.append(labels)
    
    # Ensure that we have processed data.
    if not all_epochs:
        raise ValueError("No valid data was processed. Check file names and data integrity.")
    
    # Concatenate epochs and labels from all files.
    X = np.concatenate(all_epochs, axis=0)  # Shape: (n_epochs, 500, n_channels)
    y = np.concatenate(all_labels, axis=0)
    
    print(f"Total epochs: {X.shape[0]}, Labels shape: {y.shape}")
    
    # Prepare data for EEGNet:
    # EEGNet expects input shape: (channels, samples, 1). Our data is (n_epochs, samples, n_channels).
    X = np.transpose(X, (0, 2, 1))  # New shape: (n_epochs, n_channels, samples)
    X = X[..., np.newaxis]          # Final shape: (n_epochs, n_channels, samples, 1)
    
    print(f"Data shape for EEGNet: {X.shape}")
    
    # Split data into training, validation, and test sets.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    print("Training set shape:", X_train.shape)
    print("Validation set shape:", X_val.shape)
    print("Test set shape:", X_test.shape)
    
    # Define input shape and number of classes for the model.
    input_shape = X_train.shape[1:]  # e.g., (n_channels, 500, 1)
    nb_classes = 2                  # Two classes: left and right.
    
    # ----------------------------
    # Choose hyperparameter tuning method.
    # Set use_optuna to True to run hyperparameter tuning using Optuna;
    # set it to False to use fixed hyperparameters.
    use_optuna = False
    
    if use_optuna:
        print("Starting hyperparameter tuning using Optuna...")
        best_params = run_optuna_tuning(X_train, y_train, X_val, y_val, input_shape, nb_classes, n_trials=30)
        dropoutRate = best_params['dropoutRate']
        kernLength = best_params['kernLength']
        F1 = best_params['F1']
        D = best_params['D']
        F2 = best_params['F2']
    else:
        print("Using fixed hyperparameters...")
        dropoutRate = 0.15464003272972487
        kernLength = 190
        F1 = 4
        D = 2
        F2 = 10
    
    print(f"Selected hyperparameters:\n dropoutRate: {dropoutRate}\n kernLength: {kernLength}\n F1: {F1}\n D: {D}\n F2: {F2}")
    
    # Create the EEGNet model with the selected hyperparameters.
    model = create_eegnet_model(dropoutRate, kernLength, F1, D, F2, input_shape, nb_classes)
    
    # Optionally, print the model summary.
    model.summary()
    
    # Train the model on the full training data.
    history = model.fit(
        X_train_full, y_train_full,
        validation_data=(X_test, y_test),  # Here we use the test set for validation.
        epochs=40,
        batch_size=10,
        verbose=1
    )
    
    # Evaluate the trained model on the test set.
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")
    
    # ----------------------------
    # Generate the Classification Report Table
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    report = classification_report(y_test, y_pred_classes, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    print("\nClassification Report:")
    print(report_df.to_string())
    
    # ----------------------------
    # Plot Learning Curves for Model Performance
    
    # Plot Training & Validation Accuracy
    plt.figure(figsize=(8,6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
    
    # Plot Training & Validation Loss
    plt.figure(figsize=(8,6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
    
    # Save the trained model to disk.
    model.save("eeg_model_new_active_4.h5")
    print("Model saved as 'eeg_model_active_4.h5'.")

if __name__ == '__main__':
    # Optional: Disable oneDNN optimizations if needed.
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    main()