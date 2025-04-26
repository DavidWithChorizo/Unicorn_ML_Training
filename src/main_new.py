#!/usr/bin/env python
"""
new_main.py

This script loads EEG CSV files, preprocesses the data, discards the neutral epochs,
and trains an EEGNet model to classify left and right hand movements.
It also optionally runs hyperparameter tuning with Optuna and plots the tuning progress.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, DepthwiseConv2D,
                                     Activation, AveragePooling2D, Dropout, SeparableConv2D,
                                     Flatten, Dense)
from tensorflow.keras.constraints import max_norm
import optuna


# ==================== Data Loading / Preprocessing ====================

def load_csv_file(filepath):
    df = pd.read_csv(filepath, header=None)
    df.columns = ['eeg1','eeg2','eeg3','eeg4','eeg5','eeg6','eeg7','eeg8','counter','timestamp']
    return df

def discard_settling_period(df, settling_time_seconds, sample_rate=250, counter_column='counter'):
    num_samples_to_discard = int(settling_time_seconds * sample_rate)
    df['norm_counter'] = df[counter_column] - df[counter_column].min()
    return df[df['norm_counter'] >= num_samples_to_discard].reset_index(drop=True)

def extract_eeg_channels(df, eeg_columns=None):
    if eeg_columns is None:
        eeg_columns = list(range(8))
    return df.iloc[:, eeg_columns].values

def segment_epochs(eeg_data, epoch_length_seconds, sample_rate=250):
    epoch_length_samples = int(epoch_length_seconds * sample_rate)
    n_samples = eeg_data.shape[0]
    n_epochs = n_samples // epoch_length_samples
    eeg_data = eeg_data[:n_epochs * epoch_length_samples]
    return eeg_data.reshape(n_epochs, epoch_length_samples, eeg_data.shape[1])


# ==================== EEGNet Model Definition ====================

def create_eegnet_model(dropoutRate, kernLength, F1, D, F2, input_shape, nb_classes):
    input1 = Input(shape=input_shape)
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

    x = SeparableConv2D(F2, (1, 16), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 8))(x)
    x = Dropout(dropoutRate)(x)

    x = Flatten()(x)
    x = Dense(nb_classes, kernel_constraint=max_norm(0.5))(x)
    output = Activation('softmax')(x)

    model = Model(inputs=input1, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# ==================== Optuna Hyperparameter Tuning ====================

def objective(trial, X_train, y_train, X_val, y_val, input_shape, nb_classes, epochs=20, batch_size=32):
    dropoutRate = trial.suggest_float('dropoutRate', 0.2, 0.4)
    kernLength = trial.suggest_int('kernLength', 50, 75, step=5)
    F1 = trial.suggest_int('F1', 2, 8, step=2)
    D = trial.suggest_int('D', 2, 4)
    F2 = trial.suggest_int('F2', 12, 18, step=2)

    model = create_eegnet_model(dropoutRate, kernLength, F1, D, F2, input_shape, nb_classes)
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0)
    val_accuracy = history.history['val_accuracy'][-1]
    return 1.0 - val_accuracy

def run_optuna_tuning(X_train, y_train, X_val, y_val, input_shape, nb_classes, n_trials=50):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda t: objective(t, X_train, y_train, X_val, y_val, input_shape, nb_classes),
                   n_trials=n_trials)
    best_params = study.best_trial.params
    print("Best hyperparameters from Optuna:", best_params)
    return study, best_params


# ==================== Main Function ====================

def main():
    # --- load & preprocess data ---
    filepaths = [
        "Data_Gtec/adc_last_left_1.csv", "Data_Gtec/adc_last_left_2.csv",
        "Data_Gtec/adc_last_left_3.csv", "Data_Gtec/adc_last_left_4.csv",
        "Data_Gtec/adc_last_right_1.csv","Data_Gtec/adc_last_right_2.csv",
        "Data_Gtec/adc_last_right_3.csv","Data_Gtec/adc_last_right_4.csv",
    ]
    all_epochs, all_labels = [], []
    sample_rate = 250
    settling_time_seconds = 10
    epoch_length_seconds = 2

    for fp in filepaths:
        print(f"Processing {fp}")
        df = load_csv_file(fp)
        df_clean = discard_settling_period(df, settling_time_seconds, sample_rate)
        eeg_data = extract_eeg_channels(df_clean)
        epochs = segment_epochs(eeg_data, epoch_length_seconds, sample_rate)
        action_epochs = epochs[::2]
        label = 0 if 'left' in fp.lower() else 1
        all_epochs.append(action_epochs)
        all_labels.append(np.full((action_epochs.shape[0],), label))

    X = np.concatenate(all_epochs, axis=0)
    y = np.concatenate(all_labels, axis=0)
    X = np.transpose(X, (0, 2, 1))[..., np.newaxis]
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.35, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.35, random_state=42, stratify=y_train_full
    )

    input_shape = X_train.shape[1:]
    nb_classes = 2

    # --- Optuna or fixed hyperparameters ---
    use_optuna = True
    if use_optuna:
        print("Starting hyperparameter tuning using Optuna...")
        study, best_params = run_optuna_tuning(
            X_train, y_train, X_val, y_val, input_shape, nb_classes, n_trials=50
        )
        dropoutRate = best_params['dropoutRate']
        kernLength  = best_params['kernLength']
        F1          = best_params['F1']
        D           = best_params['D']
        F2          = best_params['F2']

        # --- Plot Optuna optimization history ---
        trial_nums = [t.number for t in study.trials]
        values     = [t.value  for t in study.trials]
        best_so_far = []
        current_best = float('inf')
        for v in values:
            current_best = min(current_best, v)
            best_so_far.append(current_best)

        plt.figure(figsize=(8, 5))
        plt.plot(trial_nums, values, marker='o', linestyle='-', label='Trial Value')
        plt.plot(trial_nums, best_so_far, marker='.', linestyle='--', label='Best So Far')
        plt.xlabel('Trial Number')
        plt.ylabel('Objective (1 - val_accuracy)')
        plt.title('Optuna Trial Progression')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    else:
        print("Using fixed hyperparameters...")
        dropoutRate = 0.35095908673483583
        kernLength  = 70
        F1          = 8
        D           = 2
        F2          = 14

    print(f"Selected hyperparameters:\n dropoutRate={dropoutRate}\n kernLength={kernLength}\n F1={F1}\n D={D}\n F2={F2}")

    # --- build, train, evaluate EEGNet ---
    model = create_eegnet_model(dropoutRate, kernLength, F1, D, F2, input_shape, nb_classes)
    model.summary()
    history = model.fit(
        X_train_full, y_train_full,
        validation_data=(X_test, y_test),
        epochs=55, batch_size=18, verbose=1
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")

    # --- classification report ---
    y_pred = np.argmax(model.predict(X_test), axis=1)
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    print("\nClassification Report:")
    print(report_df.to_string())

    # --- learning curves ---
    plt.figure(figsize=(8,6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.legend(loc='upper left'); plt.grid(True); plt.show()

    plt.figure(figsize=(8,6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(loc='upper left'); plt.grid(True); plt.show()

    # --- save model ---
    model.save("eeg_model_new_active_version_2.h5")
    print("Model saved as 'eeg_model_new-active-version_2.h5'.")


if __name__ == '__main__':
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    main()