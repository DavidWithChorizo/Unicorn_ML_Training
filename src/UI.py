import tkinter as tk
from tkinter import ttk
import random
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import threading
import time
from datetime import datetime

# ---------------------
# Data Processing Functions
# ---------------------

def load_csv_file(filepath):
    """
    Load a CSV file without headers and assign default column names.
    Assumes CSV has 10 columns.
    """
    try:
        df = pd.read_csv(filepath, header=None)
        df.columns = ['eeg1', 'eeg2', 'eeg3', 'eeg4', 'eeg5', 'eeg6', 'eeg7', 'eeg8', 'counter', 'timestamp']
        return df
    except Exception as e:
        raise IOError(f"Error reading {filepath}: {e}")

def discard_settling_period(df, settling_time_seconds, sample_rate=250, counter_column='counter'):
    # Convert the counter column to numeric to avoid type issues.
    df[counter_column] = pd.to_numeric(df[counter_column], errors='coerce')
    num_samples_to_discard = int(settling_time_seconds * sample_rate)
    df['norm_counter'] = df[counter_column] - df[counter_column].min()
    print("Normalized counter range:", df['norm_counter'].min(), df['norm_counter'].max())
    df_clean = df[df['norm_counter'] >= num_samples_to_discard].reset_index(drop=True)
    return df_clean

def extract_eeg_channels(df, eeg_columns=None):
    if eeg_columns is None:
        eeg_columns = list(range(8))
    return df.iloc[:, eeg_columns].values

def segment_epochs(eeg_data, epoch_length_seconds, sample_rate=250):
    epoch_length_samples = int(epoch_length_seconds * sample_rate)
    n_samples = eeg_data.shape[0]
    n_epochs = n_samples // epoch_length_samples
    eeg_data = eeg_data[:n_epochs * epoch_length_samples]
    epochs = np.reshape(eeg_data, (n_epochs, epoch_length_samples, eeg_data.shape[1]))
    return epochs

def label_epochs_alternating(epochs, movement_label, neutral_label=2):
    n_epochs = epochs.shape[0]
    labels = []
    for i in range(n_epochs):
        if i % 2 == 0:
            labels.append(movement_label)
        else:
            labels.append(neutral_label)
    return np.array(labels)

def get_latest_csv_file(folder_path):
    """Return the most recently modified CSV file from the folder."""
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError("No CSV files found in the folder.")
    full_paths = [os.path.join(folder_path, f) for f in files]
    latest_file = max(full_paths, key=os.path.getmtime)
    return latest_file

# ---------------------
# Tkinter UI Code with Revised Training Trigger and Settling Time Default
# ---------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EEG Training and Accuracy")
        self.geometry("1600x900")
        self.configure(background="#ECF0F1")

        # Configure ttk styles for a modern look.
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TFrame', background="#ECF0F1")
        style.configure('TLabel', background="#ECF0F1", foreground="#2C3E50")
        style.configure('Header.TLabel', font=("Helvetica", 36, "bold"))
        style.configure('SubHeader.TLabel', font=("Helvetica", 28))
        style.configure('TButton', font=("Helvetica", 18), padding=8)

        self.direction = None  # For the training session (left/right)
        self.frames = {}
        for F in (StartPage, TrainingPage, AccuracyPage):
            frame = F(self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

class StartPage(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        label = ttk.Label(self, text="Welcome to EEG Trainer", style='Header.TLabel')
        label.pack(pady=40)
        train_btn = ttk.Button(self, text="Training", command=lambda: parent.show_frame(TrainingPage))
        train_btn.pack(pady=20)
        accuracy_btn = ttk.Button(self, text="Check Accuracy", command=lambda: parent.show_frame(AccuracyPage))
        accuracy_btn.pack(pady=20)

class TrainingPage(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        # Recording variables.
        self.recording = False
        self.record_thread = None
        self.recorded_data = []
        self.sample_counter = 0
        self.sample_rate = 250  # Hz

        # Top frame for info label.
        top_frame = ttk.Frame(self)
        top_frame.pack(pady=20)
        self.info_label = ttk.Label(top_frame, text="Press 'Start Training' to begin.", style='SubHeader.TLabel')
        self.info_label.pack()

        # Middle frame for canvas.
        middle_frame = ttk.Frame(self)
        middle_frame.pack(pady=10)
        self.canvas_width = 1400
        self.canvas_height = 600
        self.canvas = tk.Canvas(middle_frame, width=self.canvas_width, height=self.canvas_height, bg="white", highlightthickness=0)
        self.canvas.pack()

        # Bottom frame for control buttons.
        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(pady=20)
        self.start_button = ttk.Button(bottom_frame, text="Start Training", command=self.initiate_training)
        self.start_button.pack(side="left", padx=10)
        self.back_button = ttk.Button(bottom_frame, text="Back to Main Menu", command=lambda: parent.show_frame(StartPage))
        self.back_button.pack(side="left", padx=10)

        self.action_count = 0

    def initiate_training(self):
        # Disable the start button to prevent multiple clicks.
        self.start_button.config(state="disabled")
        self.info_label.config(text="Settling period: please wait 5 seconds...")
        # Start training after a 5-second delay.
        self.after(5000, self.start_training)

    def start_training(self):
        # Randomly choose a training direction.
        direction = random.choice(["left", "right"])
        self.parent.direction = direction
        self.info_label.config(text=f"Training Session: {direction.upper()} movement arrows")
        self.action_count = 0

        # Start recording streaming data.
        self.start_recording()

        # Begin arrow animations.
        self.run_next_action()

    def run_next_action(self):
        if self.action_count < 10:
            self.animate_arrow()
        else:
            # End of training: stop recording and save the data.
            self.stop_recording()
            self.info_label.config(text="Training complete. Data recorded. Please return to main menu.")

    def animate_arrow(self):
        self.canvas.delete("all")
        direction = self.parent.direction
        if direction == "left":
            start_x = self.canvas_width
            end_x = 0
            arrow_char = "←"
        else:
            start_x = 0
            end_x = self.canvas_width
            arrow_char = "→"
        y_pos = self.canvas_height // 2
        arrow_id = self.canvas.create_text(start_x, y_pos, text=arrow_char, font=("Helvetica", 80), fill="black")
        duration = 2000  # 2 seconds per movement.
        steps = 100
        dx = (end_x - start_x) / steps
        delay = duration // steps

        def step(count):
            if count < steps:
                self.canvas.move(arrow_id, dx, 0)
                self.after(delay, lambda: step(count + 1))
            else:
                self.action_count += 1
                self.after(500, self.run_next_action)
        step(0)

    # --- Recording Methods ---
    def start_recording(self):
        self.recording = True
        self.recorded_data = []  # Clear any previous data.
        self.sample_counter = 0
        self.record_thread = threading.Thread(target=self.record_stream, daemon=True)
        self.record_thread.start()
        print("Recording started...")

    def record_stream(self):
        sample_interval = 1.0 / self.sample_rate
        while self.recording:
            # Simulate 8 EEG channels with random data.
            sample = [random.uniform(-100, 100) for _ in range(8)]
            sample.append(self.sample_counter)  # Counter value.
            sample.append(time.time())          # Timestamp.
            self.recorded_data.append(sample)
            self.sample_counter += 1
            time.sleep(sample_interval)

    def stop_recording(self):
        self.recording = False
        if self.record_thread is not None:
            self.record_thread.join()
        print("Recording stopped.")
        self.save_recorded_data()

    def save_recorded_data(self):
        folder = "Data_Gtec"
        os.makedirs(folder, exist_ok=True)
        now_str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        direction = self.parent.direction if self.parent.direction else "unknown"
        filename = f"UnicornRecorder_{now_str}_{direction}_training.csv"
        filepath = os.path.join(folder, filename)
        df = pd.DataFrame(self.recorded_data, columns=['eeg1','eeg2','eeg3','eeg4','eeg5','eeg6','eeg7','eeg8','counter','timestamp'])
        df.to_csv(filepath, index=False)
        print(f"Recorded data saved to {filepath}")

class AccuracyPage(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title_label = ttk.Label(self, text="Accuracy Score", style='Header.TLabel')
        self.title_label.pack(pady=40)
        self.accuracy_label = ttk.Label(self, text="", style='SubHeader.TLabel')
        self.accuracy_label.pack(pady=20)
        self.calc_button = ttk.Button(self, text="Calculate Accuracy", command=self.calculate_accuracy)
        self.calc_button.pack(pady=20)
        self.back_button = ttk.Button(self, text="Back to Main Menu", command=lambda: parent.show_frame(StartPage))
        self.back_button.pack(pady=20)

    def calculate_accuracy(self):
        try:
            data_folder = "Data_Gtec"
            latest_file = get_latest_csv_file(data_folder)
            # If this is a training file, use a lower default settling time.
            default_settling = 5 if "training" in latest_file.lower() else 75
            settling_times = {
                "UnicornRecorder_06_03_2025_15_23_220_left and neutral 1.csv": 30,
                "UnicornRecorder_06_03_2025_15_30_400_left_and_neutral_2_0.20.csv": 15,
                "UnicornRecorder_06_03_2025_16_25_570_right_1_0.05.csv": 10,
                "UnicornRecorder_06_03_2025_16_32_090_right_2_1.15.csv": 75,
            }
            filename = os.path.basename(latest_file)
            settling_time = settling_times.get(filename, default_settling)
            print(f"Processing {latest_file} with settling time {settling_time}")

            df = load_csv_file(latest_file)
            df_clean = discard_settling_period(df, settling_time, sample_rate=250, counter_column='counter')
            
            if df_clean.empty:
                self.accuracy_label.config(text="Not enough data after discarding settling period.")
                return

            eeg_data = extract_eeg_channels(df_clean)
            epochs = segment_epochs(eeg_data, epoch_length_seconds=2, sample_rate=250)

            if "left" in filename.lower():
                movement_label = 0
            elif "right" in filename.lower():
                movement_label = 1
            else:
                movement_label = None

            if movement_label is not None:
                labels = label_epochs_alternating(epochs, movement_label, neutral_label=2)
            else:
                labels = np.full((epochs.shape[0],), 2)

            X = np.transpose(epochs, (0, 2, 1))
            X = X[..., np.newaxis]
            y = labels

            model = load_model("eeg_model_2.h5")
            predictions = model.predict(X)
            pred_labels = np.argmax(predictions, axis=1)

            correct = np.sum(pred_labels == y)
            accuracy = (correct / len(y)) * 100
            self.accuracy_label.config(text=f"Model Accuracy: {accuracy:.1f}%")
        except Exception as e:
            self.accuracy_label.config(text=f"Error: {str(e)}")

if __name__ == "__main__":
    app = App()
    app.mainloop()