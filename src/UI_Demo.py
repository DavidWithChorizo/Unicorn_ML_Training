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
        df.columns = ['eeg1','eeg2','eeg3','eeg4','eeg5','eeg6','eeg7','eeg8','counter','timestamp']
        # Convert all columns to numeric (non-numeric become NaN)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        raise IOError(f"Error reading {filepath}: {e}")

def discard_settling_period(df, settling_time_seconds, sample_rate=250, counter_column='counter'):
    # We assume the counter column is already numeric.
    num_samples_to_discard = int(settling_time_seconds * sample_rate)
    df['norm_counter'] = df[counter_column] - df[counter_column].min()
    print("Normalized counter range:", df['norm_counter'].min(), df['norm_counter'].max())
    df_clean = df[df['norm_counter'] >= num_samples_to_discard].reset_index(drop=True)
    return df_clean

def extract_eeg_channels(df, eeg_columns=None):
    if eeg_columns is None:
        eeg_columns = list(range(8))
    return df.iloc[:, eeg_columns].apply(pd.to_numeric, errors='coerce').values

def segment_epochs(eeg_data, epoch_length_seconds, sample_rate=250):
    epoch_length_samples = int(epoch_length_seconds * sample_rate)
    n_samples = eeg_data.shape[0]
    n_epochs = n_samples // epoch_length_samples
    eeg_data = eeg_data[:n_epochs * epoch_length_samples]
    epochs = np.reshape(eeg_data, (n_epochs, epoch_length_samples, eeg_data.shape[1]))
    return epochs

def get_latest_csv_file(folder_path):
    """Return the most recently modified CSV file from the folder."""
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError("No CSV files found in the folder.")
    full_paths = [os.path.join(folder_path, f) for f in files]
    latest_file = max(full_paths, key=os.path.getmtime)
    return latest_file

def extract_demo_segment(file_path, settling_time, sample_rate=250, segment_duration=25):
    """
    Extract a segment of length 'segment_duration' seconds (e.g., 25 seconds, 6250 rows)
    from the file, starting at a random index that is after the settling time.
    """
    df = load_csv_file(file_path)
    total_rows = len(df)
    required_rows = int(segment_duration * sample_rate)
    start_min = int(settling_time * sample_rate)
    if total_rows < start_min + required_rows:
        raise ValueError("Not enough data in file after settling time for the required segment.")
    start_index = random.randint(start_min, total_rows - required_rows)
    df_segment = df.iloc[start_index : start_index + required_rows].copy()
    return df_segment

# ---------------------
# Tkinter UI Code
# ---------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EEG Training and Accuracy")
        self.geometry("1600x900")
        self.configure(background="#ECF0F1")
        
        # Configure ttk styles.
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TFrame', background="#ECF0F1")
        style.configure('TLabel', background="#ECF0F1", foreground="#2C3E50")
        style.configure('Header.TLabel', font=("Helvetica", 36, "bold"))
        style.configure('SubHeader.TLabel', font=("Helvetica", 28))
        style.configure('TButton', font=("Helvetica", 18), padding=8)
        
        # This attribute stores the training direction ("left" or "right").
        self.direction = None  
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
        
        # Top frame: instructions.
        top_frame = ttk.Frame(self)
        top_frame.pack(pady=20)
        self.info_label = ttk.Label(top_frame, text="Press 'Start Training' to begin.", style='SubHeader.TLabel')
        self.info_label.pack()
        
        # Middle frame: canvas for arrow animation.
        middle_frame = ttk.Frame(self)
        middle_frame.pack(pady=10)
        self.canvas_width = 1400
        self.canvas_height = 600
        self.canvas = tk.Canvas(middle_frame, width=self.canvas_width, height=self.canvas_height, bg="white", highlightthickness=0)
        self.canvas.pack()
        
        # Bottom frame: control buttons.
        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(pady=20)
        self.start_button = ttk.Button(bottom_frame, text="Start Training", command=self.initiate_training)
        self.start_button.pack(side="left", padx=10)
        self.back_button = ttk.Button(bottom_frame, text="Back to Main Menu", command=lambda: parent.show_frame(StartPage))
        self.back_button.pack(side="left", padx=10)
        
        self.action_count = 0

    def initiate_training(self):
        # Randomly decide the movement direction.
        direction = random.choice(["left", "right"])
        self.parent.direction = direction
        self.start_button.config(state="disabled")
        self.info_label.config(text=f"Settling period: please wait 5 seconds... ({direction.upper()})")
        # Start training after a 5-second delay.
        self.after(5000, self.start_training)
        
    def start_training(self):
        self.info_label.config(text=f"Training Session: {self.parent.direction.upper()} movement arrows")
        self.action_count = 0
        # Begin arrow animations.
        self.run_next_action()
        
    def run_next_action(self):
        if self.action_count < 10:
            self.animate_arrow()
        else:
            self.info_label.config(text="Training complete. Please return to main menu.")
            
    def animate_arrow(self):
        self.canvas.delete("all")
        direction = self.parent.direction
        center_x = self.canvas_width / 2
        y_pos = self.canvas_height / 2
        # Set start and end positions based on direction.
        if direction == "right":
            start_x = center_x
            end_x = self.canvas_width
            arrow_char = "→"
        else:  # left
            start_x = center_x
            end_x = 0
            arrow_char = "←"
        arrow_id = self.canvas.create_text(start_x, y_pos, text=arrow_char, font=("Helvetica", 80), fill="black")
        duration = 2000  # Arrow movement lasts 2 seconds.
        steps = 100
        dx = (end_x - start_x) / steps
        delay = duration // steps
        
        def step(count):
            if count < steps:
                self.canvas.move(arrow_id, dx, 0)
                self.after(delay, lambda: step(count + 1))
            else:
                self.action_count += 1
                # Insert a 2-second neutral pause (clear canvas) before the next arrow.
                self.canvas.delete("all")
                self.after(2000, self.run_next_action)
        step(0)

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
            # Get the chosen movement direction from the training session.
            movement = self.parent.direction  # "left" or "right"
            if movement is None:
                self.accuracy_label.config(text="No training session direction set.")
                return
            
            # List all files in the data folder whose names match the movement.
            data_folder = "Data_Gtec"
            all_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
            # Filter: for movement, file name must include movement AND (if it contains "open", we use settling time 5)
            filtered_files = [f for f in all_files if movement in f.lower()]
            if not filtered_files:
                self.accuracy_label.config(text=f"No {movement} training files found.")
                return
            
            # Randomly select one file from the filtered list.
            selected_file = random.choice(filtered_files)
            file_path = os.path.join(data_folder, selected_file)
            
            # Determine settling time:
            # If the file name contains "open", use a settling time of 5 seconds; otherwise, use a default (if any).
            if "open" in selected_file.lower():
                settling_time = 5
            else:
                settling_time = 5  # You can change this default if needed.
            print(f"Selected file: {selected_file} with settling time: {settling_time} seconds.")
            
            # Extract a 25-second segment from the file, starting at a random point after the settling time.
            df_segment = extract_demo_segment(file_path, settling_time, sample_rate=250, segment_duration=25)
            # Save the extracted segment as a new file without a header.
            now_str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            demo_filename = f"openBCI_{movement}_training_demo_{now_str}.csv"
            demo_filepath = os.path.join(data_folder, demo_filename)
            df_segment.to_csv(demo_filepath, index=False, header=False)
            print(f"Demo segment saved to {demo_filepath}")
            
            # Process the extracted segment:
            # Discard the first 5 seconds (1250 rows); remaining = 25s - 5s = 20 seconds.
            discard_rows = int(5 * 250)
            df_used = df_segment.iloc[discard_rows:].copy()
            if len(df_used) < 5000:
                self.accuracy_label.config(text="Extracted segment is too short after discarding settling period.")
                return
            
            # Extract EEG data.
            eeg_data = extract_eeg_channels(df_used)
            # Split the 20 seconds (5000 rows) into 10 epochs of 2 seconds each (500 rows per epoch).
            epochs = segment_epochs(eeg_data, epoch_length_seconds=2, sample_rate=250)
            if epochs.shape[0] != 10:
                self.accuracy_label.config(text="Incorrect number of epochs extracted.")
                return
            
            # Create labels: all epochs should be the same (0 for left, 1 for right).
            label_value = 0 if movement == "left" else 1
            y = np.full((epochs.shape[0],), label_value)
            
            # Prepare X for the model: expected shape (epochs, channels, samples, 1)
            X = np.transpose(epochs, (0, 2, 1))
            X = X[..., np.newaxis]
            
            # Load the pre-trained ML model.
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