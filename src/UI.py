import tkinter as tk
import random
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

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
        # Even indices get the movement label, odd indices get the neutral label.
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
# Tkinter UI Code
# ---------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EEG Training and Accuracy")
        self.geometry("1920x1080")  # Set to a 2K resolution window
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

class StartPage(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        label = tk.Label(self, text="Welcome to EEG Trainer", font=("Helvetica", 40))
        label.pack(pady=50)
        train_btn = tk.Button(self, text="Training", width=30,
                              command=lambda: parent.show_frame(TrainingPage), font=("Helvetica", 20))
        train_btn.pack(pady=30)
        accuracy_btn = tk.Button(self, text="Check Accuracy", width=30,
                                 command=lambda: parent.show_frame(AccuracyPage), font=("Helvetica", 20))
        accuracy_btn.pack(pady=30)

class TrainingPage(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.info_label = tk.Label(self, text="Please relax. Session will start in 5 seconds.",
                                   font=("Helvetica", 32))
        self.info_label.pack(pady=30)
        # Canvas set to full 2K resolution for arrow animation.
        self.canvas_width = 1920
        self.canvas_height = 1080
        self.canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack(pady=20)
        # Back to Main Menu button
        self.back_button = tk.Button(self, text="Back to Main Menu", width=30,
                                     command=lambda: parent.show_frame(StartPage), font=("Helvetica", 20))
        self.back_button.pack(pady=20)
        self.action_count = 0
        # Start training after a 5-second delay.
        self.after(5000, self.start_training)

    def start_training(self):
        # Randomly choose a training direction.
        direction = random.choice(["left", "right"])
        self.parent.direction = direction
        self.info_label.config(text=f"Training Session: {direction.upper()} movement arrows")
        self.action_count = 0
        self.run_next_action()

    def run_next_action(self):
        if self.action_count < 10:
            self.animate_arrow()
        else:
            self.info_label.config(text="Training complete. Please return to main menu.")

    def animate_arrow(self):
        self.canvas.delete("all")
        direction = self.parent.direction
        canvas_width = self.canvas_width
        y_pos = self.canvas_height // 2  # Center the arrow vertically.
        if direction == "left":
            start_x = canvas_width
            end_x = 0
            arrow_char = "←"
        else:
            start_x = 0
            end_x = canvas_width
            arrow_char = "→"
        # Create the arrow with a large font and black fill.
        arrow_id = self.canvas.create_text(start_x, y_pos, text=arrow_char, font=("Helvetica", 100), fill="black")
        duration = 2000  # Each movement lasts 2 seconds.
        steps = 100     # Number of animation steps.
        dx = (end_x - start_x) / steps
        delay = duration // steps

        def step(count):
            if count < steps:
                self.canvas.move(arrow_id, dx, 0)
                self.after(delay, lambda: step(count + 1))
            else:
                self.action_count += 1
                self.after(500, self.run_next_action)  # Brief pause before next action.
        step(0)

class AccuracyPage(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title_label = tk.Label(self, text="Accuracy Score", font=("Helvetica", 40))
        self.title_label.pack(pady=50)
        self.accuracy_label = tk.Label(self, text="", font=("Helvetica", 32))
        self.accuracy_label.pack(pady=30)
        self.calc_button = tk.Button(self, text="Calculate Accuracy", width=30,
                                     command=self.calculate_accuracy, font=("Helvetica", 20))
        self.calc_button.pack(pady=20)
        self.back_button = tk.Button(self, text="Back to Main Menu", width=30,
                                     command=lambda: parent.show_frame(StartPage), font=("Helvetica", 20))
        self.back_button.pack(pady=20)

    def calculate_accuracy(self):
        try:
            # Folder where CSV session files are stored.
            data_folder = "Data_Gtec"
            latest_file = get_latest_csv_file(data_folder)
            # Dictionary mapping specific filenames to settling times.
            settling_times = {
                "UnicornRecorder_06_03_2025_15_23_220_left and neutral 1.csv": 30,
                "UnicornRecorder_06_03_2025_15_30_400_left_and_neutral_2_0.20.csv": 15,
                "UnicornRecorder_06_03_2025_16_25_570_right_1_0.05.csv": 10,
                "UnicornRecorder_06_03_2025_16_32_090_right_2_1.15.csv": 75,
            }
            filename = os.path.basename(latest_file)
            settling_time = settling_times.get(filename, 75)  # Default settling time is 20 seconds.
            print(f"Processing {latest_file} with settling time {settling_time}")
            
            # Run the data preparation pipeline.
            df = load_csv_file(latest_file)
            df_clean = discard_settling_period(df, settling_time, sample_rate=250, counter_column='counter')
            eeg_data = extract_eeg_channels(df_clean)
            # Each epoch is assumed to be 2 seconds long.
            epochs = segment_epochs(eeg_data, epoch_length_seconds=2, sample_rate=250)
            
            # Determine the movement label from the filename.
            if "left" in filename.lower():
                movement_label = 0  # left-hand movement
            elif "right" in filename.lower():
                movement_label = 1  # right-hand movement
            else:
                movement_label = None
            
            if movement_label is not None:
                labels = label_epochs_alternating(epochs, movement_label, neutral_label=2)
            else:
                labels = np.full((epochs.shape[0],), 2)
            
            # Convert the epochs to the shape expected by EEGNet: (epochs, channels, samples, 1)
            X = np.transpose(epochs, (0, 2, 1))
            X = X[..., np.newaxis]
            y = labels
            
            # Load your trained EEG model.
            model = load_model("eeg_model_1.h5")
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