import tkinter as tk
from tkinter import ttk
import random
import os
import pandas as pd
import numpy as np
import threading
import serial
import csv
from tensorflow.keras.models import load_model  # type: ignore
from datetime import datetime

# ---------------------
# Global Serial & Data Settings
# ---------------------
SERIAL_PORT = '/dev/tty.usbmodem103'  # Change to your actual serial port if needed.
BAUD_RATE = 115200
FLUSH_INTERVAL = 10
DATA_FOLDER = "Data_Gtec"  # Use same folder name as before (or change as desired)
os.makedirs(DATA_FOLDER, exist_ok=True)

# ---------------------
# File & Data Loading Functions
# ---------------------
def get_next_csv_filename(directory, base_name="adc_last", extension="csv"):
    """
    Returns the next available filename in the directory.
    E.g. If adc_data_1.csv exists, returns adc_data_2.csv, etc.
    """
    n = 1
    while True:
        filename = f"{base_name}_{n}.{extension}"
        full_path = os.path.join(directory, filename)
        if not os.path.exists(full_path):
            return full_path
        n += 1

def load_csv_file(filepath):
    """
    Load a CSV file without headers and assign default column names.
    Assumes CSV has 10 columns.
    """
    try:
        # Read file assuming no header row.
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
    # Assign alternating labels: movement for even and neutral for odd indices.
    return np.array([movement_label if i % 2 == 0 else neutral_label for i in range(n_epochs)])

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
        self.geometry("1600x900")
        self.configure(background="#ECF0F1")

        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TFrame', background="#ECF0F1")
        style.configure('TLabel', background="#ECF0F1", foreground="#2C3E50")
        style.configure('Header.TLabel', font=("Helvetica", 36, "bold"))
        style.configure('SubHeader.TLabel', font=("Helvetica", 28))
        style.configure('TButton', font=("Helvetica", 18), padding=8)

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
        train_btn = ttk.Button(self, text="Training",
                               command=lambda: parent.show_frame(TrainingPage))
        train_btn.pack(pady=20)
        accuracy_btn = ttk.Button(self, text="Check Accuracy",
                                  command=lambda: parent.show_frame(AccuracyPage))
        accuracy_btn.pack(pady=20)

class TrainingPage(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        top_frame = ttk.Frame(self)
        top_frame.pack(pady=20)
        self.info_label = ttk.Label(top_frame, text="Press 'Start Training' to begin.", style='SubHeader.TLabel')
        self.info_label.pack()

        self.start_button = ttk.Button(top_frame, text="Start Training", command=self.start_training)
        self.start_button.pack(pady=10)

        middle_frame = ttk.Frame(self)
        middle_frame.pack(pady=10)
        self.canvas_width = 1400
        self.canvas_height = 600
        self.canvas = tk.Canvas(middle_frame, width=self.canvas_width, height=self.canvas_height, bg="white", highlightthickness=0)
        self.canvas.pack()

        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(pady=20)
        self.back_button = ttk.Button(bottom_frame, text="Back to Main Menu",
                                      command=lambda: parent.show_frame(StartPage))
        self.back_button.pack()

        # Training parameters
        self.total_instruction_pairs = 50  # 50 movement + 10 neutral phases.
        self.current_action = 0
        # Instead of launching an external script, we record serial data in a thread:
        self.serial_thread = None
        self.stop_event = None
        self.output_file = None

    def start_training(self):
        self.start_button.config(state=tk.DISABLED)
        self.parent.direction = random.choice(["left", "right"])
        self.info_label.config(text="Calm down and relax...")

        # Set up serial data recording (using auto-incremented numbering).
        self.output_file = get_next_csv_filename(DATA_FOLDER)
        self.stop_event = threading.Event()
        self.serial_thread = threading.Thread(target=self.read_serial_data, args=(self.output_file, self.stop_event))
        self.serial_thread.start()
        print("Started data streaming to", self.output_file)

        # 10-second calm period before instructions begin.
        self.after(10000, self.run_next_block)

    def run_next_block(self):
        if self.current_action >= self.total_instruction_pairs * 2:
            # Finished training: stop serial recording thread.
            if self.stop_event:
                self.stop_event.set()
            if self.serial_thread:
                self.serial_thread.join()
            self.info_label.config(text=f"Training complete. Data saved as:\n{self.output_file}")
            self.start_button.config(state=tk.NORMAL)
            return

        if self.current_action % 2 == 0:
            self.animate_arrow()
        else:
            self.canvas.delete("all")
            self.info_label.config(text="Neutral. Stay relaxed.")
            self.after(2000, self.run_next_block)

        self.current_action += 1

    def animate_arrow(self):
        self.canvas.delete("all")
        direction = self.parent.direction
        self.info_label.config(text=f"{direction.upper()} movement")
        start_x = self.canvas_width if direction == "left" else 0
        end_x = 0 if direction == "left" else self.canvas_width
        arrow_char = "←" if direction == "left" else "→"
        y_pos = self.canvas_height // 2

        arrow_id = self.canvas.create_text(start_x, y_pos, text=arrow_char,
                                           font=("Helvetica", 80), fill="black")

        duration = 2000  # in milliseconds
        steps = 100
        dx = (end_x - start_x) / steps
        delay = duration // steps

        def step(count):
            if count < steps:
                self.canvas.move(arrow_id, dx, 0)
                self.after(delay, lambda: step(count + 1))
            else:
                self.after(0, self.run_next_block)

        step(0)

    def read_serial_data(self, output_file, stop_event):
        """
        Open the serial port and continuously record data to output_file until stop_event is set.
        Data is written without a header row so that downstream processing remains consistent.
        """
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
        except Exception as e:
            print(f"Error connecting to serial port: {e}")
            return

        sample_count = 0
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Do NOT write header, to match load_csv_file expectations.
            while not stop_event.is_set():
                try:
                    line = ser.readline().decode(errors='ignore').strip()
                except Exception as e:
                    print("Error reading serial line:", e)
                    continue
                if line:
                    parts = line.split(',')
                    # Expect at least 8 values for the ADC channels.
                    if len(parts) >= 8:
                        sample_count += 1
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                        # Write the 8 ADC values, a counter and a timestamp.
                        writer.writerow(parts[:8] + [sample_count, timestamp])
                        print(f"Sample {sample_count}: {parts[:8]}")
                        if sample_count % FLUSH_INTERVAL == 0:
                            f.flush()
        ser.close()
        print("Serial port closed.")

class AccuracyPage(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title_label = ttk.Label(self, text="Accuracy Score", style='Header.TLabel')
        self.title_label.pack(pady=40)
        self.accuracy_label = ttk.Label(self, text="", style='SubHeader.TLabel')
        self.accuracy_label.pack(pady=20)
        self.calc_button = ttk.Button(self, text="Calculate Accuracy",
                                      command=self.calculate_accuracy)
        self.calc_button.pack(pady=20)
        self.back_button = ttk.Button(self, text="Back to Main Menu",
                                      command=lambda: parent.show_frame(StartPage))
        self.back_button.pack(pady=20)

    def calculate_accuracy(self):
        try:
            # Get the latest file from the DATA_FOLDER
            latest_file = get_latest_csv_file(DATA_FOLDER)
            filename = os.path.basename(latest_file)
            # Determine a settling time based on the filename (example logic).
            if "test_1" in filename.lower() and "righthand" in filename.lower():
                settling_time = 5
            elif "test_1" in filename.lower() and "lefthand" in filename.lower():
                settling_time = 5
            elif "_demo_" in filename.lower():
                settling_time = 5
            else:
                settling_time = 5

            df = load_csv_file(latest_file)
            df_clean = discard_settling_period(df, settling_time, sample_rate=250, counter_column='counter')
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