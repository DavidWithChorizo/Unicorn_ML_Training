import tkinter as tk
import random

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EEG Training and Accuracy")
        self.geometry("400x300")
        # Store the chosen training direction ("left" or "right")
        self.direction = None  
        # Create a container for the pages (frames)
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
        label = tk.Label(self, text="Welcome to EEG Trainer", font=("Helvetica", 16))
        label.pack(pady=20)
        train_btn = tk.Button(self, text="Training", width=20,
                              command=lambda: parent.show_frame(TrainingPage))
        train_btn.pack(pady=10)
        accuracy_btn = tk.Button(self, text="Check Accuracy", width=20,
                                 command=lambda: parent.show_frame(AccuracyPage))
        accuracy_btn.pack(pady=10)


class TrainingPage(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.info_label = tk.Label(self, text="Please relax. Session will start in 5 seconds.",
                                   font=("Helvetica", 14))
        self.info_label.pack(pady=20)
        self.canvas = tk.Canvas(self, width=300, height=100, bg="white")
        self.canvas.pack(pady=20)
        self.action_count = 0
        # Start the training after 5 seconds (5000 ms)
        self.after(5000, self.start_training)

    def start_training(self):
        # Randomly choose a direction (50/50 chance)
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
        # Setup initial and target positions for the arrow
        if direction == "left":
            start_x = 300
            end_x = 0
            arrow_char = "←"
        else:
            start_x = 0
            end_x = 300
            arrow_char = "→"
        y_pos = 50
        # Create the arrow on the canvas
        arrow_id = self.canvas.create_text(start_x, y_pos, text=arrow_char, font=("Helvetica", 32))
        duration = 2000  # 2 seconds for the animation
        steps = 100  # total animation steps
        dx = (end_x - start_x) / steps
        delay = duration // steps

        def step(count):
            if count < steps:
                self.canvas.move(arrow_id, dx, 0)
                self.after(delay, lambda: step(count + 1))
            else:
                self.action_count += 1
                # Pause briefly between actions (500ms) before next animation
                self.after(500, self.run_next_action)

        step(0)


class AccuracyPage(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title_label = tk.Label(self, text="Accuracy Score", font=("Helvetica", 16))
        self.title_label.pack(pady=20)
        self.accuracy_label = tk.Label(self, text="", font=("Helvetica", 14))
        self.accuracy_label.pack(pady=20)
        self.calc_button = tk.Button(self, text="Calculate Accuracy", width=20,
                                     command=self.calculate_accuracy)
        self.calc_button.pack(pady=10)
        self.back_button = tk.Button(self, text="Back to Main Menu", width=20,
                                     command=lambda: parent.show_frame(StartPage))
        self.back_button.pack(pady=10)

    def calculate_accuracy(self):
        # Simulate EEG data processing and ML model predictions.
        # In a real application, here you would load EEG data and feed it into your ML model.
        if self.parent.direction is None:
            self.accuracy_label.config(text="No training data available.")
            return

        correct_predictions = 0
        # Simulate 10 decisions; assume 80% chance the ML model correctly predicts the intended direction.
        for _ in range(10):
            # Simulate the ML model's prediction
            prediction = self.parent.direction if random.random() < 0.8 else (
                "right" if self.parent.direction == "left" else "left")
            if prediction == self.parent.direction:
                correct_predictions += 1

        accuracy = (correct_predictions / 10) * 100
        self.accuracy_label.config(text=f"Model Accuracy: {accuracy:.1f}%")

if __name__ == "__main__":
    app = App()
    app.mainloop()