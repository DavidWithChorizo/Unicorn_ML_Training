import tkinter as tk
import random

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EEG Training and Accuracy")
        self.geometry("1920x1080")  # Set to 2K resolution
        self.direction = None  # Stores the training direction ("left" or "right")
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
        # Canvas set to 2K resolution
        self.canvas_width = 1200
        self.canvas_height = 900
        self.canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack(pady=20)
        # Back to Main Menu button
        self.back_button = tk.Button(self, text="Back to Main Menu", width=30,
                                     command=lambda: self.parent.show_frame(StartPage), font=("Helvetica", 20))
        self.back_button.pack(pady=20)
        self.action_count = 0
        # Start training after 5 seconds (5000ms)
        self.after(5000, self.start_training)

    def start_training(self):
        # Choose a direction at random (50/50 chance)
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
        # Use the full canvas width
        canvas_width = self.canvas_width
        y_pos = self.canvas_height // 2  # Center vertically
        if direction == "left":
            start_x = canvas_width
            end_x = 0
            arrow_char = "←"
        else:
            start_x = 0
            end_x = canvas_width
            arrow_char = "→"

        # Create the arrow with a large font and explicit fill color for contrast
        arrow_id = self.canvas.create_text(start_x, y_pos, text=arrow_char, font=("Helvetica", 100), fill="black")
        duration = 2000  # 2 seconds for the animation
        steps = 100     # Number of animation steps
        dx = (end_x - start_x) / steps
        delay = duration // steps

        def step(count):
            if count < steps:
                self.canvas.move(arrow_id, dx, 0)
                self.after(delay, lambda: step(count + 1))
            else:
                self.action_count += 1
                # Pause 500ms between actions before the next animation
                self.after(500, self.run_next_action)

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
        # In a real application, this would process the EEG data and run an ML model.
        if self.parent.direction is None:
            self.accuracy_label.config(text="No training data available.")
            return

        correct_predictions = 0
        # Simulate 10 predictions with an assumed 80% accuracy per action.
        for _ in range(10):
            prediction = self.parent.direction if random.random() < 0.8 else (
                "right" if self.parent.direction == "left" else "left")
            if prediction == self.parent.direction:
                correct_predictions += 1

        accuracy = (correct_predictions / 10) * 100
        self.accuracy_label.config(text=f"Model Accuracy: {accuracy:.1f}%")

if __name__ == "__main__":
    app = App()
    app.mainloop()