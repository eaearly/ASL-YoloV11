import cv2
import time
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
from PIL import Image, ImageTk
from ultralytics import YOLO


class MinimalStyle:
    """Minimal color palette and styling"""
    BG_COLOR = "#ffffff"  # White background
    ACCENT_COLOR = "#2D3436"  # Dark gray for text
    HIGHLIGHT_COLOR = "#0984E3"  # Blue for highlights
    TEXT_COLOR = "#2D3436"  # Dark gray for text
    BORDER_COLOR = "#DFE6E9"  # Light gray for borders


class ModernButton(tk.Button):
    """Custom button with minimal design"""

    def __init__(self, master, **kwargs):
        super().__init__(
            master,
            bg=MinimalStyle.BG_COLOR,
            fg=MinimalStyle.TEXT_COLOR,
            activebackground=MinimalStyle.HIGHLIGHT_COLOR,
            activeforeground=MinimalStyle.BG_COLOR,
            relief="flat",
            borderwidth=0,
            padx=20,
            pady=10,
            cursor="hand2",
            **kwargs
        )
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self.config(bg=MinimalStyle.HIGHLIGHT_COLOR, fg=MinimalStyle.BG_COLOR)

    def on_leave(self, e):
        self.config(bg=MinimalStyle.BG_COLOR, fg=MinimalStyle.TEXT_COLOR)


class ASLDetectorGUI:
    def __init__(self, window, model_path, conf_threshold=0.25):
        """Initialize ASL detector with minimal GUI"""
        self.window = window
        self.window.title("ASL Detector")
        self.window.geometry("1200x800")
        self.window.configure(bg=MinimalStyle.BG_COLOR)

        # Initialize model
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.cap = None
        self.is_running = False

        # Configure fonts
        self.title_font = tkFont.Font(family="Helvetica", size=16, weight="bold")
        self.text_font = tkFont.Font(family="Helvetica", size=12)

        self.setup_gui()

    def setup_gui(self):
        """Setup minimal GUI elements"""
        # Main container
        self.main_container = tk.Frame(self.window, bg=MinimalStyle.BG_COLOR)
        self.main_container.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        # Video frame
        self.video_frame = tk.Frame(
            self.main_container,
            bg=MinimalStyle.BG_COLOR,
            highlightbackground=MinimalStyle.BORDER_COLOR,
            highlightthickness=1
        )
        self.video_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 20))

        self.video_label = tk.Label(
            self.video_frame,
            bg=MinimalStyle.BG_COLOR
        )
        self.video_label.pack(expand=True, fill=tk.BOTH)

        # Control panel
        self.control_panel = tk.Frame(
            self.main_container,
            bg=MinimalStyle.BG_COLOR,
            width=300
        )
        self.control_panel.pack(side=tk.RIGHT, fill=tk.Y)
        self.control_panel.pack_propagate(False)

        # Title
        tk.Label(
            self.control_panel,
            text="ASL Detection",
            font=self.title_font,
            bg=MinimalStyle.BG_COLOR,
            fg=MinimalStyle.TEXT_COLOR
        ).pack(pady=(0, 20))

        # Controls
        self.setup_controls()

        # Results area
        self.setup_results_area()

    def setup_controls(self):
        """Setup minimal control elements"""
        # Control container
        controls = tk.Frame(
            self.control_panel,
            bg=MinimalStyle.BG_COLOR
        )
        controls.pack(fill=tk.X, pady=10)

        # Start/Stop button
        self.start_button = ModernButton(
            controls,
            text="Start Detection",
            font=self.text_font,
            command=self.toggle_detection
        )
        self.start_button.pack(fill=tk.X, pady=(0, 20))

        # Confidence threshold
        tk.Label(
            controls,
            text="Confidence Threshold",
            font=self.text_font,
            bg=MinimalStyle.BG_COLOR,
            fg=MinimalStyle.TEXT_COLOR
        ).pack(anchor='w')

        # Custom styled slider
        self.confidence_slider = ttk.Scale(
            controls,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            value=self.conf_threshold
        )
        self.confidence_slider.pack(fill=tk.X, pady=(5, 20))

    def setup_results_area(self):
        """Setup minimal results display"""
        # Results container
        results_container = tk.Frame(
            self.control_panel,
            bg=MinimalStyle.BG_COLOR
        )
        results_container.pack(fill=tk.BOTH, expand=True)

        # Results title
        tk.Label(
            results_container,
            text="Detection Results",
            font=self.text_font,
            bg=MinimalStyle.BG_COLOR,
            fg=MinimalStyle.TEXT_COLOR
        ).pack(anchor='w', pady=(0, 10))

        # Results text area
        self.results_text = tk.Text(
            results_container,
            height=12,
            font=self.text_font,
            bg=MinimalStyle.BG_COLOR,
            fg=MinimalStyle.TEXT_COLOR,
            relief="flat",
            wrap=tk.WORD,
            padx=10,
            pady=10
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # Add border to text area
        self.results_text.configure(
            highlightbackground=MinimalStyle.BORDER_COLOR,
            highlightthickness=1
        )

        # FPS display
        self.fps_label = tk.Label(
            self.control_panel,
            text="FPS: 0",
            font=self.text_font,
            bg=MinimalStyle.BG_COLOR,
            fg=MinimalStyle.TEXT_COLOR
        )
        self.fps_label.pack(pady=10)

    def update_frame(self):
        """Update video frame and run detection"""
        if self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # Mirror frame
                frame = cv2.flip(frame, 1)

                # Run detection
                results = self.model.predict(frame, conf=self.confidence_slider.get())
                result = results[0]

                # Process detections
                detected_letters = []
                for box in result.boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if confidence >= 0.60:
                        predicted_letter = self.model.names[class_id]
                        detected_letters.append(f"Letter {predicted_letter}: {confidence:.2f}")
                        # Minimal green box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, predicted_letter, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Update results display with minimal styling
                self.results_text.delete(1.0, tk.END)
                for letter in detected_letters:
                    self.results_text.insert(tk.END, f"{letter}\n")

                # Convert frame to PhotoImage
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_tk = ImageTk.PhotoImage(image=frame_pil)

                # Update video display
                self.video_label.configure(image=frame_tk)
                self.video_label.image = frame_tk

            # Schedule next update
            self.window.after(10, self.update_frame)

    def toggle_detection(self):
        """Toggle detection with UI updates"""
        if self.is_running:
            self.is_running = False
            self.start_button.config(text="Start Detection")
            self.release_camera()
        else:
            self.is_running = True
            self.start_button.config(text="Stop Detection")
            self.initialize_camera()
            self.update_frame()

    def initialize_camera(self):
        """Initialize camera capture"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open camera")

    def release_camera(self):
        """Release camera resources"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None

    def __del__(self):
        """Destructor to ensure camera release"""
        self.release_camera()


def main():
    root = tk.Tk()
    app = ASLDetectorGUI(root, 'model/best.pt')
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {str(e)}")