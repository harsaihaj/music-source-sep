import tkinter as tk
from tkinter import ttk
import subprocess

class MainGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Music Source Separation")
        self.root.attributes('-fullscreen', True)
        
        # Set background color
        self.root.configure(bg='#2C3E50')
        
        # Create a main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        # Configure style
        style = ttk.Style()
        style.configure('TLabel', font=('Arial', 24), foreground='#ECF0F1', background='#2C3E50')
        style.configure('TButton', font=('Arial', 14), padding=10)
        
        # Title
        label = ttk.Label(main_frame, text="Music Source Separation")
        label.pack(pady=40)
        
        # Create a frame for buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(expand=True)
        
        # Buttons
        buttons = [
            ("Upload Music", self.open_gui),
            ("Listen to Separated Components", self.open_gui1),
            ("Break Down Drums", self.run_nmf),
            ("Listen to Separated Drums", self.run_gui2),
            ("Download from YouTube", self.open_ytdownload)
        ]
        
        for text, command in buttons:
            btn = ttk.Button(button_frame, text=text, command=command)
            btn.pack(pady=20, padx=50, fill=tk.X)
        
        # Exit button
        exit_button = ttk.Button(main_frame, text="Exit", command=self.root.quit)
        exit_button.pack(pady=40)

    def open_gui(self):
        subprocess.Popen(['python', 'gui.py'])

    def open_gui1(self):
        subprocess.Popen(['python', 'gui1.py'])

    def run_nmf(self):
        subprocess.Popen(['python', 'nmf.py'])

    def run_gui2(self):
        subprocess.Popen(['python', 'gui2.py'])

    def open_ytdownload(self):
        subprocess.Popen(['python', 'ytdownload.py'])

if __name__ == "__main__":
    root = tk.Tk()
    app = MainGUI(root)
    root.mainloop()
