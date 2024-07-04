import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import os
import shutil

class StemPlayerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stem Player")
        
        # Variables to store input file path
        self.input_file_path = tk.StringVar()
        
        # Initialize GUI
        self.create_widgets()
        
    def create_widgets(self):
        # Label and Entry for input file path
        input_frame = ttk.Frame(self.root)
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        ttk.Label(input_frame, text="Select Input Audio File:").grid(row=0, column=0, padx=5, pady=5)
        self.input_entry = ttk.Entry(input_frame, textvariable=self.input_file_path, width=50)
        self.input_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Button(input_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5, pady=5)
        
        # Button to process stems
        action_frame = ttk.Frame(self.root)
        action_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        
        ttk.Button(action_frame, text="Process Stems", command=self.process_stems).grid(row=0, column=0, padx=5, pady=5)
        
    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3;*.wav")])
        if file_path:
            # Copy the selected file to the target directory
            target_dir = r'C:\Users\harsa\OneDrive\Desktop\python\stemplayer'
            target_file = os.path.join(target_dir, 'audio_example.wav')
            try:
                shutil.copy(file_path, target_file)
                self.input_file_path.set(target_file)
            except IOError as e:
                messagebox.showerror("Error", f"Error copying file: {e}")
    
    def process_stems(self):
        input_file = self.input_file_path.get()
        
        if not input_file:
            messagebox.showerror("Error", "Please select an input audio file.")
            return
        
        # Run spleeter command
        try:
            subprocess.run(['spleeter', 'separate', '-o', 'audio_output', '-p', 'spleeter:4stems', input_file], check=True)
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Error running spleeter: {e}")
            return
        
        messagebox.showinfo("Success", "Stems processed successfully!")

if __name__ == "__main__":
    root = tk.Tk()
    app = StemPlayerApp(root)
    root.mainloop()
