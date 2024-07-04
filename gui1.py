import tkinter as tk
from tkinter import ttk
import pygame
import os

class StemPlayerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stem Player")
        
        # Initialize pygame
        pygame.mixer.init()
        
        # Create audio players
        self.tracks = ['drums', 'other', 'vocals', 'bass']
        self.players = {}
        self.sliders = {}
        
        # Load audio files
        self.audio_dir = 'C:\\Users\\harsa\\OneDrive\\Desktop\\python\\stemplayer\\audio_output\\audio_example'
        self.load_audio_files()
        
        # Create GUI
        self.create_widgets()
        
        # Initialize play/pause state
        self.playing = False
        
    def load_audio_files(self):
        for track in self.tracks:
            file_path = os.path.join(self.audio_dir, f'{track}.wav')
            try:
                self.players[track] = pygame.mixer.Sound(file_path)
            except pygame.error:
                print(f"Error loading {track}.wav")
                self.players[track] = None
        
    def create_widgets(self):
        for i, track in enumerate(self.tracks):
            frame = ttk.Frame(self.root)
            frame.grid(row=i, column=0, padx=10, pady=10, sticky="ew")
            
            ttk.Label(frame, text=track.capitalize()).grid(row=0, column=0, padx=5, pady=5)
            slider = ttk.Scale(frame, from_=0, to=1, orient="horizontal", command=lambda x, t=track: self.set_volume(t, x))
            slider.set(1)  # Set initial volume to maximum
            slider.grid(row=0, column=1, padx=5, pady=5)
            self.sliders[track] = slider
        
        # Create play/pause button
        self.play_pause_button = ttk.Button(self.root, text="Play", command=self.play_pause_audio)
        self.play_pause_button.grid(row=len(self.tracks), column=0, padx=10, pady=10, sticky="ew")
        
    def play_pause_audio(self):
        if not self.playing:
            self.playing = True
            self.play_pause_button.config(text="Pause")
            for track, player in self.players.items():
                if player:
                    player.play(loops=-1)  # Loop indefinitely
        else:
            self.playing = False
            self.play_pause_button.config(text="Play")
            pygame.mixer.pause()
            
    def set_volume(self, track, value):
        if self.players[track]:
            self.players[track].set_volume(float(value))

if __name__ == "__main__":
    root = tk.Tk()
    app = StemPlayerApp(root)
    root.mainloop()
