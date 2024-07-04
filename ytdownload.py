import os
import tkinter as tk
from tkinter import ttk, messagebox
from pytube import YouTube
from moviepy.editor import *

class YouTubeAudioDownloaderGUI:
    def __init__(self, master):
        self.master = master
        master.title("YouTube Audio Downloader")
        master.geometry("400x200")

        self.label = ttk.Label(master, text="Enter YouTube URL:")
        self.label.pack(pady=10)

        self.url_entry = ttk.Entry(master, width=50)
        self.url_entry.pack(pady=10)

        self.download_button = ttk.Button(master, text="Download Audio", command=self.download_audio)
        self.download_button.pack(pady=10)

        self.status_label = ttk.Label(master, text="")
        self.status_label.pack(pady=10)

    def download_audio(self):
        url = self.url_entry.get()
        if not url:
            messagebox.showerror("Error", "Please enter a YouTube URL")
            return

        try:
            self.status_label.config(text="Downloading...")
            self.master.update()

            # Download the video
            yt = YouTube(url)
            stream = yt.streams.get_highest_resolution()
            video_filename = "temp_video.mp4"
            stream.download(filename=video_filename)

            # Extract the audio
            video = VideoFileClip(video_filename)
            audio = video.audio
            audio.write_audiofile("audio_example.mp3")

            # Delete the downloaded video file
            os.remove(video_filename)

            self.status_label.config(text="Audio downloaded as 'audio_example.mp3'")
        except Exception as e:
            self.status_label.config(text="")
            messagebox.showerror("Error", str(e))

def main():
    root = tk.Tk()
    app = YouTubeAudioDownloaderGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()