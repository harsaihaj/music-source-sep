import os
import subprocess

# Assuming you have the audio file in the same directory
audio_file = 'audio_example.wav'

# Create the output directory if it doesn't exist
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Run the spleeter command using subprocess
spleeter_cmd = f"spleeter separate -i {audio_file} -o {output_dir} -p spleeter:4stems"
result = subprocess.run(spleeter_cmd, shell=True, capture_output=True, text=True)

# Print the output and error (if any)
print("Spleeter output:", result.stdout)
print("Spleeter error:", result.stderr)

print("Separation complete. Separated files should be in the 'output' directory.")

# Check if the files exist before trying to play them
from playsound import playsound

stems = ['vocals', 'bass', 'drums', 'other']
for stem in stems:
    file_path = f"{output_dir}/{os.path.splitext(audio_file)[0]}/{stem}.wav"
    if os.path.exists(file_path):
        print(f"Playing {stem}...")
        try:
            playsound(file_path)
        except Exception as e:
            print(f"Error playing {stem}: {e}")
    else:
        print(f"File not found: {file_path}")