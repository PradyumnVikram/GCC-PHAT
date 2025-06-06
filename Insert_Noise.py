import librosa
import numpy as np
import soundfile as sf
import subprocess
import os

# Paths
wav_path = "/home/ramdaftari/code/GCC-PHAT/Test_Audio/output.wav"
mp3_path = "/home/ramdaftari/code/GCC-PHAT/Test_Audio/output.mp3"
source_path = "/home/ramdaftari/code/GCC-PHAT/Test_Audio/B_R-vocals_delayed.mp3"

# Step 1: Load audio
print("Loading audio...")
y, sr = librosa.load(source_path, sr=None)
print(f"Loaded {len(y)} samples at {sr} Hz")

# Step 2: Inject noise for first 2.62 s
duration_sec = 2.62
num_samples = int(duration_sec * sr)
if len(y) < num_samples:
    raise ValueError("Audio is shorter than 2.62 seconds")
print(f"Injecting white noise into first {duration_sec} seconds")

y[:num_samples] = np.random.normal(0, 0.1, num_samples).astype(np.float32)

# Step 3: Write WAV file
sf.write(wav_path, y, sr)
print(f"Wrote WAV: {wav_path} — Exists: {os.path.exists(wav_path)}")

# Step 4: Convert to MP3 using FFmpeg
print("Converting to MP3...")
result = subprocess.run(
    ["ffmpeg", "-y", "-i", wav_path, mp3_path],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Print FFmpeg logs
print("--- FFmpeg STDOUT ---")
print(result.stdout)
print("--- FFmpeg STDERR ---")
print(result.stderr)

# Step 5: Check final MP3
if os.path.exists(mp3_path):
    print(f"MP3 saved at: {mp3_path}")
else:
    print("MP3 file was not created.")
