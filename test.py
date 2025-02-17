import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import joblib
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pyaudio
import wave
from playsound import playsound  # Replaced simpleaudio with playsound

# Update dataset path
DATASET_PATH = r"C:\Users\Asus\Downloads\archive\TESS Toronto emotional speech set data"

# Define emotion labels based on folder names
EMOTION_LABELS = {
    'angry': 'Angry',
    'disgust': 'Disgust',
    'fear': 'Fearful',
    'happy': 'Happy',
    'neutral': 'Neutral',
    'pleasant_surprise': 'Surprised',
    'sad': 'Sad'
}

# Function to extract features
def extract_features(file_path):
    y, sr = sf.read(file_path)
    
    if len(y.shape) > 1:  # Convert stereo to mono
        y = np.mean(y, axis=1)
    
    if sr != 16000:  # Resample to 16kHz
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000

    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
    
    return np.hstack([mfccs, chroma, mel, contrast, zcr, rms])

# Load dataset and extract features
features, labels = [], []
for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                file_path = os.path.join(folder_path, file)
                matched = False
                for key in EMOTION_LABELS:
                    if key in folder.lower():
                        emotion = EMOTION_LABELS[key]
                        try:
                            features.append(extract_features(file_path))
                            labels.append(emotion)
                            matched = True
                        except Exception as e:
                            print(f"Error processing {file_path}: {str(e)}")
                        break
                if not matched:
                    print(f"Warning: No emotion label found for {folder}")

df = pd.DataFrame(features)
df['label'] = labels

# Split dataset
X = df.iloc[:, :-1].values
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, "emotion_model.pkl")

# Function to predict emotion
def predict_emotion(file_path):
    features = extract_features(file_path).reshape(1, -1)
    model = joblib.load("emotion_model.pkl")
    prediction = model.predict(features)[0]
    return prediction

# Function to plot waveform and FFT
def plot_waveform_fft(file_path):
    for widget in plot_frame.winfo_children():
        widget.destroy()
    
    y, sr = sf.read(file_path)
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    ax1.plot(y)
    ax1.set_title('Waveform')
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('Amplitude')

    fft_y = np.fft.fft(y)
    fft_freq = np.fft.fftfreq(len(fft_y), 1/sr)
    ax2.plot(fft_freq[:len(fft_freq)//2], np.abs(fft_y)[:len(fft_y)//2])
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Amplitude')

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Function to record audio
def record_audio(output_file, duration=5, sample_rate=16000):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    CHUNK = 1024

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=sample_rate, input=True, frames_per_buffer=CHUNK)

    print("Recording...")
    frames = [stream.read(CHUNK) for _ in range(0, int(sample_rate / CHUNK * duration))]
    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(output_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

# Variable to store last used audio file
last_audio_file = None

# Function to browse and process file
def browse_file():
    global last_audio_file
    file_path = filedialog.askopenfilename(filetypes=[("WAV Files", "*.wav")])
    if not file_path:
        messagebox.showwarning("No File Selected", "Please select a valid WAV file.")
        return

    last_audio_file = file_path
    status_label.config(text="Processing...")
    window.update_idletasks()

    try:
        predicted_emotion = predict_emotion(file_path)
        emotion_label.config(text=f"Predicted Emotion: {predicted_emotion}")
        plot_waveform_fft(file_path)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
    finally:
        status_label.config(text="Ready")

# Function to record and predict
def record_and_predict():
    global last_audio_file
    output_file = "recorded_audio.wav"
    record_audio(output_file, duration=5)
    last_audio_file = output_file

    try:
        predicted_emotion = predict_emotion(output_file)
        emotion_label.config(text=f"Predicted Emotion: {predicted_emotion}")
        plot_waveform_fft(output_file)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Function to play last selected or recorded audio file
def play_audio():
    global last_audio_file
    if last_audio_file and os.path.exists(last_audio_file):
        try:
            playsound(last_audio_file)  # Using playsound instead of simpleaudio
        except Exception as e:
            messagebox.showerror("Error", f"Could not play audio: {str(e)}")
    else:
        messagebox.showwarning("No File", "Please select or record an audio file first.")

# UI setup
window = tk.Tk()
window.title("Emotion Recognition from Audio")
window.geometry("900x700")

title_label = tk.Label(window, text="Emotion Recognition", font=("Helvetica", 16, "bold"), fg="blue")
title_label.pack(pady=10)

button_frame = tk.Frame(window)
button_frame.pack(pady=10)

tk.Button(button_frame, text="Browse Audio", command=browse_file).pack(side=tk.LEFT, padx=10)
tk.Button(button_frame, text="Record", command=record_and_predict).pack(side=tk.LEFT, padx=10)
tk.Button(button_frame, text="Play Audio", command=play_audio).pack(side=tk.LEFT, padx=10)

emotion_label = tk.Label(window, text="Predicted Emotion: ", font=("Helvetica", 14, "bold"))
emotion_label.pack(pady=10)

# Add status label for feedback
status_label = tk.Label(window, text="Ready", font=("Helvetica", 12))
status_label.pack(pady=10)

plot_frame = tk.Frame(window)
plot_frame.pack()

window.mainloop()

