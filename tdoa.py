import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

# Try different audio loading methods
def load_mp3_file(filepath, target_sr=22050):
    """
    Load MP3 file using multiple fallback methods
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None, None
    
    print(f"Attempting to load: {filepath}")
    
    # Method 1: Try librosa
    try:
        import librosa
        audio_data, sample_rate = librosa.load(filepath, sr=target_sr, mono=True)
        print(f"✓ Loaded with librosa: {filepath}")
        print(f"  - Duration: {len(audio_data)/sample_rate:.2f} seconds")
        print(f"  - Sample rate: {sample_rate} Hz")
        print(f"  - Number of samples: {len(audio_data)}")
        return audio_data, sample_rate
    except Exception as e:
        print(f"✗ Librosa failed: {e}")
    
    # Method 2: Try pydub + numpy
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(filepath)
        audio = audio.set_channels(1)  # Convert to mono
        audio = audio.set_frame_rate(target_sr)  # Set sample rate
        
        # Convert to numpy array
        audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
        audio_data = audio_data / (2**15)  # Normalize to [-1, 1]
        
        print(f"✓ Loaded with pydub: {filepath}")
        print(f"  - Duration: {len(audio_data)/target_sr:.2f} seconds")
        print(f"  - Sample rate: {target_sr} Hz")
        print(f"  - Number of samples: {len(audio_data)}")
        return audio_data, target_sr
    except Exception as e:
        print(f"✗ Pydub failed: {e}")
    
    # Method 3: Try scipy with wav conversion
    try:
        from pydub import AudioSegment
        from scipy.io import wavfile
        import tempfile
        
        # Convert MP3 to WAV temporarily
        audio = AudioSegment.from_mp3(filepath)
        audio = audio.set_channels(1).set_frame_rate(target_sr)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            audio.export(temp_wav.name, format='wav')
            sample_rate, audio_data = wavfile.read(temp_wav.name)
            os.unlink(temp_wav.name)  # Clean up temp file
        
        # Normalize
        audio_data = audio_data.astype(np.float32) / (2**15)
        
        print(f"✓ Loaded via WAV conversion: {filepath}")
        print(f"  - Duration: {len(audio_data)/sample_rate:.2f} seconds")
        print(f"  - Sample rate: {sample_rate} Hz")
        print(f"  - Number of samples: {len(audio_data)}")
        return audio_data, sample_rate
    except Exception as e:
        print(f"✗ WAV conversion failed: {e}")
    
    print(f"✗ All methods failed for: {filepath}")
    return None, None

def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT) method.
    '''
    
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    # find max cross correlation index
    shift = np.argmax(cc) - max_shift

    tau = shift / float(interp * fs)
    
    return tau, cc

def preprocess_audio(audio1, audio2, max_length_seconds=30, sample_rate=22050):
    """
    Preprocess audio signals for GCC-PHAT analysis
    """
    # Truncate to max_length if needed (to speed up processing)
    max_samples = int(max_length_seconds * sample_rate)
    
    if len(audio1) > max_samples:
        audio1 = audio1[:max_samples]
        print(f"Truncated audio1 to {max_length_seconds} seconds")
    
    if len(audio2) > max_samples:
        audio2 = audio2[:max_samples]
        print(f"Truncated audio2 to {max_length_seconds} seconds")
    
    # Make both signals the same length
    min_length = min(len(audio1), len(audio2))
    audio1 = audio1[:min_length]
    audio2 = audio2[:min_length]
    
    return audio1, audio2

def find_delay_between_mp3s(file1_path, file2_path, max_tau=5.0, analysis_length=30):
    """
    Find time delay between two MP3 files using GCC-PHAT
    """
    
    print("Loading MP3 files...")
    print("=" * 60)
    
    # Load both MP3 files with same sample rate
    audio1, sr1 = load_mp3_file(file1_path, target_sr=22050)
    audio2, sr2 = load_mp3_file(file2_path, target_sr=22050)
    
    if audio1 is None or audio2 is None:
        print("Could not load audio files. Try installing: pip install pydub")
        return None, None
    
    print(f"\nSample rates - File1: {sr1} Hz, File2: {sr2} Hz")
    
    # Preprocess audio
    print(f"\nPreprocessing audio...")
    audio1, audio2 = preprocess_audio(audio1, audio2, analysis_length, sr1)
    
    # Apply GCC-PHAT
    print(f"\nApplying GCC-PHAT...")
    print(f"Max expected delay: {max_tau} seconds")
    
    delay, correlation = gcc_phat(audio2, audio1, fs=sr1, max_tau=max_tau, interp=16)
    
    # Results
    print(f"\n" + "="*60)
    print(f"RESULTS:")
    print(f"Estimated delay: {delay:.6f} seconds")
    print(f"Estimated delay: {delay*1000:.3f} milliseconds")
    print(f"Estimated delay: {int(delay*sr1)} samples")
    print(f"="*60)
    
    return delay, correlation


    
