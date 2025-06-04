import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np


def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
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

    # Sometimes, there is a 180-degree phase difference between the two microphones.
    # shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)
    
    return tau, cc

# New test function that uses your GCC-PHAT code
def test_gcc_phat_with_synthetic():
    """Test your GCC-PHAT code with synthetic signals"""
    
    print("Testing GCC-PHAT with synthetic signals...")
    print("=" * 50)
    
    # Parameters
    fs = 8000  # Sample rate (Hz)
    duration = 1.0  # Duration in seconds
    t = np.linspace(0, duration, int(fs * duration))
    
    # Create reference signal (chirp works well for testing)
    refsig = signal.chirp(t, f0=100, f1=2000, t1=duration, method='linear')
    
    # Add some realistic noise
    refsig += 0.05 * np.random.randn(len(refsig))
    
    # Test different delays
    test_delays_samples = [10, 25, 50, 100, 200]
    
    for delay_samples in test_delays_samples:
        print(f"\nTesting with {delay_samples} sample delay:")
        
        # Calculate expected delay in seconds
        expected_delay = delay_samples / fs
        
        # Create delayed signal
        sig = np.concatenate([
            np.zeros(delay_samples),  # Add the delay
            refsig,
            np.zeros(100)  # Extra padding
        ])
        
        # Apply your GCC-PHAT function
        estimated_tau, correlation = gcc_phat(sig, refsig, fs, max_tau=0.1)
        
        # Calculate error
        error = abs(estimated_tau - expected_delay)
        
        # Print results
        print(f"  Expected delay: {expected_delay:.6f} seconds")
        print(f"  Estimated delay: {estimated_tau:.6f} seconds")
        print(f"  Error: {error:.6f} seconds")
        print(f"  Accuracy: {(1 - error/expected_delay)*100:.2f}%")
    
    return sig, refsig, correlation

def visualize_gcc_phat_results():
    """Visualize the GCC-PHAT results"""
    
    # Create test signals
    fs = 8000
    duration = 0.5
    t = np.linspace(0, duration, int(fs * duration))
    
    # Reference signal
    refsig = signal.chirp(t, f0=200, f1=1000, t1=duration, method='linear')
    refsig += 0.05 * np.random.randn(len(refsig))
    
    # Delayed signal
    delay_samples = 80
    sig = np.concatenate([np.zeros(delay_samples), refsig, np.zeros(100)])
    
    # Apply GCC-PHAT
    tau, cc = gcc_phat(sig, refsig, fs)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Reference signal
    plt.subplot(4, 1, 1)
    plt.plot(t[:1000], refsig[:1000])
    plt.title('Reference Signal (refsig)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Plot 2: Delayed signal
    plt.subplot(4, 1, 2)
    t_delayed = np.linspace(0, len(sig)/fs, len(sig))
    plt.plot(t_delayed[:1000], sig[:1000])
    plt.title('Delayed Signal (sig)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Plot 3: Correlation function
    plt.subplot(4, 1, 3)
    plt.plot(cc)
    plt.title('GCC-PHAT Correlation Function')
    plt.xlabel('Lag (samples)')
    plt.ylabel('Correlation')
    plt.grid(True)
    
    # Plot 4: Zoomed correlation around peak
    plt.subplot(4, 1, 4)
    peak_idx = np.argmax(cc)
    zoom_range = 200
    start_idx = max(0, peak_idx - zoom_range)
    end_idx = min(len(cc), peak_idx + zoom_range)
    
    plt.plot(range(start_idx, end_idx), cc[start_idx:end_idx])
    plt.axvline(x=peak_idx, color='red', linestyle='--', label=f'Peak at {peak_idx}')
    plt.title('Zoomed Correlation Function Around Peak')
    plt.xlabel('Lag (samples)')
    plt.ylabel('Correlation')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    expected_delay = delay_samples / fs
    print(f"\nResults:")
    print(f"Expected delay: {expected_delay:.6f} seconds")
    print(f"Estimated delay: {tau:.6f} seconds")
    print(f"Error: {abs(tau - expected_delay):.6f} seconds")

# Main execution
if __name__ == "__main__":
    # Run basic tests
    test_gcc_phat_with_synthetic()
    
    print("\n" + "="*50)
    print("Generating visualization...")
    
    # Run visualization
    visualize_gcc_phat_results()
