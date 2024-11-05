import numpy as np
from rtlsdr import RtlSdr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU
import time

# Constants (adjust if necessary)
SAMPLE_RATE = 2.4e6  # Sampling rate in Hz
CENTER_FREQUENCY = 100.7e6  # Frequency to listen to (adjust to your target frequency)
N_SAMPLES = 1024  # Number of IQ samples to capture
INPUT_SHAPE = (32, 32, 1)  # Model's expected input shape per channel

# Define the modulation classes
SELECTED_MODULATION_CLASSES = ['4ASK', 'BPSK', 'QPSK', '16PSK', '16QAM', 'FM', 'AM-DSB-WC', '32APSK']

def load_model_with_custom_objects():
    """Load the trained model with proper custom objects handling."""
    try:
        custom_objects = {
            'LeakyReLU': LeakyReLU,
            'leaky_re_lu': LeakyReLU(alpha=0.1)
        }
        return load_model('model/trained_model.h5', custom_objects=custom_objects)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

# Initialize RTL-SDR device
def capture_samples(sdr, num_samples):
    """Capture IQ samples from the SDR device."""
    try:
        iq_samples = sdr.read_samples(num_samples)
        return iq_samples
    except Exception as e:
        print(f"Error capturing samples: {str(e)}")
        raise

def preprocess_samples(iq_samples):
    """
    Preprocess the IQ samples to match the model's expected input format.
    Returns separate I and Q channels formatted for the model.
    """
    try:
        # Split into I and Q components
        I_samples = np.real(iq_samples)
        Q_samples = np.imag(iq_samples)

        # Normalize each channel independently
        I_samples = I_samples / (np.max(np.abs(I_samples)) + 1e-10)
        Q_samples = Q_samples / (np.max(np.abs(Q_samples)) + 1e-10)

        # Reshape into 32x32 arrays
        I_channel = I_samples[:1024].reshape(1, 32, 32, 1)
        Q_channel = Q_samples[:1024].reshape(1, 32, 32, 1)

        return I_channel, Q_channel
    except Exception as e:
        print(f"Error preprocessing samples: {str(e)}")
        raise

def predict_modulation(model, I_channel, Q_channel):
    """
    Predict modulation using the pre-trained model with separate I/Q inputs.
    """
    try:
        # Make prediction using both channels
        predictions = model.predict([I_channel, Q_channel], verbose=0)
        
        # Get the predicted class and confidence
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]

        return SELECTED_MODULATION_CLASSES[predicted_class], confidence
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        raise

def initialize_sdr(frequency, sample_rate):
    """Initialize and configure the SDR device."""
    try:
        sdr = RtlSdr()
        sdr.sample_rate = sample_rate
        sdr.center_freq = frequency
        sdr.gain = 'auto'
        return sdr
    except Exception as e:
        print(f"Error initializing SDR: {str(e)}")
        raise

def main(frequency):
    """Main function to run the WaveID system."""
    print("Initializing WaveID system...")
    
    try:
        # Load the model
        model = load_model_with_custom_objects()
        print("Model loaded successfully")

        # Initialize SDR
        sdr = initialize_sdr(frequency, SAMPLE_RATE)
        print(f"SDR initialized - Listening on {frequency/1e6:.1f} MHz")

        while True:
            try:
                # Capture IQ samples
                iq_samples = capture_samples(sdr, N_SAMPLES)

                # Preprocess the samples
                I_channel, Q_channel = preprocess_samples(iq_samples)

                # Predict modulation
                modulation, confidence = predict_modulation(model, I_channel, Q_channel)

                # Print results
                print(f"Predicted modulation: {modulation} with confidence: {confidence:.2f}")
                
                # Add a small delay to prevent overwhelming the SDR
                time.sleep(1)

            except KeyboardInterrupt:
                print("\nStopping WaveID system...")
                break
            except Exception as e:
                print(f"Error during processing: {str(e)}")
                print("Continuing to next sample...")
                continue

    except Exception as e:
        print(f"Fatal error: {str(e)}")
    finally:
        try:
            sdr.close()
            print("SDR device closed")
        except:
            pass

if __name__ == "__main__":
    main(CENTER_FREQUENCY)
