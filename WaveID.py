import numpy as np
from rtlsdr import RtlSdr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU
import time
from scipy import signal
import sys
import os
import logging
from datetime import datetime
import json

# Configuration dictionary
CONFIG = {
    'sdr': {
        'sample_rate': 2.4e6,  # Hz
        'default_freq': 89.7e6,  # Hz
        'gain': 35.0,  # dB
        'n_samples': 1024,
        'num_captures': 3,  # Number of captures to average
        'settling_time': 0.1,  # seconds
    },
    'signal': {
        'filter': {
            'low_cut': 10e3,  # Hz
            'high_cut': 100e3,  # Hz
            'order': 3,
        },
        'min_snr_db': 10.0,  # Minimum SNR for reliable classification
    },
    'model': {
        'input_shape': (32, 32, 1),
        'path': 'model/trained_model.h5',
    },
    'display': {
        'update_interval': 0.1,  # seconds
        'snr_threshold_good': 15.0,  # dB
        'snr_threshold_marginal': 10.0,  # dB
        'confidence_threshold': 0.7,
    }
}

# Modulation classes matching the trained model
MODULATION_CLASSES = ['4ASK', 'BPSK', 'QPSK', '16PSK', '16QAM', 'FM', 'AM-DSB-WC', '32APSK']

class SignalQualityError(Exception):
    """Exception raised for poor signal quality."""
    pass

def setup_logging():
    """Configure logging system."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"waveid_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def calculate_snr(samples):
    """Calculate Signal-to-Noise Ratio in dB."""
    # Use power method for SNR estimation
    signal_power = np.mean(np.abs(samples)**2)
    noise_power = np.var(np.abs(samples))
    if noise_power == 0:
        return float('-inf')
    return 10 * np.log10(signal_power / noise_power)

def initialize_sdr(freq, config):
    """Initialize and configure the SDR device with error handling."""
    try:
        sdr = RtlSdr()
        sdr.sample_rate = config['sdr']['sample_rate']
        sdr.center_freq = freq
        sdr.gain = config['sdr']['gain']
        
        # Test read to verify device is working
        test_samples = sdr.read_samples(1024)
        if len(test_samples) != 1024:
            raise RuntimeError("Failed to read expected number of samples")
        
        return sdr
    except Exception as e:
        logging.error(f"Failed to initialize SDR: {str(e)}")
        raise

def capture_and_process_samples(sdr, config, logger):
    """Capture and process IQ samples with quality checks."""
    samples_list = []
    snr_values = []
    
    for _ in range(config['sdr']['num_captures']):
        samples = sdr.read_samples(config['sdr']['n_samples'] * 2)
        current_snr = calculate_snr(samples)
        snr_values.append(current_snr)
        
        if current_snr < config['signal']['min_snr_db']:
            logger.warning(f"Low SNR detected: {current_snr:.1f} dB")
        
        # Apply bandpass filter
        nyq = config['sdr']['sample_rate'] / 2
        low = config['signal']['filter']['low_cut'] / nyq
        high = config['signal']['filter']['high_cut'] / nyq
        b, a = signal.butter(config['signal']['filter']['order'], [low, high], btype='band')
        filtered_samples = signal.filtfilt(b, a, samples)
        
        samples_list.append(filtered_samples)
        time.sleep(config['sdr']['settling_time'])
    
    # Average the samples and SNR
    avg_samples = np.mean(samples_list, axis=0)
    avg_snr = np.mean(snr_values)
    
    # Prepare I/Q channels for model
    I_samples = np.real(avg_samples)
    Q_samples = np.imag(avg_samples)
    
    # Normalize
    max_val = max(np.max(np.abs(I_samples)), np.max(np.abs(Q_samples)))
    I_samples = I_samples / (max_val + 1e-10)
    Q_samples = Q_samples / (max_val + 1e-10)
    
    # Reshape for model input
    I_channel = I_samples[:1024].reshape(1, 32, 32, 1)
    Q_channel = Q_samples[:1024].reshape(1, 32, 32, 1)
    
    return I_channel, Q_channel, avg_snr

def load_model_with_custom_objects(config, logger):
    """Load the trained model with proper custom objects handling."""
    if not os.path.exists(config['model']['path']):
        raise FileNotFoundError(f"Model file not found: {config['model']['path']}")
    
    try:
        custom_objects = {
            'LeakyReLU': LeakyReLU,
            'leaky_re_lu': LeakyReLU(alpha=0.1)
        }
        model = load_model(config['model']['path'], custom_objects=custom_objects)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def get_signal_quality_indicator(snr, confidence, config):
    """Return a signal quality indicator based on SNR and confidence."""
    if snr >= config['display']['snr_threshold_good'] and confidence >= config['display']['confidence_threshold']:
        return "✓✓"  # Strong signal
    elif snr >= config['display']['snr_threshold_marginal']:
        return "✓"   # Acceptable signal
    else:
        return "✗"   # Poor signal

def update_display(freq, snr, modulation, confidence, quality_indicator):
    """Update the display with current status and signal quality."""
    status = (
        f"\rFreq: {freq/1e6:6.3f} MHz | "
        f"SNR: {snr:5.1f} dB | "
        f"Quality: {quality_indicator} | "
        f"{modulation:8s} ({confidence:5.3f})"
    )
    print(status, end='', flush=True)

def main():
    """Main function to run the WaveID system."""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting WaveID system")
    
    # Parse command line argument for frequency
    if len(sys.argv) > 1:
        try:
            freq = float(sys.argv[1]) * 1e6  # Convert MHz to Hz
        except ValueError:
            logger.warning(f"Invalid frequency format. Using default {CONFIG['sdr']['default_freq']/1e6} MHz")
            freq = CONFIG['sdr']['default_freq']
    else:
        freq = CONFIG['sdr']['default_freq']
    
    # Clear screen and show header
    os.system('clear' if os.name == 'posix' else 'cls')
    print(f"WaveID System - Analyzing frequency {freq/1e6:.6f} MHz")
    print("Press Ctrl+C to stop\n")
    
    try:
        model = load_model_with_custom_objects(CONFIG, logger)
        sdr = initialize_sdr(freq, CONFIG)
        logger.info("System initialized successfully")
        
        while True:
            try:
                # Capture and process signal
                I_channel, Q_channel, snr = capture_and_process_samples(sdr, CONFIG, logger)
                
                # Make prediction
                predictions = model.predict([I_channel, Q_channel], verbose=0)
                predicted_class = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class]
                modulation = MODULATION_CLASSES[predicted_class]
                
                # Get signal quality indicator
                quality_indicator = get_signal_quality_indicator(snr, confidence, CONFIG)
                
                # Update display
                update_display(freq, snr, modulation, confidence, quality_indicator)
                
                time.sleep(CONFIG['display']['update_interval'])
                
            except SignalQualityError as e:
                logger.warning(str(e))
                continue
            except Exception as e:
                logger.error(f"Error during signal processing: {str(e)}")
                continue
    
    except KeyboardInterrupt:
        logger.info("Stopping WaveID system")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
    finally:
        try:
            sdr.close()
            logger.info("SDR device closed")
        except:
            pass

if __name__ == "__main__":
    main()
