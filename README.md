# WaveID

A real-time radio signal modulation classifier using RTL-SDR and deep learning.
Designed for use with DeepSig's RadioML 2018.01A dataset.

## Features
- Real-time signal classification
- Supports 8 modulation types: 4ASK, BPSK, QPSK, 16PSK, 16QAM, FM, AM-DSB-WC, 32APSK
- Automatic gain control
- Built with RTL-SDR support

## Requirements
- Python 3.x
- RTL-SDR device
- Required packages: tensorflow, numpy, rtlsdr-python

## Quick Start
1. Install dependencies:
```bash
pip install tensorflow numpy rtlsdr-python
```

2. Connect your RTL-SDR device

3. Run WaveID:
```bash
python WaveID.py
```

## Training
To train a new model on your own dataset:
```bash
python model/train_model.py
