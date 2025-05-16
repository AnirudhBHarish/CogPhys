# Cognitive Load Classification Scripts

This directory contains code for classifying cognitive load using physiological signals.

## Overview

The code implements both machine learning (ML) and deep learning (DL) approaches for classifying cognitive load (high/low) using:
1. Remote PPG signals
2. Remote Respiration signals
3. Blink markers

## Experiment Configurations

Three experiment configurations are supported:
1. `remote_ppg_contact_resp`: Remote PPG + Contact Resp
2. `remote_ppg_remote_resp`: Remote PPG + Remote Resp 
3. `remote_ppg_remote_resp_blink`: Remote PPG + Remote Resp + Blink Markers

## Files

- `data_loader.py`: Data loading and preprocessing functions
- `train_ml_models.py`: Machine learning models pipeline (RF, GB, SVM, LR)
- `train_dl_models.py`: Deep learning models pipeline (CNN, LSTM, ResNet)
- `run_experiments.py`: Script to run all experiments

## Running the Code

To run all experiments:

```bash
python run_experiments.py --all
```

To run only machine learning models:

```bash
python run_experiments.py --ml
```

To run only deep learning models:

```bash
python run_experiments.py --dl
```

To run a specific experiment:

```bash
python run_experiments.py --exp remote_ppg_remote_resp_blink
```

## Results

Results are saved in the `results` directory with timestamps. Each run produces:
- Performance metrics (accuracy, F1, precision, recall)
- Trained models (for DL models)
- Log files

## Data Requirements

The code expects the following data files:
- `./data/rppg_waveforms.pkl`: Remote PPG signals (30 Hz, 3600 samples)
- `./data/resp_fusion_waveforms.pkl`: Remote respiration signals (15 Hz, 1800 samples)
- `./data/eos_dict.pkl`: Blink markers (30 Hz, 3600 samples)

These files should contain dictionaries with keys in the format `vXX_taskname`. 