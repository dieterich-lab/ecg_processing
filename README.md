# ECG Processing and Feature Extraction

This package performs ECG processing on custom datasets followed by feature extraction.
1. **R-Peak Detection** - Detect R-peaks in 12-lead ecgs using Neurokit2 tool.
2. **ECG Delineation** - to detect P/QRS/T onset/peak/offset.
3. **Feature Extraction** - Compute baseline features such as heart rate and ecg-related intervals (PR interval, QT interval, etc.)

## Workflow

* Load ECG data from a specified dataset.
*  Perform cleaning and preprocessing of ECG signals.
*  Detect R-peaks and delineate ECG signals into their physiological components.
*  Extract features from the cleaned ECG data.
*  Save the extracted features and annotations to output files.

## Installation

1. Clone the repository:
   ```bash
   git@github.com:dieterich-lab/ecg_processing.git
   cd ecg_processing

2. Install the package:
   ```bash
   pip install .

3. To run the script:
   ```bash
   python main.py

## Configuration

Edit the `config.json` file to specify dataset paths and output directories.

   
