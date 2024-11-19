# ECG Processor

This package processes ECG data and extracts baseline features with related intervals.

## Workflow

1. Load ECG data from a specified dataset.
2. Perform cleaning and preprocessing of ECG signals.
3. Detect R-peaks and delineate ECG signals into their physiological components.
4. Extract features from the cleaned ECG data.
5. Save the extracted features and annotations to output files.

## Installation

1. Clone the repository:
   ```bash
   git@github.com:dieterich-lab/ecg_processing.git
   cd ecg_processing/neurokit_workflow

2. Install the package:
   ```bash
   pip install .

3. To run the script:
   ```bash
   process_ecg_data

## Configuration

Edit the `config.json` file to specify dataset paths and output directories.

   
