# ECG Processing and Feature Extraction

This package performs ECG processing on custom datasets followed by feature extraction.
1. **QRS Beat Detection** - use a QRS detector to compute an average beat.
     -  neurokit provided by neurokit2 packages
     -  xqrs provided by wfdb package
2. **ECG Delineation** - to detect P/QRS/T onset/peak/offset.
3. **End-to-end pipeline processing raw ECGs**

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
   cd ecg_processing

2. Install the package:
   ```bash
   pip install .

3. To run the script:
   ```bash
   process_ecg_data

## Configuration

Edit the `config.json` file to specify dataset paths and output directories.

   
