# ECG Processing and Featue Extraction

This repo will include all the ECG feature extraction methods followed by ECG processing as part of ACRIBIS (WP4 module in co-ordination with WP3 module).
1. **QRS Beat Detection** - use a QRS detector to compute an average beat.
     -  neurokit provided by neurokit2 packages
     -  xqrs provided by wfdb package
2. **ECG Delineation** - to detect P/QRS/T onset/peak/offset.
3. **End-to-end models processing raw ECGs**

## Refer to `neurokit_workflow/README.md` for the pipeline setup and documentation