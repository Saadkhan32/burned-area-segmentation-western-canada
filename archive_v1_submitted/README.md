# Archive of the submitted study (version 1.0, July 2026)

Kept for the record; not used by the revised study.

- `scripts/` - the scripts of the submitted repository: Google Earth Engine export of nine-band composites, binary labelling and chip export helpers for ArcGIS Pro, batch inference on validation tiles, evaluation and plotting.
- `README_v1.md`, `requirements_v1.txt` - the README and requirements of version 1.0.
- `audit/` - manifests and scripts of the audit of the submitted chips: `event_manifest.csv` (the 30 exported events with their chip grids), `chip_manifest_trainingexports.csv` (the 8,109 exported chips with footprints), `test_chip_overlap.csv` and `test_event_summary.csv` (every test chip of the submitted Table 7 located in the training exports), and the scripts that produced them and re-scored the archived predictions. The scripts read the archived study folder and the audit folder from the environment variables `SUBMITTED_STUDY_DIR`, `AUDIT_DIR` and `AGENCY_RECORDS_DIR`.
- `submitted_model_files_sha256.txt` - checksums of the five model files of the submitted study.
