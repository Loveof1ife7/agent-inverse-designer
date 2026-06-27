# P222 Simple Fast Subset

This folder is a no-copy subset of `P222_paired_dataset_0_99999_20260620` for fast closed-loop iteration.

- `geometry/`: NTFS hardlinks to selected simple geometry txt files.
- `properties/`: NTFS hardlinks to matching stress-strain CSV files.
- `fast_subset_manifest.json`: selected ids and node/edge counts.
- Preprocessed JSONL: `D:\codes\agent-material-windows-lite\train_datas\preprocessed\P222_paired_dataset_simple_fast_20260626\inverse_truss_property_grid_simple_fast.jsonl`

Symbolic links were requested, but Windows denied file symlink creation in this session (`WinError 1314`).
Hardlinks avoid data copies and work with normal file APIs.
