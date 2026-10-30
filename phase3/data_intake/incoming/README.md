# Incoming native collection

Put actual new JPEG/PNG images and annotation JSON/PNG files under this folder.
Register relative paths in the parent intake CSVs. No images have been added
by the data gate. The validator reads only referenced paths and rejects paths
that leave this directory.

For Kaggle, upload the parent `data_intake` folder as an optional dataset. Keep
`INTAKE_VERSION.json` at its root. An optional `protocol_policy.json` at that
root supplies the actual future prospective policy; the shipped policy is a
draft and cannot authorize training.
