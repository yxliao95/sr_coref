# Rule-based resolvation of MIMIC-CXR raw reports

Check the config in `/config/data_preprocessing/mimic-cxr.yaml`, `config/common.yaml` and `config/machine/xxx.yaml`

Download and copy `manually_processed_records.json` to `/data_preprocessing/resources` if you want to get the ultimately preprocessed data.

Modify `/config/data_preprocessing/db.yaml` if you want mysql output.

## Requirements

```
pip install hydra-core --upgrade
pip install pathspec
pip install tqdm
```

## Run

```bash
python preprocess_mimic_cxr.py
```

It takes approximately 10 minutes to complete. By the end, you will get a json file with all documents separated by sections.
