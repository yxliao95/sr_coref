# Fine-tuning coreference resolution for different styles of clinical narratives

## Model inference

Our coref model is based on [fast-coref](https://github.com/shtoshni/fast-coref) with [minor modification](https://github.com/liaoooyx/fast-coref). Please follow their instruction for model inference.

Our fine-tuned models are available [here](https://drive.google.com/drive/folders/1ZAVJYo9c5bobNeQdQexlOCoGzhF-u02G?usp=sharing)

## Model training

The training data is currently not available as the MIMIC-CXR data is restricted to sharing with anyone else. We are preparing to share the labelled mimic-cxr coref data on PhysioNet.

### Python requirements

python >= 3.9

```bash
conda create --name sr_coref python=3.9 -y
pip install -r requirements.txt
```

### Python paths

If the scripts fail to import modules, please make sure the following paths are added to the PYTHONPATH environment variable.

```bash
export PYTHONPATH=/path_to/fast-coref/src
export PYTHONPATH=/path_to/sr_coref/src:$PYTHONPATH
```

### Pre-process the MIMIC-CXR data

We split the whole reports into sections.

```bash
cd ../str_rep_coref/src/data_preprocessing
python preprocess_mimic_cxr.py
```

The script output is: /output/mimic_cxr/mimic_cxr_sections.jsonlines

Check the src/data_preprocessing/README.md file for more configuation details.

### Linguistic pre-processing

Check the src/nlp_ensemble/README.md file for details.

### Coreference resolution pre-processing

Check the src/coreference_resolution/README.md file for details.

### Active Learning

Check the src/active_learning/README.md for details.


## The ensemble algorithm

Check the src/nlp_ensemble/README.md for details.

## Cautions

The column of CSV files might not follow the same order. When the reports are being processed by CoreNLP with multiple coref annotators, some of the reports may not be successfully processed in the first round. We will re-run the coref annotators on `unfinished records` in the second round. This will lead to a different order of the columns for those second-round-processed reports. For those disorder reports' sid, you can find them from `/output/nlp_ensamble/run.log or corenlp_unfinished_records.log`

## Config

Please read the [Hydra Docs](https://hydra.cc/docs/intro/) for more details.

## Others

If you are using VSCode, add the following configs to get build-in supports:

.vscode/settings.json:

```json
{
    "python.analysis.extraPaths": [
        "/path_to/git_clone_repos/fast-coref/src",
        "/path_to/str_rep_coref/src",
    ],
    "terminal.integrated.env.linux": {
        "PYTHONPATH": "${workspaceFolder}/src:/path_to/fast-coref/src"
    },
}
```