# Linguistic pre-processing

Process the all the documents w.r.t:

1. linguistic pre-processing
2. Coreference Resolution baselines

## Requirements

### Install spaCy 3.5

Using the requirement.txt to install the spacy en_core_web_md model will cause error.

"ERROR: Could not find a version that satisfies the requirement en-core-web-md==3.5.0 (from versions: none)
ERROR: No matching distribution found for en-core-web-md==3.5.0"

So we comment out the model from the txt file, and ask the user install it using the following commands:

```bash
pip install spacy
python -m spacy download en_core_web_md
```

### Install CoreNLP

- Download CoreNLP from: <https://stanfordnlp.github.io/CoreNLP/>
- Install CoreNLP following the instruction: <https://stanfordnlp.github.io/CoreNLP/download.html#getting-a-copy>
  - Install Java
  - Setup CLASSPATH
- Make sure you can start the CoreNLP server. But shutdown the server before running our script, as the script will start a new server automatically.

## Run script

```bash
cd src/nlp_ensemble
python process_mimic_cxr.py
```