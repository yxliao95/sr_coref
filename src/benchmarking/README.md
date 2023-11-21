# Comparative benchmark

## Hugging Face NeuralCoref: 

ref: https://github.com/huggingface/neuralcoref

1. conda env
    - conda create --name coref_benchmark python=3.6 -y (3.9 is not working)
    - conda activate coref_benchmark
    - pip install notebook
    - pip install jupyter
2. neuralCoref
    - pip install neuralcoref --no-binary neuralcoref
    - pip install -U spacy==2.1.0
    - python -m spacy download en_core_web_sm

## CorefQA (unavailable)

ref: https://github.com/ShannonAI/CorefQA, https://github.com/colinsongf/CorefQA

1. conda env
    - conda create --name coref_benchmark_qa python=3.6 -y
    - conda activate coref_benchmark_qa
    - pip install notebook
    - pip install jupyter
2. corefqa
    - pip install --upgrade pip setuptools==44.1.0
    - pip install -r requirements.txt
3. setup
    - ./setup_all.sh
    - ./optimization.py: `from radam import RAdam` -> `import RAdam`
    - ./coref_ops.py: `tf.load_op_library("./coref_kernels.so")` -> Use absolute file path
    - ./util.py: `ConfigFactory.parse_file("experiments.conf")[name]` under `else` statement -> Use absolute file path
    - ./corefqa.py `self.get_span_embmax_top_antecedents` -> `self.get_span_emb`
    - ./experiments.cong:
        - model_type = corefqa

## CAW-coref 

ref: https://github.com/kareldo/wl-coref

1. conda env
    - conda create -y --name wl-coref python=3.7 openjdk perl
    - conda activate wl-coref
    - python -m pip install -r requirements.txt
    - pip install notebook 
    - pip install jupyter
2. ./config.toml
    - [roberta] bert_model -> Use local_dir if using a offline huggingface model
    - data_dir -> Use the dir of the downloaded checkpoint (.pt)
3. predict
    - python predict.py roberta input.jsonlines output.jsonlines --weights /root/autodl-tmp/hg_offline_models/caw-coref/roberta_release.pt

## allennlp coref

https://demo.allennlp.org/coreference-resolution#:~:text=Coreference%20resolution%20is%20the%20task%20of%20finding%20all,as%20document%20summarization%2C%20question%20answering%2C%20and%20information%20extraction.

1. conda env
    - conda create --name allennlp python=3.7 -y
    - conda activate allennlp
    - pip install allennlp==2.1.0 allennlp-models==2.1.0
    - pip install timm==0.4.5 transformers==4.5.1 packaging==21.3 (for '0.10.1<0.11' error)
    - python -m spacy download en_core_web_sm