# Instrcutions

1. Install our fast-coref fork and put its correct path into `config/common.yaml` (or see the yaml file in `config/machine/`)
2. Put the `unlabeled_pool_5k` and `conll` folders into `/resources/active_learning/`

## Iteration Loop

1. Specify the current iter's num and the expected sampling num in `/src/config/active_learning.yaml`
   1. The first `current_iter` should be `0`.
2. main_part1_sampling.ipynb: model inference and unlabeled data sampling
3. Using BRAT for manual annotation
4. main_part2_training.ipynb: labeled data resolution and building training data
5. Model training following `fast-coref`'s instrction
6. Repeat 1-5

## Pipeline

iter(0): Model(init)  ->  Sampling ->  Labeling  ->  Training  ->  Model(0)
iter(1): Model(0)  ->  Sampling ->  Labeling  ->  Training  ->  Model(1)
iter(2): Model(1)  ->  Sampling ->  Labeling  ->  Training  ->  Model(2)
iter(n): Model(n-1)  ->  Sampling ->  Labeling  ->  Training  ->  Model(n)

## config
We are using a 
Create a config file at `src/config/remote_server/brat.yaml`, with the following config:

```
brat:
  hostname: ???
  username: ???
  password: ???
  port: 22
```