# Instrcutions

1. Install our fast-coref fork and put its correct path into `config/common.yaml` (or see the yaml file in `config/machine/`)
2. Put the `unlabeled_pool_5k` and `conll` folders into `/resources/active_learning/`
3. 

# Loop

1. sampling：模型预测+采样
3. process_brat_annotation：生成待注释数据
4. BRAT：人工注释数据
5. process_brat_annotation：解析注释数据
6. build_training_data：生成训练数据
7. 重复1

## config

Create a config file at `src/config/remote_server/brat.yaml`, with the following config:

```
brat:
  hostname: ???
  username: ???
  password: ???
  port: 22
```