# Memoria Experiment

The directory contains the model architecture, data loader, config files, and training and evaluation script to conduct experiments in my paper. You can reproduce my research by refering to the Memoria paper or develop your own idea on this.

## Package Install

You should install the required packages before running the code.

```sh
$ pip install -r requirements.txt
```

## Structure

```
longseq_formers
├── configs
└── longseq_formers
    ├── data
    ├── dataset
    ├── model
    │   ├── compressive_former
    │   ├── gpt2_with_memoria
    │   ├── infinity_gpt2
    │   ├── memoria_bert
    │   └── memoria_roberta
    └── task
```
- `longseq_formers` directory is main directory for experiment. There are data loaders, task training, and model architectures.
- `configs` directory includes multiple config files for language modeling and synthetic task (sorting) for multiple models.

## Models

You can load modeles from `longseq_formers.model` module regardless of the training or evaluation script.

```python
import torch
from longseq_formers.model import MemoriaBertModel

memoria_bert = MemoriaBertModel.from_pretrained("bert-base-uncased")
input_ids = torch.randint(0, 10, [1,10])
outputs = memoria_bert(input_ids)
```

```python
import torch
from longseq_formers.model import GPT2WithMemoriaLMHeadModel

memoria_gpt2 = GPT2WithMemoriaLMHeadModel.from_pretrained("gpt2")
input_ids = torch.randint(0, 10, [1,10])
outputs = memoria_gpt2(input_ids)
```

## Train

You can train the model with training scripts depending on the task. With the `--help` option, you can see the options for training or evaluation. Because all the datasets except for the sorting task will be loaded from web, you don't have to download the dataset separately.

```sh
$ python train_language_modeling.py --help
usage: train [-h] [--model-config MODEL_CONFIG] [--model MODEL] [--model-type MODEL_TYPE] [--tokenizer TOKENIZER] [--dataset {wikitext103,pg19,enwik8}]
             [--batch-size BATCH_SIZE] [--valid-batch-size VALID_BATCH_SIZE] [--accumulate-grad-batches ACCUMULATE_GRAD_BATCHES] [--max-length MAX_LENGTH] [--epochs EPOCHS]
             [--learning-rate LEARNING_RATE] [--warmup-rate WARMUP_RATE] [--max-grad-norm MAX_GRAD_NORM] [--seed SEED] [--shuffle] [--test-ckpt {best,last}]
             [--output-dir OUTPUT_DIR] [--gpus GPUS] [--logging-interval LOGGING_INTERVAL] [--valid-interval VALID_INTERVAL] [--wandb-run-name WANDB_RUN_NAME]
             [--wandb-entity WANDB_ENTITY] [--wandb-project WANDB_PROJECT]

Train & Test Language Modeling

optional arguments:
  -h, --help            show this help message and exit

Train Parameter:
  --model-config MODEL_CONFIG
                        huggingface model config
  --model MODEL         huggingface model
  --model-type MODEL_TYPE
                        specific model type
  --tokenizer TOKENIZER
                        huggingface tokenizer
  --dataset {wikitext103,pg19,enwik8}
                        dataset name
  --batch-size BATCH_SIZE
                        global training batch size
  --valid-batch-size VALID_BATCH_SIZE
                        validation batch size
  --accumulate-grad-batches ACCUMULATE_GRAD_BATCHES
                        the number of gradident accumulation steps
  --max-length MAX_LENGTH
                        max sequence length
  --epochs EPOCHS       the number of training epochs
  --learning-rate LEARNING_RATE
                        learning rate
  --warmup-rate WARMUP_RATE
                        warmup step rate
  --max-grad-norm MAX_GRAD_NORM
                        maximum gradient norm
  --seed SEED           random seed
  --shuffle             shuffle data order
  --test-ckpt {best,last}
                        checkpoint type for testing

Personal Options:
  --output-dir OUTPUT_DIR
                        output directory path to save artifacts
  --gpus GPUS           the number of gpus, use all devices by default
  --logging-interval LOGGING_INTERVAL
                        logging interval
  --valid-interval VALID_INTERVAL
                        validation interval rate

Wandb Options:
  --wandb-run-name WANDB_RUN_NAME
                        wanDB run name
  --wandb-entity WANDB_ENTITY
                        wanDB entity name
  --wandb-project WANDB_PROJECT
                        wanDB project name
```

```sh
$ python train_language_modeling.py --model gpt2
[2023-09-29 21:31:29,995]  ====== Arguements ======
[2023-09-29 21:31:29,995] model_config             : None
[2023-09-29 21:31:29,995] model                    : gpt2
[2023-09-29 21:31:29,995] model_type               : None
[2023-09-29 21:31:29,995] tokenizer                : None
[2023-09-29 21:31:29,995] dataset                  : wikitext103
[2023-09-29 21:31:29,995] batch_size               : 8
[2023-09-29 21:31:29,995] valid_batch_size         : 1
[2023-09-29 21:31:29,995] accumulate_grad_batches  : 1
[2023-09-29 21:31:29,995] max_length               : 150
[2023-09-29 21:31:29,995] epochs                   : 6
[2023-09-29 21:31:29,995] learning_rate            : 0.0002
...
```
- You can start training simply this command without any download.

```sh
$ python train_language_modeling.py --model-config configs/memoria-gpt2.json --tokenizer gpt2 --output-dir trained-model
[2023-09-29 21:43:27,347] [+] Save output to "trained-model"
[2023-09-29 21:43:27,347]  ====== Arguements ======
[2023-09-29 21:43:27,347] model_config             : configs/memoria-gpt2.json
[2023-09-29 21:43:27,347] model                    : None
[2023-09-29 21:43:27,347] model_type               : None
[2023-09-29 21:43:27,347] tokenizer                : gpt2
[2023-09-29 21:43:27,347] dataset                  : wikitext103
[2023-09-29 21:43:27,347] batch_size               : 8
[2023-09-29 21:43:27,347] valid_batch_size         : 1
[2023-09-29 21:43:27,347] accumulate_grad_batches  : 1
[2023-09-29 21:43:27,347] max_length               : 150
[2023-09-29 21:43:27,347] epochs                   : 6
[2023-09-29 21:43:27,347] learning_rate            : 0.0002
...
```
- You can train MemoriaGPT2 model by adding `--model-type gpt2_with_memoria` option or `--model-config configs/memoria-gpt2.json` simply.
- To save model checkpoint, you can add `--output-dir [OUTPUT-DIR]` option. The model checkpoint will be saved in `trained-model` directory.
- Refer help description and the Memoria paper for detail hyperparameters.

## Evaluation

```sh
$ python eval_language_modeling.py --model trained-model/checkpoint/last.ckpt
[2023-09-29 21:45:03,214]  ====== Arguements ======
[2023-09-29 21:45:03,214] model                    : trained-model/checkpoint/last.ckpt
[2023-09-29 21:45:03,214] tokenizer                : gpt2
[2023-09-29 21:45:03,214] dataset                  : wikitext103
[2023-09-29 21:45:03,214] valid_batch_size         : 1
[2023-09-29 21:45:03,214] max_length               : 512
[2023-09-29 21:45:03,214] seed                     : 42
[2023-09-29 21:45:03,214] [+] Set Random Seed to 42
Global seed set to 42
[2023-09-29 21:45:03,237] [+] GPU: 1
[2023-09-29 21:45:03,237] [+] Load Tokenizer: "gpt2"
```
- You should give save model checkpoint with `--model [MODEL-CHECKPOINT]` option.
