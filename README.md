# CODA



This repo contains codes for the following paper:

Jiaao Chen, Diyi Yang: Simple Conversational Data Augmentation for Semi-supervised Abstractive Conversation Summarization. EMNLP 2021

If you would like to refer to it, please cite the paper mentioned above.

## Natural Language Undertanding 
### Prerequisite: 
* CUDA, cudnn
* Python 3.7
* PyTorch 1.4.0

### Pre-training the Utterance Generation Model
1. Install Huggingface Transformers according to the instructions here: https://github.com/huggingface/transformers.

2. Train the utterance generation model:
```python
>>> chmod +x pre-training.sh
>>> ./pre-training.sh
```


### Run CODA
1. Install Huggingface Transformers according to the instructions here: https://github.com/huggingface/transformers.


2.Train model with the Semi-CODA data augmentation strategies:
```python
>>> chmod +x sf_train.sh
>>> ./sf_train.sh
```


# Acknowledgement

This repository is built upon https://github.com/dinghanshen/Cutoff


