# GET
This repository includes the code and demo of our research "Introduce Graph Context into Language Models through Parameter-Efficient Fine-Tuning for Lexical Relation Mining".

## Environment Requirements
```
numpy>=1.25.1
torch>=1.13.1
pandas>=2.0.3
tqdm==4.65.0
sklearn>=1.3.0
datasets>=3.1.0
transformers>=4.30.2
networkx>=3.0
dgl>=2.4.0
tqdm>=4.66.6
```

## Datasets
We have uploaded the datasets used for the LE experiments, and the datasets for LRC can be accessed by specifying the datapath to the corresponding Huggingface repository. Detailed links are provided in the paper.


## Training and Evaluation
The scripts train_lrc.py, train_causal.py, and train_le.py are used to run the sequence classification experiments for LRC, the Instruction-Tuning experiments for LRC, and the LE experiments, respectively. Detailed descriptions of the hyperparameter settings can be found in the main text of the paper.


