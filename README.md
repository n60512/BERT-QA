# BERT-QA

Building a machine reading comprehension system using pretrained model bert.


## Quick Start

### 1. Prepare your training data and install the package in requirement.txt

### 2. Fine-tune BERT model

```
sh tarin.sh
```

### 3. Interaction

```
sh interaction.sh
```

## Experiment

### Input context format like below：
```
{ "sentence":"YOUR_SENTENCE。", "question":"YOUR_QUESTION"}
```


### The experimental result of F1-measure：
```
Evaluation 100%|███████████████████████████████████| 495/495 [00:05<00:00, 91.41it/s]
Average f1 : 0.5596300989105781
```

### Display Result

![res1](https://i.imgur.com/bxJ0oyV.png)

![res2](https://i.imgur.com/F3mB5jQ.png)

## Model architectures
BERT (from Google) released with the paper[ BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.

## Dataset

In this experiments, we use the datasets from DRCKnowledgeTeam. (https://github.com/DRCKnowledgeTeam/DRCD)